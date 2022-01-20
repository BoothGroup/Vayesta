import logging

import numpy as np

import pyscf.lib
from vayesta.core.util import *
from vayesta.rpa.rirpa import momzero_NI, energy_NI


class ssRIRPA:
    """Approach based on equations expressed succinctly in the appendix of
    Furche, F. (2001). PRB, 64(19), 195120. https://doi.org/10.1103/PhysRevB.64.195120
    WARNING: Should only be used with canonical mean-field orbital coefficients in mf.mo_coeff and RHF.
    """

    def __init__(self, dfmf, rixc=None, log=None, err_tol=1e-6, svd_tol=1e-10):
        self.mf = dfmf
        self.rixc = rixc
        self.log = log or logging.getLogger(__name__)
        self.err_tol = err_tol
        self.svd_tol = svd_tol
        self.e_corr_ss = None

    @property
    def nocc(self):
        return sum(self.mf.mo_occ > 0)

    @property
    def nvir(self):
        return len(self.mf.mo_occ) - self.nocc

    @property
    def naux_eri(self):
        return self.mf.with_df.get_naoaux()

    @property
    def ov(self):
        return self.nocc * self.nvir

    @property
    def ov_tot(self):
        return 2 * self.ov

    @property
    def mo_coeff(self):
        """Occupied MO coefficients."""
        return self.mf.mo_coeff

    @property
    def mo_coeff_occ(self):
        """Occupied MO coefficients."""
        return self.mo_coeff[:, :self.nocc]

    @property
    def mo_coeff_vir(self):
        """Virtual MO coefficients."""
        return self.mo_coeff[:, self.nocc:]

    @property
    def mo_energy(self):
        return self.mf.mo_energy

    @property
    def mo_energy_occ(self):
        return self.mo_energy[:self.nocc]

    @property
    def mo_energy_vir(self):
        return self.mo_energy[self.nocc:]

    @property
    def e_corr(self):
        try:
            return self.e_corr_ss
        except AttributeError as e:
            self.log.critical("Can only access rpa.e_corr after running rpa.kernel.")

    @property
    def e_tot(self):
        return self.mf.e_tot + self.e_corr

    @property
    def D(self):
        eps = np.zeros((self.nocc, self.nvir))
        eps = eps + self.mo_energy_vir
        eps = (eps.T - self.mo_energy_occ).T
        eps = eps.reshape((self.ov,))
        D = np.concatenate([eps, eps])
        return D

    def kernel_moms(self, target_rot=None, npoints=48, ainit=10, integral_deduct="HO", opt_quad=True,
                    adaptive_quad=False):

        if target_rot is None:
            self.log.warning("Warning; generating full moment rather than local component. Will scale as O(N^5).")
            target_rot = np.eye(self.ov_tot)
        ri_mp, ri_apb, ri_amb = self.get_compressed_MP()

        # We our integral as
        #   integral = (MP)^{1/2} - (moment_offset) P - integral_offset
        # and so
        #   eta0 = (integral + integral_offset) P^{-1} + moment_offset
        offset_niworker = None
        inputs = (self.D, ri_mp[0], ri_mp[1], target_rot, npoints, self.log)
        if integral_deduct == "D":
            # Evaluate (MP)^{1/2} - D,
            niworker = momzero_NI.MomzeroDeductD(*inputs)
            integral_offset = einsum("lp,p->lp", target_rot, self.D)
            moment_offset = np.zeros_like(target_rot)
        elif integral_deduct is None:
            # Explicitly evaluate (MP)^{1/2}, with no offsets.
            niworker = momzero_NI.MomzeroDeductNone(*inputs)
            integral_offset = np.zeros_like(target_rot)
            moment_offset = np.zeros_like(target_rot)
        elif integral_deduct == "HO":
            niworker = momzero_NI.MomzeroDeductHigherOrder(*inputs)
            offset_niworker = momzero_NI.MomzeroOffsetCalcGaussLag(*inputs)
            estval, offset_err = offset_niworker.kernel()
            # This computes the required value analytically, but at N^5 cost. Just keeping around for debugging.
            # mat = np.zeros(self.D.shape * 2)
            # mat = mat + self.D
            # mat = (mat.T + self.D).T
            # estval2 = einsum("rp,pq,np,nq->rq", target_rot, mat ** (-1), ri_mp[0], ri_mp[1])
            # self.log.info("Error in numerical Offset Approximation=%6.4e",abs(estval - estval2).max())
            integral_offset = einsum("lp,p->lp", target_rot, self.D) + estval
            moment_offset = np.zeros_like(target_rot)
        else:
            raise ValueError("Unknown integral offset specification.`")
        # niworker.test_diag_derivs(4.0)
        if adaptive_quad:
            # Can also make use of scipy adaptive quadrature routines; this is more expensive but a good sense-check.
            integral, err = 2 * niworker.kernel_adaptive()
        else:
            integral, err = niworker.kernel(a=ainit, opt_quad=opt_quad)
        # Need to construct RI representation of P^{-1}
        ri_apb_inv = self.compress_low_rank(*construct_inverse_RI(self.D, ri_apb), name="(A+B)^-1")
        mom0 = einsum("pq,q->pq", integral + integral_offset, self.D ** (-1)) - np.dot(
            np.dot(integral + integral_offset, ri_apb_inv[0].T), ri_apb_inv[1])
        # Also need to convert error estimate of the integral into one for the actual evaluated quantity.
        # Use Cauchy-Schwartz to both obtain an upper bound on resulting mom0 error, and efficiently obtain upper bound
        # on norm of low-rank portion of P^{-1}.
        if err is not None:
            pinv_norm = (sum(self.D ** (-2)) + 2 * einsum("p,np,np->", self.D ** (-1), ri_apb_inv[0], ri_apb_inv[1]) +
                         np.linalg.norm(ri_apb_inv) ** 4) ** (0.5)
            mom0_err = err * pinv_norm
            self.check_errors(mom0_err, target_rot.size)
        else:
            mom0_err = None
        return mom0 + moment_offset, mom0_err

    def kernel_trMPrt(self, npoints=48, ainit=10):
        """Evaluate """
        ri_mp, ri_apb, ri_amb = self.get_compressed_MP()
        inputs = (self.D, ri_mp[0], ri_mp[1], npoints, self.log)

        niworker = energy_NI.NITrRootMP(*inputs)
        integral, err = niworker.kernel(a=ainit, opt_quad=True)
        # Compute offset; possible analytically in N^3 for diagonal.
        offset = sum(self.D) + 0.5 * einsum("p,np,np->", self.D**(-1), ri_mp[0], ri_mp[1])
        return integral[0] + offset, err

    def kernel_energy(self, npoints=48, ainit=10, use_correction=True):
        e1, err = self.kernel_trMPrt(npoints, ainit)
        e2 = 0.0
        ri_apb_eri = self.get_apb_eri_ri()
        # Note that eri contribution to A and B is equal, so can get trace over one by dividing by two
        e3 = sum(self.D) + einsum("np,np->", ri_apb_eri, ri_apb_eri) / 2
        print(e1, e2, e3)
        if self.rixc is not None:
            if use_correction:
                ri_a_xc, ri_b_xc = self.get_ab_xc_ri()
                eta0_xc, err2 = self.kernel_moms(target_rot=ri_b_xc[0], npoints=npoints, ainit=ainit)
                err += err2
                e2 -= np.dot(eta0_xc, ri_b_xc[1].T).trace()
                e3 += 2 * einsum("np,np->", ri_a_xc[0], ri_a_xc[1]) - einsum("np,np->", ri_b_xc[0], ri_b_xc[1])
        self.e_corr_ss = 0.5 * (e1 + e2 - e3)
        print(e1,e2,e3)
        err /= 2
        return self.e_corr_ss, err

    def get_compressed_MP(self):
        ri_apb, ri_amb = self.construct_RI_AB()
        # Compress RI representations before manipulation, since compression costs O(N^2 N_aux) while most operations
        # are O(N^2 N_aux^2), so for systems large enough that calculation cost is noticeable the cost reduction
        # from a reduced rank will exceed the cost of compression.
        ri_apb = self.compress_low_rank(*ri_apb, name="A+B")
        ri_amb = self.compress_low_rank(*ri_amb, name="A-B")
        ri_mp = self.compress_low_rank(*construct_product_RI(self.D, ri_amb, ri_apb), name="(A-B)(A+B)")
        return ri_mp, ri_apb, ri_amb

    def check_errors(self, error, nelements):
        if error / nelements > self.err_tol:
            self.log.warning("Estimated error per element exceeded tolerance %6.4e. Please increase number of points.",
                             error / nelements)

    def construct_RI_AB(self):
        """Construct the RI expressions for the deviation of A+B and A-B from D."""
        ri_apb_eri = self.get_apb_eri_ri()
        # Use empty AmB contrib initially; this is the dRPA contrib.
        ri_amb_eri = np.zeros((0, self.ov * 2))
        if self.rixc is not None:
            ri_a_xc, ri_b_xc = self.get_ab_xc_ri()

            ri_apb_xc = [np.concatenate([ri_a_xc[0], ri_b_xc[0]], axis=0), np.concatenate([ri_a_xc[1], ri_b_xc[1]],
                                                                                          axis=0)]
            ri_amb_xc = [np.concatenate([ri_a_xc[0], ri_b_xc[0]], axis=0), np.concatenate([ri_a_xc[1], -ri_b_xc[1]],
                                                                                          axis=0)]
        else:
            ri_apb_xc = [np.zeros((0, self.ov * 2))] * 2
            ri_amb_xc = [np.zeros((0, self.ov * 2))] * 2

        ri_apb = [np.concatenate([ri_apb_eri, x], axis=0) for x in ri_apb_xc]
        ri_amb = [np.concatenate([ri_amb_eri, x], axis=0) for x in ri_amb_xc]

        return ri_apb, ri_amb

    def compress_low_rank(self, ri_l, ri_r, name=None):
            return compress_low_rank(ri_l, ri_r, tol=self.svd_tol, log=self.log, name=name)

    def get_apb_eri_ri(self):
        # Coulomb integrals only contribute to A+B.
        # This needs to be optimised, but will do for now.
        v = self.get_3c_integrals()  # pyscf.lib.unpack_tril(self.mf._cderi)
        Lov = einsum("npq,pi,qa->nia", v, self.mo_coeff_occ, self.mo_coeff_vir).reshape((self.naux_eri, self.ov))
        ri_apb_eri = np.zeros((self.naux_eri, self.ov_tot))

        # Need to include factor of two since eris appear in both A and B.
        ri_apb_eri[:, :self.ov] = ri_apb_eri[:, self.ov:2 * self.ov] = np.sqrt(2) * Lov
        return ri_apb_eri

    def get_ab_xc_ri(self):
        # Have low-rank representation for interactions over and above coulomb interaction.
        # Note that this is usually asymmetric, as correction is non-PSD.
        ri_a_aa = [einsum("npq,pi,qa->nia", x, self.mo_coeff_occ, self.mo_coeff_vir).reshape((-1, self.ov)) for x in
                   self.rixc[0]]
        ri_a_bb = [einsum("npq,pi,qa->nia", x, self.mo_coeff_occ, self.mo_coeff_vir).reshape((-1, self.ov)) for x in
                   self.rixc[1]]

        ri_b_aa = [ri_a_aa[0],
                   einsum("npq,qi,pa->nia", self.rixc[0][1], self.mo_coeff_occ, self.mo_coeff_vir).reshape(
                       (-1, self.ov))]
        ri_b_bb = [ri_a_bb[0],
                   einsum("npq,qi,pa->nia", self.rixc[1][1], self.mo_coeff_occ, self.mo_coeff_vir).reshape(
                       (-1, self.ov))]

        ri_a_xc = [np.concatenate([x, y], axis=1) for x, y in zip(ri_a_aa, ri_a_bb)]
        ri_b_xc = [np.concatenate([x, y], axis=1) for x, y in zip(ri_b_aa, ri_b_bb)]
        return ri_a_xc, ri_b_xc

    def get_3c_integrals(self):
        return pyscf.lib.unpack_tril(next(self.mf.with_df.loop(blksize=self.naux_eri)))

    def test_spectral_rep(self, freqs):
        from vayesta.rpa import ssRPA
        import scipy
        import scipy.integrate

        xc_kernel = None
        if self.rixc is not None:
            xc_kernel = [
                einsum("npq,nrs->pqrs", *self.rixc[0]),
                einsum("npq,nrs->pqrs", self.rixc[0][0], self.rixc[0][1]),
                einsum("npq,nrs->pqrs", *self.rixc[1]),
            ]
        fullrpa = ssRPA(self.mf)
        fullrpa.kernel(xc_kernel=xc_kernel)
        print(fullrpa.freqs_ss[:20])
        target_rot = np.eye(self.ov_tot)
        ri_apb, ri_amb = self.construct_RI_AB()
        ri_mp = construct_product_RI(self.D, ri_amb, ri_apb)
        inputs = (self.D, ri_mp[0], ri_mp[1], target_rot, 48, self.log)
        niworker = momzero_NI.MomzeroDeductHigherOrder(*inputs)
        naux = ri_mp[0].shape[0]

        def get_qval(freq):
            q = niworker.get_Q(freq)
            return q.trace()

        def get_log_qval(freq):
            q = niworker.get_Q(freq)
            return scipy.linalg.logm(np.eye(naux) + q).trace()
        log_qvals = [get_log_qval(x) for x in freqs]
        def get_log_specvals(freq):
            return sum(np.log(fullrpa.freqs_ss**2 + freq**2) - np.log(self.D**2 + freq**2))
        log_specvals = [get_log_specvals(x) for x in freqs]

        return log_qvals, log_specvals, get_log_qval, get_log_specvals




def construct_product_RI(D, ri_1, ri_2):
    """Given two matrices expressed as low-rank modifications, cderi_1 and cderi_2, of some full-rank matrix D,
    construct the RI expression for the deviation of their product from D**2.
    The rank of the resulting deviation is at most the sum of the ranks of the original modifications."""
    # Construction of this matrix is the computationally limiting step of this construction (O(N^4)) in our usual use,
    # but we only need to perform it once per calculation since it's frequency-independent.
    if type(ri_1) == np.ndarray:
        ri_1_L = ri_1_R = ri_1
    else:
        (ri_1_L, ri_1_R) = ri_1

    if type(ri_2) == np.ndarray:
        ri_2_L = ri_2_R = ri_2
    else:
        (ri_2_L, ri_2_R) = ri_2

    U = np.dot(ri_1_R, ri_2_L.T)

    ri_L = np.concatenate([ri_1_L, einsum("p,np->np", D, ri_2_L) + np.dot(U.T, ri_1_L) / 2], axis=0)
    ri_R = np.concatenate([einsum("p,np->np", D, ri_1_R) + np.dot(U, ri_2_R) / 2, ri_2_R], axis=0)
    return ri_L, ri_R


def construct_inverse_RI(D, ri):
    if type(ri) == np.ndarray and len(ri.shape) == 2:
        ri_L = ri_R = ri
    else:
        (ri_L, ri_R) = ri

    naux = ri_R.shape[0]
    # This construction scales as O(N^4).
    U = einsum("np,p,mp->nm", ri_R, D ** (-1), ri_L)
    # This inversion and square root should only scale as O(N^3).
    U = np.linalg.inv(np.eye(naux) + U)
    # Want to split matrix between left and right fairly evenly; could just associate to one side or the other.
    u, s, v = np.linalg.svd(U)
    urt_l = einsum("nm,m->nm", u, s ** (0.5))
    urt_r = einsum("n,nm->nm", s ** (0.5), v)
    # Evaluate the resulting RI
    return einsum("p,np,nm->mp", D ** (-1), ri_L, urt_l), einsum("p,np,nm->mp", D ** (-1), ri_R, urt_r.T)


def compress_low_rank(ri_l, ri_r, tol=1e-8, log=None, name=None):
    naux_init = ri_l.shape[0]
    u, s, v = np.linalg.svd(ri_l, full_matrices=False)
    nwant = sum(s > tol)
    rot = u[:, :nwant]
    ri_l = dot(rot.T, ri_l)
    ri_r = dot(rot.T, ri_r)
    u, s, v = np.linalg.svd(ri_r, full_matrices=False)
    nwant = sum(s > tol)
    rot = u[:, :nwant]
    ri_l = dot(rot.T, ri_l)
    ri_r = dot(rot.T, ri_r)
    if nwant < naux_init and log is not None:
        if name is None:
            log.info("Compressed low-rank representation from rank %d to %d.", naux_init, nwant)
        else:
            log.info("Compressed low-rank representation of %s from rank %d to %d.", name, naux_init, nwant)
    return ri_l, ri_r

ssRIRRPA = ssRIRPA