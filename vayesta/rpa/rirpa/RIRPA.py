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

    def __init__(self, dfmf, rixc=None, log=None, err_tol=1e-6, svd_tol=1e-12):
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

    def kernel_moms(self, max_moment, target_rot=None, npoints=48, ainit=10, integral_deduct="HO", opt_quad=True,
                    adaptive_quad=False, alpha=1.0):
        if target_rot is None:
            self.log.warning("Warning; generating full moment rather than local component. Will scale as O(N^5).")
            target_rot = np.eye(self.ov_tot)
        # First need to calculate zeroth moment.
        moments = np.zeros((max_moment+1,) + target_rot.shape)
        moments[0], err0 = self._kernel_mom0(target_rot, npoints, ainit, integral_deduct, opt_quad, adaptive_quad, alpha)

        if max_moment > 0:
            # Grab mean.
            D = self.D
            ri_mp, ri_apb, ri_amb = self.get_compressed_MP()
            moments[1] = einsum("pq,q->pq", target_rot, D) + dot(target_rot, ri_amb[0].T, ri_amb[1])

        if max_moment > 1:
            Dsq = D**2
            for i in range(2, max_moment+1):
                moments[i] = einsum("pq,q->pq", moments[i-2], Dsq) + dot(moments[i-2], ri_mp[1].T, ri_mp[0])
        return moments, err0

    def _kernel_mom0(self, target_rot=None, npoints=48, ainit=10, integral_deduct="HO", opt_quad=True,
                    adaptive_quad=False, alpha=1.0):

        if target_rot is None:
            self.log.warning("Warning; generating full moment rather than local component. Will scale as O(N^5).")
            target_rot = np.eye(self.ov_tot)
        ri_mp, ri_apb, ri_amb = self.get_compressed_MP(alpha)

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
        return mom0 + moment_offset, (mom0_err, self.test_eta0_error(mom0 + moment_offset, target_rot, ri_apb, ri_amb))

    def test_eta0_error(self, mom0, target_rot, ri_apb, ri_amb):
        """Test how well our obtained zeroth moment obeys relation used to derive it, namely
                A-B = eta0 (A+B) eta0
        From this we can estimate the error in eta0 using Cauchy-Schwartz.
        """
        l1 = [dot(mom0, x.T) for x in ri_apb]
        l2 = [dot(target_rot, x.T) for x in ri_amb]
        #amb_exact = einsum("pq,q,rq->pr", target_rot, self.D, target_rot) + dot(l2[0], l2[1].T)
        #print(amb_exact)
        #amb_approx = einsum("pq,q,rq->pr", mom0, self.D, mom0) + dot(l1[0], l1[1].T)
        #error = amb_approx - amb_exact
        amb = np.diag(self.D) + dot(ri_amb[0].T, ri_amb[1])
        apb = np.diag(self.D) + dot(ri_apb[0].T, ri_apb[1])
        amb_exact = dot(target_rot, amb, target_rot.T)
        self.save = (apb, amb_exact, mom0)
        error = amb_exact - dot(mom0, apb, mom0.T)
        self.error = error
        e_norm = np.linalg.norm(error)
        p_norm = np.linalg.norm(self.D) + np.linalg.norm(ri_apb) ** 2
        peta_norm = np.linalg.norm(einsum("p,qp->pq", self.D, mom0) + dot(ri_apb[0].T, l1[1].T))

        # Now to estimate resulting error estimate in eta0.
        poly = np.polynomial.Polynomial([e_norm/p_norm, -2 * peta_norm / p_norm, 1])
        roots = poly.roots()
        self.log.info("Proportional error in eta0 relation=%6.4e", e_norm / np.linalg.norm(amb_exact))
        self.log.info("Resulting error lower bound: %6.4e", roots.min())

        return roots.min()

    def kernel_trMPrt(self, npoints=48, ainit=10):
        """Evaluate Tr((MP)^(1/2))."""
        ri_mp, ri_apb, ri_amb = self.get_compressed_MP()
        inputs = (self.D, ri_mp[0], ri_mp[1], npoints, self.log)

        niworker = energy_NI.NITrRootMP(*inputs)
        integral, err = niworker.kernel(a=ainit, opt_quad=True)
        # Compute offset; possible analytically in N^3 for diagonal.
        offset = sum(self.D) + 0.5 * einsum("p,np,np->", self.D ** (-1), ri_mp[0], ri_mp[1])
        return integral[0] + offset, err

    def kernel_energy(self, npoints=48, ainit=10, correction="linear"):
        e1, err = self.kernel_trMPrt(npoints, ainit)
        e2 = 0.0
        ri_apb_eri = self.get_apb_eri_ri()
        # Note that eri contribution to A and B is equal, so can get trace over one by dividing by two
        e3 = sum(self.D) + einsum("np,np->", ri_apb_eri, ri_apb_eri) / 2
        if self.rixc is not None and correction is not None:

            if correction.lower() == "linear":
                ri_a_xc, ri_b_xc = self.get_ab_xc_ri()
                eta0_xc, errs = self.kernel_moms(0, target_rot=ri_b_xc[0], npoints=npoints, ainit=ainit)
                eta0_xc = eta0_xc[0]
                err += errs[1]
                val = np.dot(eta0_xc, ri_b_xc[1].T).trace() / 2
                self.log.info("Approximated correlation energy contribution: %e", val)
                e2 -= val
                e3 += einsum("np,np->", ri_a_xc[0], ri_a_xc[1]) - einsum("np,np->", ri_b_xc[0], ri_b_xc[1]) / 2
            elif correction.lower() == "xc_ac":
                pass
        self.e_corr_ss = 0.5 * (e1 + e2 - e3)
        err /= 2
        return self.e_corr_ss, err

    def direct_AC_integration(self, local_rot=None, fragment_projectors=None, deg=5, npoints=48,
                              cluster_constrain=False):
        """Perform direct integration of the adiabatic connection for RPA correlation energy.
        This will be preferable when the xc kernel is comparable or larger in magnitude to the coulomb kernel, as it
        only requires evaluation of the moment and not its inverse.
        local_rot describes the rotation of the ov-excitations to the space of local excitations within a cluster, while
        fragment_projectors gives the projector within this local excitation space to the actual fragment."""
        # Get the coulomb integrals.
        ri_eri = self.get_apb_eri_ri() / np.sqrt(2)

        def get_eta_alpha(alpha, target_rot):
            newrirpa = self.__class__(self.mf, rixc=self.rixc, log=self.log)
            moms, errs = newrirpa.kernel_moms(0, target_rot=target_rot, npoints=npoints, alpha=alpha)
            return moms[0]

        def run_ac_inter(func, deg=5):
            points, weights = np.polynomial.legendre.leggauss(deg)
            # Shift and reweight to interval of [0,1].
            points += 1
            points /= 2
            weights /= 2
            return sum([w * func(p) for w, p in zip(weights, points)])

        naux_eri = ri_eri.shape[0]
        if local_rot is None or fragment_projectors is None:
            lrot = ri_eri
            rrot = ri_eri
        else:

            if cluster_constrain:
                lrot = np.concatenate(local_rot, axis=0)
                nloc_cum = np.cumsum([x.shape[0] for x in local_rot])

                rrot = np.zeros_like(lrot)

                def get_contrib(rot, proj):
                    lloc = dot(rot, ri_eri.T)
                    return dot(lloc, lloc[:proj.shape[0]].T, proj, rot[:proj.shape[0]])
                # return dot(rot[:proj.shape[0]].T, proj, lloc[:proj.shape[0]], lloc.T)
            else:

                lrot = np.concatenate(
                    [dot(x[:p.shape[0]], ri_eri.T, ri_eri) for x, p in zip(local_rot, fragment_projectors)], axis=0)
                nloc_cum = np.cumsum([x.shape[0] for x in fragment_projectors])

                rrot = np.zeros_like(lrot)

                def get_contrib(rot, proj):
                    return dot(proj, rot[:proj.shape[0]])

            rrot[:nloc_cum[0]] = get_contrib(local_rot[0], fragment_projectors[0])
            # rrot[nloc_cum[-1]:] = get_contrib(local_rot[-1], fragment_projectors[-1])
            if len(nloc_cum) > 1:
                for i, (r, p) in enumerate(zip(local_rot[1:], fragment_projectors[1:])):
                    rrot[nloc_cum[i]:nloc_cum[i + 1]] = get_contrib(r, p)
            lrot = np.concatenate([ri_eri, lrot], axis=0)
            rrot = np.concatenate([ri_eri, rrot], axis=0)

        def get_contrib(alpha):
            eta0 = get_eta_alpha(alpha, target_rot=lrot)
            return np.array([einsum("np,np->", (eta0 - lrot)[:naux_eri], rrot[:naux_eri]),
                             einsum("np,np->", (eta0 - lrot)[naux_eri:], rrot[naux_eri:])])

        integral = run_ac_inter(get_contrib, deg=deg) / 2
        return integral, get_contrib

    def get_gap(self, calc_xy=False, tol_eig=1e-2, max_space=12, nroots=1, **kwargs):
        """Calculate the RPA gap using a Davidson solver. First checks that A+B and A-B are PSD by calculating their
        lowest eigenvalues. For a fixed number of eigenvalues in each case this scales as O(N^3), so shouldn't be
        prohibitively expensive.
        """
        ri_mp, ri_apb, ri_amb = self.get_compressed_MP()

        min_d = self.D.min()

        mininds = np.where((self.D - min_d) < tol_eig)[0]
        nmin = len(mininds)
        if max_space < nmin:
            self.log.info("Expanded Davidson space size to %d to span degenerate lowest mean-field eigenvalues.", nmin)
            max_space = nmin

        def get_unit_vec(pos):
            x = np.zeros_like(self.D)
            x[pos] = 1.0
            return x
        c0 = [get_unit_vec(pos) for pos in mininds]

        def get_lowest_eigenvals(diag, ri_l, ri_r, x0, nroots=1, nosym=False):
            def hop(x):
                return einsum("p,p->p", diag, x) + einsum("np,nq,q->p", ri_l, ri_r, x)

            mdiag = diag + einsum("np,np->p", ri_l, ri_r)

            def precond(x, e, *args):
                return x / (mdiag - e + 1e-4)
            if nosym:
                # Ensure left isn't in our kwargs.
                kwargs.pop("left", None)
                e, c_l, c_r = pyscf.lib.eig(hop, x0, precond, max_space=max_space, nroots=nroots, left=True, **kwargs)
                return e, np.array(c_l), np.array(c_r)
            else:
                e, c = pyscf.lib.davidson(hop, x0, precond, max_space=max_space, nroots=nroots, **kwargs)
                return e, np.array(c)

        # Since A+B and A-B are symmetric can get eigenvalues straightforwardly.
        e_apb, c = get_lowest_eigenvals(self.D, *ri_apb, c0)
        if e_apb < 0.0:
            self.log.critical("Lowest eigenvalue of A+B is negative!")
            raise RuntimeError("RPA approximation broken down!")
        e_amb, c = get_lowest_eigenvals(self.D, *ri_amb, c0)
        if e_amb < 0.0:
            self.log.critical("Lowest eigenvalue of A-B is negative!")
            raise RuntimeError("RPA approximation broken down!")
        # MP is asymmetric, so need to take care to obtain actual eigenvalues.
        # Use Davidson to obtain accurate right eigenvectors...
        e_mp_r, c_l_approx, c_r = get_lowest_eigenvals(self.D**2, *ri_mp, c0, nroots=nroots, nosym=True)

        if not calc_xy:
            return e_mp_r ** (0.5)

        # Then solve for accurate left eigenvectors, starting from subspace approximation from right eigenvectors. Take
        # the real component since all solutions should be real.
        e_mp_l, c_r_approx, c_l = get_lowest_eigenvals(self.D**2, ri_mp[1], ri_mp[0], c_l_approx.real,
                                                       nroots=nroots, nosym=True)
        # We use c_r and c_l2, since these are likely the most accurate.
        # Enforce correct RPA orthonormality.
        ovlp = np.dot(c_l, c_r.T)

        if nroots > 1:
            c_l = np.dot(np.linalg.inv(ovlp), c_l)
            # Now diagonalise in corresponding subspace to get eigenvalues.
            subspace = einsum("np,p,mp->nm", c_l, self.D**2, c_r) + einsum("np,yp,yq,mq->nm", c_l, *ri_mp, c_r)
            e, c_sub = np.linalg.eig(subspace)
            # Now fold these eigenvectors into our definitions,
            xpy = np.dot(c_sub.T, c_r)
            xmy = np.dot(np.linalg.inv(c_sub), c_l)

            sorted_args = e.argsort()
            xpy = xpy[sorted_args]
            xmy = xmy[sorted_args]
            e = e[sorted_args]
        else:
            xpy = c_r / (ovlp**(0.5))
            xmy = c_l / (ovlp ** (0.5))
            e = einsum("p,p,p->", xmy, self.D**2, xpy) + einsum("p,yp,yq,q->", xmy, *ri_mp, xpy)

        return e**(0.5), xpy, xmy

    def get_compressed_MP(self, alpha=1.0):
        # AB corresponds to scaling RI components at this point.
        ri_apb, ri_amb = self.construct_RI_AB()
        # Compress RI representations before manipulation, since compression costs O(N^2 N_aux) while most operations
        # are O(N^2 N_aux^2), so for systems large enough that calculation cost is noticeable the cost reduction
        # from a reduced rank will exceed the cost of compression.
        ri_apb = self.compress_low_rank(*ri_apb, name="A+B")
        ri_amb = self.compress_low_rank(*ri_amb, name="A-B")
        ri_apb = [x * alpha ** (0.5) for x in ri_apb]
        ri_amb = [x * alpha ** (0.5) for x in ri_amb]

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
            return sum(np.log(fullrpa.freqs_ss ** 2 + freq ** 2) - np.log(self.D ** 2 + freq ** 2))

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


def compress_low_rank(ri_l, ri_r, tol=1e-12, log=None, name=None):
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
