import numpy as np
import scipy

from vayesta.core.util import *
from vayesta.rpa.rirpa import momzero_NI

import pyscf.lib

import logging


class ssRIRPA:
    """Approach based on equations expressed succinctly in the appendix of
    Furche, F. (2001). PRB, 64(19), 195120. https://doi.org/10.1103/PhysRevB.64.195120
    WARNING: Should only be used with canonical mean-field orbital coefficients in mf.mo_coeff and RHF.
    """

    def __init__(self, dfmf, rixc=None, log=None, err_tol=1e-6):
        self.mf = dfmf
        self.rixc = rixc
        self.log = log or logging.getLogger(__name__)
        self.err_tol = err_tol

    @property
    def nocc(self):
        return sum(self.mf.mo_occ > 0)

    @property
    def nvir(self):
        return len(self.mf.mo_occ) - self.nocc

    @property
    def naux(self):
        return self.mf.with_df.get_naoaux()

    @property
    def ov(self):
        return self.nocc * self.nvir

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

    def kernel(self, maxmom=0):
        pass

    def kernel_moms(self, target_rot=None, npoints=100, ainit=10, integral_deduct="HO", opt_quad=True,
                    adaptive_quad=False):
        if target_rot is None:
            self.log.warning("Warning; generating full moment rather than local component. Will scale as O(N^5).")
            target_rot = np.eye(2 * self.ov)
        ri_apb, ri_amb = self.construct_RI_AB()
        ri_mp = construct_product_RI(self.D, ri_amb, ri_apb)
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
            # Can also make use of scipy adaptive quadrature routines; this is likely more expensive but more reliable.
            integral, err = 2 * niworker.kernel_adaptive()
        else:
            integral, err = niworker.kernel(a=ainit, opt_quad=opt_quad)
        # Need to construct RI representation of P^{-1}
        ri_apb_inv = construct_inverse_RI(self.D, ri_apb)

        mom0 = einsum("pq,q->pq", integral + integral_offset, self.D ** (-1)) - np.dot(
            np.dot(integral + integral_offset, ri_apb_inv.T), ri_apb_inv)

        if not ((err is None) or adaptive_quad):
            self.check_errors(err, target_rot.size)

        return mom0 + moment_offset, err  # integral, (niworker, offset_niworker),

    def kernel_energy(self):

        pass

    def check_errors(self, error, nelements):
        if error / nelements > self.err_tol:
            self.log.warning("Estimated error per element exceeded tolerance %6.4e. Please increase number of points.",
                             error / nelements)

    def construct_RI_AB(self):
        """Construct the RI expressions for the deviation of A+B and A-B from D."""
        # Coulomb integrals only contribute to A+B.
        # This needs to be optimised, but will do for now.
        v = self.get_3c_integrals()  # pyscf.lib.unpack_tril(self.mf._cderi)
        Lov = einsum("npq,pi,qa->nia", v, self.mo_coeff_occ, self.mo_coeff_vir).reshape((self.naux, self.ov))
        ri_ApB = np.zeros((self.naux, self.ov * 2))
        # Need to include factor of two since eris appear in both A and B.
        ri_ApB[:, :self.ov] = ri_ApB[:, self.ov:2 * self.ov] = np.sqrt(2) * Lov
        # Use empty AmB contrib initially; this is the dRPA contrib.
        ri_AmB = np.zeros((0, self.ov * 2))
        return ri_ApB, ri_AmB

    def get_3c_integrals(self):
        return pyscf.lib.unpack_tril(next(self.mf.with_df.loop(blksize=self.naux)))


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
    if type(ri) == np.ndarray:
        ri_L = ri_R = ri
    else:
        (ri_L, ri_R) = ri

    naux = ri_R.shape[0]
    # This construction scales as O(N^4).
    U = einsum("np,p,mp->nm", ri_R, D ** (-1), ri_L)
    # This inversion and square root should only scale as O(N^3).
    U = np.linalg.inv(np.eye(naux) + U)
    Urt = scipy.linalg.sqrtm(U)
    # Evaluate the resulting RI
    if type(ri) == np.ndarray:
        return einsum("p,np,nm->mp", D ** (-1), ri_L, Urt)
    else:
        return einsum("p,np,nm->mp", D ** (-1), ri_L, Urt), einsum("p,np,nm->mp", D ** (-1), ri_R, Urt.T)
