import numpy as np
from vayesta.core.util import *
from vayesta.rpa.rirpa import momzero_calculation


from pyscf.ao2mo import _ao2mo
import pyscf.lib


import logging

class ssRPA:
    """Approach based on equations expressed succinctly in the appendix of
    Furche, F. (2001). PRB, 64(19), 195120. https://doi.org/10.1103/PhysRevB.64.195120
    WARNING: Should only be used with canonical mean-field orbital coefficients in mf.mo_coeff and RHF.
    """

    def __init__(self, dfmf, rixc = None, log=None):
        self.mf = dfmf
        self.rixc = rixc
        self.log = log or logging.getLogger(__name__)

    @property
    def nocc(self):
        return sum(self.mf.mo_occ > 0)
    @property
    def nvir(self):
        return len(self.mf.mo_occ) - self.nocc
    @property
    def naux(self):
        return self.mf._cderi.shape[0]
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
        return self.mo_coeff[:,:self.nocc]
    @property
    def mo_coeff_vir(self):
        """Virtual MO coefficients."""
        return self.mo_coeff[:,self.nocc:]
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

    def kernel(self, maxmom = 0):
        pass

    def kernel_moms(self, maxmom = 0, npoints = 100, ainit=200):

        eps = np.zeros((self.nocc, self.nvir))
        eps = eps + self.mo_energy_vir
        eps = (eps.T - self.mo_energy_occ).T
        eps = eps.reshape((self.ov,))
        D = np.concatenate([eps, eps])

        ri_ApB, ri_AmB = self.construct_RI_AB()

        return momzero_calculation.eval_eta0(D, ri_ApB, ri_AmB, np.eye(2*self.ov), npoints, ainit)

    def kernel_energy(self):

        pass

    def construct_RI_AB(self):
        """Construct the RI expressions for the deviation of A+B and A-B from D."""
        # Coulomb integrals only contribute to A+B.
        # This needs to be optimised, but will do for now.
        v = pyscf.lib.unpack_tril(self.mf._cderi)
        Lov = einsum("npq,pi,qa->nia", v, self.mo_coeff_occ, self.mo_coeff_vir).reshape((self.naux, self.ov))
        ri_ApB = np.zeros((self.naux, self.ov*2))
        # Need to include factor of two since eris appear in both A and B.
        ri_ApB[:,:self.ov] = ri_ApB[:,self.ov:2*self.ov] = np.sqrt(2) * Lov
        # Use empty AmB contrib initially; this is the dRPA contrib.
        ri_AmB = np.zeros((0, self.ov*2))
        return ri_ApB, ri_AmB


