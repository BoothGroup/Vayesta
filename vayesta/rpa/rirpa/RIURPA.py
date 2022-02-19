import numpy as np

import pyscf.lib
from vayesta.core.util import *

from .RIRPA import ssRIRRPA


class ssRIURPA(ssRIRRPA):

    @property
    def mo_occ(self):
        return self.mf.mo_occ

    @property
    def nmo(self):
        """Total number of molecular orbitals (MOs)."""
        return (self.mo_coeff[0].shape[-1],
                self.mo_coeff[1].shape[-1])

    @property
    def nocc(self):
        """Number of occupied MOs."""
        return (np.count_nonzero(self.mo_occ[0] > 0),
                np.count_nonzero(self.mo_occ[1] > 0))

    @property
    def nvir(self):
        """Number of virtual MOs."""
        return (np.count_nonzero(self.mo_occ[0] == 0),
                np.count_nonzero(self.mo_occ[1] == 0))

    @property
    def naux_eri(self):
        return self.mf.with_df.get_naoaux()

    @property
    def mo_coeff(self):
        """Occupied MO coefficients."""
        return self.mf.mo_coeff

    @property
    def mo_coeff_occ(self):
        """Occupied MO coefficients."""
        return (self.mo_coeff[0][:, :self.nocc[0]],
                self.mo_coeff[1][:, :self.nocc[1]])

    @property
    def mo_coeff_vir(self):
        """Virtual MO coefficients."""
        return (self.mo_coeff[0][:, self.nocc[0]:],
                self.mo_coeff[1][:, self.nocc[1]:])

    @property
    def mo_energy(self):
        return self.mf.mo_energy

    @property
    def mo_energy_occ(self):
        return self.mo_energy[0][:self.nocc[0]], self.mo_energy[1][:self.nocc[1]]

    @property
    def mo_energy_vir(self):
        return self.mo_energy[0][self.nocc[0]:], self.mo_energy[1][self.nocc[1]:]

    @property
    def ov(self):
        return self.nocc[0] * self.nvir[0], self.nocc[1] * self.nvir[1]

    @property
    def ov_tot(self):
        return sum(self.ov)

    @property
    def D(self):
        epsa = np.zeros((self.nocc[0], self.nvir[0]))
        epsb = np.zeros((self.nocc[1], self.nvir[1]))
        epsa = epsa + self.mo_energy_vir[0]
        epsa = (epsa.T - self.mo_energy_occ[0]).T
        epsa = epsa.reshape((self.ov[0],))

        epsb = epsb + self.mo_energy_vir[1]
        epsb = (epsb.T - self.mo_energy_occ[1]).T
        epsb = epsb.reshape((self.ov[1],))

        D = np.concatenate([epsa, epsb])
        return D

    def construct_RI_AB(self):
        """Construct the RI expressions for the deviation of A+B and A-B from D."""
        ri_apb_eri = self.get_apb_eri_ri()
        # Use empty AmB contrib initially; this is the dRPA contrib.
        ri_amb_eri = np.zeros((0, self.ov_tot))
        if self.rixc is not None:
            ri_a_xc, ri_b_xc = self.get_ab_xc_ri()

            ri_apb_xc = [np.concatenate([ri_a_xc[0], ri_b_xc[0]], axis=0), np.concatenate([ri_a_xc[1], ri_b_xc[1]],
                                                                                          axis=0)]
            ri_amb_xc = [np.concatenate([ri_a_xc[0], ri_b_xc[0]], axis=0), np.concatenate([ri_a_xc[1], -ri_b_xc[1]],
                                                                                          axis=0)]
        else:
            ri_apb_xc = [np.zeros((0, self.ov_tot))] * 2
            ri_amb_xc = [np.zeros((0, self.ov_tot))] * 2

        ri_apb = [np.concatenate([ri_apb_eri, x], axis=0) for x in ri_apb_xc]
        ri_amb = [np.concatenate([ri_amb_eri, x], axis=0) for x in ri_amb_xc]

        return ri_apb, ri_amb

    def get_apb_eri_ri(self):
        # Coulomb integrals only contribute to A+B.
        # This needs to be optimised, but will do for now.
        v = self.get_3c_integrals()
        Lov_a = einsum("npq,pi,qa->nia", v, self.mo_coeff_occ[0], self.mo_coeff_vir[0]).reshape(
            (self.naux_eri, self.ov[0]))
        Lov_b = einsum("npq,pi,qa->nia", v, self.mo_coeff_occ[1], self.mo_coeff_vir[1]).reshape(
            (self.naux_eri, self.ov[1]))

        ri_apb_eri = np.zeros((self.naux_eri, sum(self.ov)))

        # Need to include factor of two since eris appear in both A and B.
        ri_apb_eri[:, :self.ov[0]] = np.sqrt(2) * Lov_a
        ri_apb_eri[:, self.ov[0]:self.ov_tot] = np.sqrt(2) * Lov_b
        return ri_apb_eri

    def get_ab_xc_ri(self):
        # Have low-rank representation for interactions over and above coulomb interaction.
        # Note that this is usually asymmetric, as correction is non-PSD.
        ri_a_aa = [einsum("npq,pi,qa->nia", x, self.mo_coeff_occ[0], self.mo_coeff_vir[0]).reshape((-1, self.ov[0])) for
                   x in
                   self.rixc[0]]
        ri_a_bb = [einsum("npq,pi,qa->nia", x, self.mo_coeff_occ[1], self.mo_coeff_vir[1]).reshape((-1, self.ov[1])) for
                   x in
                   self.rixc[1]]

        ri_b_aa = [ri_a_aa[0],
                   einsum("npq,qi,pa->nia", self.rixc[0][1], self.mo_coeff_occ[0], self.mo_coeff_vir[0]).reshape(
                       (-1, self.ov[0]))]
        ri_b_bb = [ri_a_bb[0],
                   einsum("npq,qi,pa->nia", self.rixc[1][1], self.mo_coeff_occ[1], self.mo_coeff_vir[1]).reshape(
                       (-1, self.ov[1]))]

        ri_a_xc = [np.concatenate([x, y], axis=1) for x, y in zip(ri_a_aa, ri_a_bb)]
        ri_b_xc = [np.concatenate([x, y], axis=1) for x, y in zip(ri_b_aa, ri_b_bb)]
        return ri_a_xc, ri_b_xc