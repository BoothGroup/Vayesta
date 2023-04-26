import numpy as np
from vayesta.core.util import einsum

from vayesta.rpa.rirpa.RIRPA import ssRIRRPA
import pyscf.lib
from vayesta.core.util import einsum
from vayesta.core.eris import get_cderi

from .RIRPA import ssRIRRPA


class ssRIURPA(ssRIRRPA):
    @property
    def mo_occ(self):
        return self.mf.mo_occ

    @property
    def nmo(self):
        """Total number of molecular orbitals (MOs)."""
        return (self.mo_coeff[0].shape[-1], self.mo_coeff[1].shape[-1])

    @property
    def nocc(self):
        """Number of occupied MOs."""
        return (
            np.count_nonzero(self.mo_occ[0] > 0),
            np.count_nonzero(self.mo_occ[1] > 0),
        )

    @property
    def nvir(self):
        """Number of virtual MOs."""
        return (
            np.count_nonzero(self.mo_occ[0] == 0),
            np.count_nonzero(self.mo_occ[1] == 0),
        )

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
        return (
            self.mo_coeff[0][:, : self.nocc[0]],
            self.mo_coeff[1][:, : self.nocc[1]],
        )

    @property
    def mo_coeff_vir(self):
        """Virtual MO coefficients."""
        return (
            self.mo_coeff[0][:, self.nocc[0] :],
            self.mo_coeff[1][:, self.nocc[1] :],
        )

    @property
    def mo_energy(self):
        return self.mf.mo_energy

    @property
    def mo_energy_occ(self):
        return self.mo_energy[0][: self.nocc[0]], self.mo_energy[1][: self.nocc[1]]

    @property
    def mo_energy_vir(self):
        return self.mo_energy[0][self.nocc[0] :], self.mo_energy[1][self.nocc[1] :]

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

    def get_apb_eri_ri(self):
        # Coulomb integrals only contribute to A+B.
        # This needs to be optimised, but will do for now.
        if self.Lpq is None:
            v = self.get_3c_integrals()
            Lov_a = einsum(
                "npq,pi,qa->nia", v, self.mo_coeff_occ[0], self.mo_coeff_vir[0]
            ).reshape((self.naux_eri, self.ov[0]))
            Lov_b = einsum(
                "npq,pi,qa->nia", v, self.mo_coeff_occ[1], self.mo_coeff_vir[1]
            ).reshape((self.naux_eri, self.ov[1]))
        else:
            Lov_a = self.Lpq[0][:, : self.nocc[0], self.nocc[0] :].reshape(
                (self.naux_eri, self.ov[0])
            )
            Lov_b = self.Lpq[1][:, : self.nocc[1], self.nocc[1] :].reshape(
                (self.naux_eri, self.ov[1])
            )

        ri_apb_eri = np.zeros((self.naux_eri, sum(self.ov)))

        # Need to include factor of two since eris appear in both A and B.
        ri_apb_eri[:, : self.ov[0]] = np.sqrt(2) * Lov_a
        ri_apb_eri[:, self.ov[0] : self.ov_tot] = np.sqrt(2) * Lov_b
        return ri_apb_eri

    def get_apb_eri_ri(self):
        # Coulomb integrals only contribute to A+B.
        # This needs to be optimised, but will do for now.
        if self.lov is None:
            (lova, lovb), (lova_neg, lovb_neg) = self.get_cderi()  # pyscf.lib.unpack_tril(self.mf._cderi)
        else:
            if isinstance(self.lov, tuple):
                (lova, lovb), (lova_neg, lovb_neg) = self.lov
            else:
                assert self.lov[0][0].shape == (self.naux_eri, self.nocc[0], self.nvir[0])
                assert self.lov[0][1].shape == (self.naux_eri, self.nocc[1], self.nvir[1])
                lova, lovb = self.lov
                lova_neg = lovb_neg = None

        lova = lova.reshape((lova.shape[0], -1))
        lovb = lovb.reshape((lovb.shape[0], -1))
        if lova_neg is not None:
            if lovb_neg is None:
                raise RuntimeError("Encountered negative cderi contribution in only one spin channel."
                                   "Isn't this impossible?")
            lova_neg = lova_neg.reshape((lova_neg.shape[0], -1))
            lovb_neg = lovb_neg.reshape((lovb_neg.shape[0], -1))

        # Need to include factor of two since eris appear in both A and B.
        ri_apb_eri = np.sqrt(2) * np.concatenate([lova, lovb], axis = 1)

        ri_neg_apb_eri = None
        if lova_neg is not None:
            ri_neg_apb_eri = np.sqrt(2) * np.concatenate([lova_neg, lovb_neg], axis=1)

        return ri_apb_eri, ri_neg_apb_eri

    def get_cderi(self, blksize=None):

        la, la_neg = get_cderi(self, (self.mo_coeff_occ[0], self.mo_coeff_vir[0]), compact=False, blksize=blksize)
        lb, lb_neg = get_cderi(self, (self.mo_coeff_occ[1], self.mo_coeff_vir[1]), compact=False, blksize=blksize)
        return (la, lb), (la_neg, lb_neg)
