import numpy as np
from vayesta.core.util import einsum
from vayesta.core.types import wf as wf_types


def HF_WaveFunction(mo):
    if mo.nspin == 1:
        cls = RHF_WaveFunction
    elif mo.nspin == 2:
        cls = UHF_WaveFunction
    return cls(mo)


class RHF_WaveFunction(wf_types.WaveFunction):
    def make_rdm1(self, mo_coeff=None, mo_occ=None, ao_basis=True):
        if mo_occ is None:
            mo_occ = self.mo.occ
        if not ao_basis:
            return np.diag(mo_occ)
        if mo_coeff is None:
            mo_coeff = self.mo.coeff
        occ = mo_occ > 0
        return np.dot(mo_coeff[:, occ] * mo_occ[occ], mo_coeff[:, occ].T)

    def make_rdm2(self, mo_coeff=None, mo_occ=None, ao_basis=True):
        dm1 = self.make_rdm1(mo_coeff=mo_coeff, mo_occ=mo_occ, ao_basis=ao_basis)
        dm2 = einsum("ij,kl->ijkl", dm1, dm1) - einsum("ij,kl->iklj", dm1, dm1) / 2
        return dm2

    def as_restricted(self):
        return self

    def as_unrestricted(self):
        raise NotImplementedError


class UHF_WaveFunction(RHF_WaveFunction):
    def make_rdm1(self, mo_coeff=None, mo_occ=None, ao_basis=True):
        if mo_coeff is None:
            mo_coeff = self.mo.coeff
        if mo_occ is None:
            mo_occ = self.mo.occ
        dm1a = super().make_rdm1(mo_coeff=mo_coeff[0], mo_occ=mo_occ[0], ao_basis=ao_basis)
        dm1b = super().make_rdm1(mo_coeff=mo_coeff[1], mo_occ=mo_occ[1], ao_basis=ao_basis)
        return (dm1a, dm1b)

    def make_rdm2(self, mo_coeff=None, mo_occ=None, ao_basis=True):
        dm1a, dm1b = self.make_rdm1(mo_coeff=mo_coeff, mo_occ=mo_occ, ao_basis=ao_basis)
        dm2aa = einsum("ij,kl->ijkl", dm1a, dm1a) - einsum("ij,kl->iklj", dm1a, dm1a)
        dm2bb = einsum("ij,kl->ijkl", dm1b, dm1b) - einsum("ij,kl->iklj", dm1b, dm1b)
        dm2ab = einsum("ij,kl->ijkl", dm1a, dm1b)
        return (dm2aa, dm2ab, dm2bb)

    def as_restricted(self):
        raise NotImplementedError

    def as_unrestricted(self):
        return self
