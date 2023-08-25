import numpy as np
from vayesta.core import spinalg
from vayesta.core.util import callif, dot, einsum
from vayesta.core.types import wf as wf_types
from vayesta.core.types.orbitals import SpatialOrbitals
from vayesta.core.types.wf.project import project_c2, project_uc2, symmetrize_c2, symmetrize_uc2
from vayesta.core.helper import pack_arrays, unpack_arrays


def MP2_WaveFunction(mo, t2, **kwargs):
    if mo.nspin == 1:
        cls = RMP2_WaveFunction
    elif mo.nspin == 2:
        cls = UMP2_WaveFunction
    return cls(mo, t2, **kwargs)


class RMP2_WaveFunction(wf_types.WaveFunction):
    def __init__(self, mo, t2, projector=None):
        super().__init__(mo, projector=projector)
        self.t2 = t2

    def make_rdm1(self, with_mf=True, ao_basis=False):
        t2 = self.t2
        doo = -(2 * einsum("ikab,jkab->ij", t2, t2) - einsum("ikab,kjab->ij", t2, t2))
        dvv = 2 * einsum("ijac,ijbc->ab", t2, t2) - einsum("ijac,ijcb->ab", t2, t2)
        if with_mf:
            doo += np.eye(self.nocc)
        dm1 = np.zeros((self.norb, self.norb))
        occ, vir = np.s_[: self.nocc], np.s_[self.nocc :]
        dm1[occ, occ] = doo + doo.T
        dm1[vir, vir] = dvv + dvv.T
        if ao_basis:
            dm1 = dot(self.mo.coeff, dm1, self.mo.coeff.T)
        return dm1

    def make_rdm2(self, with_dm1=True, ao_basis=False, approx_cumulant=True):
        dm2 = np.zeros((self.norb, self.norb, self.norb, self.norb))
        occ, vir = np.s_[: self.nocc], np.s_[self.nocc :]
        dovov = 4 * self.t2.transpose(0, 2, 1, 3) - 2 * self.t2.transpose(0, 3, 1, 2)
        dm2[occ, vir, occ, vir] = dovov
        dm2[vir, occ, vir, occ] = dovov.transpose(1, 0, 3, 2)
        if with_dm1:
            dm1 = self.make_rdm1(with_mf=False)
            dm1[np.diag_indices(self.nocc)] += 1
            for i in range(self.nocc):
                dm2[i, i, :, :] += 2 * dm1
                dm2[:, :, i, i] += 2 * dm1
                dm2[:, i, i, :] -= dm1
                dm2[i, :, :, i] -= dm1
        else:
            if int(approx_cumulant) != 1:
                raise NotImplementedError
        if ao_basis:
            dm2 = einsum("ijkl,ai,bj,ck,dl->abcd", dm2, *(4 * [self.mo.coeff]))
        return dm2

    def as_restricted(self):
        return self

    def as_unrestricted(self):
        mo = self.mo.to_spin_orbitals()
        t2 = self.t2.copy()
        t2aa = t2 - t2.transpose(0, 1, 3, 2)
        t2 = (t2aa, t2, t2aa)
        return UMP2_WaveFunction(mo, t2)

    def multiply(self, factor):
        self.t2 *= factor

    def project(self, projector, inplace=False):
        wf = self if inplace else self.copy()
        wf.t2 = project_c2(wf.t2, projector)
        wf.projector = projector
        return wf

    def restore(self, projector=None, inplace=False, sym=True):
        if projector is None:
            projector = self.projector
        wf = self.project(projector.T, inplace=inplace)
        wf.projector = None
        if not sym:
            return wf
        wf.t2 = symmetrize_c2(wf.t2)
        return wf

    def as_mp2(self):
        return self

    def as_cisd(self, c0=1.0):
        nocc1 = self.t2.shape[0]
        c1 = np.zeros((nocc1, self.nvir))
        c2 = c0 * self.t2
        return wf_types.RCISD_WaveFunction(self.mo, c0, c1, c2, projector=self.projector)

    def as_ccsd(self):
        nocc1 = self.t2.shape[0]
        t1 = np.zeros((nocc1, self.nvir))
        return wf_types.CCSD_WaveFunction(self.mo, t1, self.t2, l1=t1, l2=self.t2, projector=self.projector)

    def as_fci(self):
        raise NotImplementedError

    def copy(self):
        proj = callif(spinalg.copy, self.projector)
        t2 = spinalg.copy(self.t2)
        return type(self)(self.mo.copy(), t2, projector=proj)

    def pack(self, dtype=float):
        """Pack into a single array of data type `dtype`.

        Useful for communication via MPI."""
        mo = self.mo.pack(dtype=dtype)
        data = (mo, self.t2, self.projector)
        pack = pack_arrays(*data, dtype=dtype)
        return pack

    @classmethod
    def unpack(cls, packed):
        """Unpack from a single array of data type `dtype`.

        Useful for communication via MPI."""
        mo, t2, projector = unpack_arrays(packed)
        mo = SpatialOrbitals.unpack(mo)
        return cls(mo, t2, projector=projector)


class UMP2_WaveFunction(RMP2_WaveFunction):
    @property
    def t2aa(self):
        return self.t2[0]

    @property
    def t2ab(self):
        return self.t2[1]

    @property
    def t2ba(self):
        if len(self.t2) == 4:
            return self.t2[2]
        return self.t2ab.transpose(1, 0, 3, 2)

    @property
    def t2bb(self):
        return self.t2[-1]

    def make_rdm1(self, *args, **kwargs):
        raise NotImplementedError

    def make_rdm2(self, *args, **kwargs):
        raise NotImplementedError

    def project(self, projector, inplace=False):
        wf = self if inplace else self.copy()
        wf.t2 = project_uc2(wf.t2, projector)
        wf.projector = projector
        return wf

    def restore(self, projector=None, inplace=False, sym=True):
        if projector is None:
            projector = self.projector
        wf = self.project((projector[0].T, projector[1].T), inplace=inplace)
        wf.projector = None
        if not sym:
            return wf
        wf.t2 = symmetrize_uc2(wf.t2)
        return wf

    def as_mp2(self):
        return self

    def as_cisd(self, c0=1.0):
        nocc1a = self.t2aa.shape[0]
        nocc1b = self.t2bb.shape[0]
        c1 = (np.zeros((nocc1a, self.nvira)), np.zeros((nocc1b, self.nvirb)))
        c2aa = c0 * self.t2aa
        c2ab = c0 * self.t2ab
        c2bb = c0 * self.t2bb
        if len(self.t2) == 3:
            c2 = (c2aa, c2ab, c2bb)
        elif len(self.t2) == 4:
            c2ba = c0 * self.t2ba
            c2 = (c2aa, c2ab, c2ba, c2bb)
        return wf_types.UCISD_WaveFunction(self.mo, c0, c1, c2, projector=self.projector)

    def as_ccsd(self):
        nocc1a = self.t2aa.shape[0]
        nocc1b = self.t2bb.shape[0]
        t1 = (np.zeros((nocc1a, self.nvira)), np.zeros((nocc1b, self.nvirb)))
        return wf_types.UCCSD_WaveFunction(self.mo, t1, self.t2, l1=t1, l2=self.t2, projector=self.projector)

    def as_fci(self):
        return NotImplementedError

    def multiply(self, factor):
        self.t2 = spinalg.multiply(self.t2, len(self.t2) * [factor])
