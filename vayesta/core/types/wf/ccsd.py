import numpy as np
# TODO: Remove once unnecessary:
from vayesta.core.vpyscf import ccsd_rdm
from vayesta.core.vpyscf import uccsd_rdm
# import pyscf
# import pyscf.cc
from vayesta.core import spinalg
from vayesta.core.util import *
from vayesta.core.types import wf
from vayesta.core.types.orbitals import *
from vayesta.core.types.wf.project import *


def CCSD_WaveFunction(mo, t1, t2, **kwargs):
    if mo.nspin == 1:
        cls = RCCSD_WaveFunction
    elif mo.nspin == 2:
        cls = UCCSD_WaveFunction
    return cls(mo, t1, t2, **kwargs)


class RCCSD_WaveFunction(wf.WaveFunction):

    # TODO: Once standard PySCF accepts additional keyword arguments:
    #_make_rdm1_backend = pyscf.cc.ccsd_rdm.make_rdm1
    #_make_rdm2_backend = pyscf.cc.ccsd_rdm.make_rdm2
    _make_rdm1_backend = ccsd_rdm.make_rdm1
    _make_rdm2_backend = ccsd_rdm.make_rdm2

    def __init__(self, mo, t1, t2, l1=None, l2=None, projector=None):
        super().__init__(mo, projector=projector)
        self.t1 = t1
        self.t2 = t2
        self.l1 = l1
        self.l2 = l2

    def make_rdm1(self, t_as_lambda=False, with_mf=True, ao_basis=False):
        if t_as_lambda:
            l1, l2 = self.t1, self.t2
        elif (self.l1 is None or self.l2 is None):
            raise NotCalculatedError("Lambda-amplitudes required for RDM1.")
        else:
            l1, l2 = self.l1, self.l2
        fakecc = Object()
        fakecc.mo_coeff = self.mo.coeff
        dm1 = type(self)._make_rdm1_backend(fakecc, t1=self.t1, t2=self.t2, l1=l1, l2=l2,
                with_frozen=False, ao_repr=ao_basis, with_mf=with_mf)
        return dm1

    def make_rdm2(self, t_as_lambda=False, with_dm1=True, ao_basis=False, approx_cumulant=True):
        if t_as_lambda:
            l1, l2 = self.t1, self.t2
        elif (self.l1 is None or self.l2 is None):
            raise NotCalculatedError("Lambda-amplitudes required for RDM2.")
        else:
            l1, l2 = self.l1, self.l2
        fakecc = Object()
        fakecc.mo_coeff = self.mo.coeff
        fakecc.stdout = None
        fakecc.verbose = 0
        fakecc.max_memory = int(10e9)   # 10 GB
        dm2 = type(self)._make_rdm2_backend(fakecc, t1=self.t1, t2=self.t2, l1=l1, l2=l2,
                with_frozen=False, ao_repr=ao_basis, with_dm1=with_dm1)
        if not with_dm1:
            if not approx_cumulant:
                dm2nc = self.make_rdm2_non_cumulant(t_as_lambda=t_as_lambda, ao_basis=ao_basis)
                if isinstance(dm2nc, np.ndarray):
                    dm2 -= dm2nc
                # UHF:
                else:
                    dm2 = tuple((dm2[i]-dm2nc[i]) for i in range(len(dm2nc)))
            elif (approx_cumulant in (1, True)):
                pass
            elif (approx_cumulant == 2):
                raise NotImplementedError
            else:
                raise ValueError
        return dm2

    def make_rdm2_non_cumulant(self, t_as_lambda=False, ao_basis=False):
        dm1 = self.make_rdm1(t_as_lambda=t_as_lambda, with_mf=False, ao_basis=ao_basis)
        dm2 = (einsum('ij,kl->ijkl', dm1, dm1) - einsum('ij,kl->iklj', dm1, dm1)/2)
        return dm2

    def multiply(self, factor):
        self.t1 *= factor
        self.t2 *= factor
        if self.l1 is not None:
            self.l1 *= factor
        if self.l2 is not None:
            self.l2 *= factor

    def project(self, projector, inplace=False):
        wf = self if inplace else self.copy()
        wf.t1 = project_c1(wf.t1, projector)
        wf.t2 = project_c2(wf.t2, projector)
        wf.l1 = project_c1(wf.l1, projector)
        wf.l2 = project_c2(wf.l2, projector)
        wf.projector = projector
        return wf

    def pack(self, dtype=float):
        """Pack into a single array of data type `dtype`.

        Useful for communication via MPI."""
        mo = self.mo.pack(dtype=dtype)
        data = (mo, self.t1, self.t2, self.l1, self.l2, self.projector)
        pack = pack_arrays(*data, dtype=dtype)
        return pack

    @classmethod
    def unpack(cls, packed):
        """Unpack from a single array of data type `dtype`.

        Useful for communication via MPI."""
        mo, t1, t2, l1, l2, projector = unpack_arrays(packed)
        mo = SpatialOrbitals.unpack(mo)
        wf = cls(mo, t1, t2, l1=l1, l2=l2, projector=projector)
        return wf

    def restore(self, projector=None, inplace=False, sym=True):
        if projector is None: projector = self.projector
        wf = self.project(projector.T, inplace=inplace)
        wf.projector = None
        if not sym:
            return wf
        wf.t2 = symmetrize_c2(wf.t2)
        if wf.l2 is None:
            return wf
        wf.l2 = symmetrize_c2(wf.l2)
        return wf

    def copy(self):
        t1 = spinalg.copy(self.t1)
        t2 = spinalg.copy(self.t2)
        l1 = callif(spinalg.copy, self.l1)
        l2 = callif(spinalg.copy, self.l2)
        proj = callif(spinalg.copy, self.projector)
        return type(self)(self.mo.copy(), t1, t2, l1=l1, l2=l2, projector=proj)

    def as_unrestricted(self):
        if self.projector is not None:
            raise NotImplementedError
        mo = self.mo.to_spin_orbitals()
        def _to_uccsd(t1, t2):
            t1, t2 = self.t1.copy, self.t2.copy()
            t2aa = t2 - t2.transpose(0,1,3,2)
            return (t1, t1), (t2aa, t2, t2aa)
        t1, t2 = _to_uccsd(self.t1, self.t2)
        l1 = l2 = None
        if self.l1 is not None and self.l2 is not None:
            l1, l2 = _to_uccsd(self.l1, self.l2)
        return UCCSD_WaveFunction(mo, t1, t2, l1=l1, l2=l2)

    def as_mp2(self):
        raise NotImplementedError

    def as_cisd(self, c0=1.0):
        """In intermediate normalization."""
        if self.projector is not None:
            raise NotImplementedError
        c1 = c0*self.t1
        c2 = c0*(self.t2 + einsum('ia,jb->ijab', self.t1, self.t1))
        return wf.RCISD_WaveFunction(self.mo, c0, c1, c2, projector=self.projector)

    def as_ccsd(self):
        return self

    def as_fci(self):
        raise NotImplementedError


class UCCSD_WaveFunction(RCCSD_WaveFunction):

    # TODO
    #_make_rdm1_backend = pyscf.cc.uccsd_rdm.make_rdm1
    #_make_rdm2_backend = pyscf.cc.uccsd_rdm.make_rdm2
    _make_rdm1_backend = uccsd_rdm.make_rdm1
    _make_rdm2_backend = uccsd_rdm.make_rdm2

    # Spin blocks of T-Amplitudes

    @property
    def t1a(self):
        return self.t1[0]

    @property
    def t1b(self):
        return self.t1[1]

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
        return self.t2ab.transpose(1,0,3,2)

    @property
    def t2bb(self):
        return self.t2[-1]

    # Spin blocks of L-Amplitudes

    @property
    def l1a(self):
        return self.l1[0]

    @property
    def l1b(self):
        return self.l1[1]

    @property
    def l2aa(self):
        return self.l2[0]

    @property
    def l2ab(self):
        return self.l2[1]

    @property
    def l2ba(self):
        if len(self.l2) == 4:
            return self.l2[2]
        return self.l2ab.transpose(1,0,3,2)

    @property
    def l2bb(self):
        return self.l2[-1]

    def make_rdm2_non_cumulant(self, t_as_lambda=False, ao_basis=False):
        dm1a, dm1b = self.make_rdm1(t_as_lambda=t_as_lambda, with_mf=False, ao_basis=ao_basis)
        dm2aa = (einsum('ij,kl->ijkl', dm1a, dm1a) - einsum('ij,kl->iklj', dm1a, dm1a))
        dm2bb = (einsum('ij,kl->ijkl', dm1b, dm1b) - einsum('ij,kl->iklj', dm1b, dm1b))
        dm2ab = einsum('ij,kl->ijkl', dm1a, dm1b)
        dm2 = (dm2aa, dm2ab, dm2bb)
        return dm2

    def project(self, projector, inplace=False):
        wf = self if inplace else self.copy()
        wf.t1 = project_uc1(wf.t1, projector)
        wf.t2 = project_uc2(wf.t2, projector)
        wf.l1 = project_uc1(wf.l1, projector)
        wf.l2 = project_uc2(wf.l2, projector)
        wf.projector = projector
        return wf

    def restore(self, projector=None, inplace=False, sym=True):
        if projector is None: projector = self.projector
        wf = self.project((projector[0].T, projector[1].T), inplace=inplace)
        wf.projector = None
        if not sym:
            return wf
        wf.t2 = symmetrize_uc2(wf.t2)
        if self.l2 is None:
            return wf
        wf.l2 = symmetrize_uc2(wf.l2)
        return wf

    def as_mp2(self):
        raise NotImplementedError

    def as_cisd(self, c0=1.0):
        if self.projector is not None:
            raise NotImplementedError
        c1a = c0*self.t1a
        c1b = c0*self.t1b
        c2aa = c0*(self.t2aa + einsum('ia,jb->ijab', self.t1a, self.t1a)
                             - einsum('ib,ja->ijab', self.t1a, self.t1a))
        c2bb = c0*(self.t2bb + einsum('ia,jb->ijab', self.t1b, self.t1b)
                             - einsum('ib,ja->ijab', self.t1b, self.t1b))
        c2ab = c0*(self.t2ab + einsum('ia,jb->ijab', self.t1a, self.t1b))
        c1 = (c1a, c1b)
        if len(self.t2) == 3:
            c2 = (c2aa, c2ab, c2bb)
        elif len(self.t2) == 4:
            c2ba = c0*self.t2ba + einsum('ia,jb->ijab', c1b, c1a)
            c2 = (c2aa, c2ab, c2ba, c2bb)
        return wf.UCISD_WaveFunction(self.mo, c0, c1, c2, projector=self.projector)

    def as_fci(self):
        raise NotImplementedError

    def multiply(self, factor):
        self.t1 = spinalg.multiply(self.t1, len(self.t1)*[factor])
        self.t2 = spinalg.multiply(self.t2, len(self.t2)*[factor])
        if self.l1 is not None:
            self.l1 = spinalg.multiply(self.l1, len(self.l1)*[factor])
        if self.l2 is not None:
            self.l2 = spinalg.multiply(self.l2, len(self.l2)*[factor])

    #def pack(self, dtype=float):
    #    """Pack into a single array of data type `dtype`.

    #    Useful for communication via MPI."""
    #    mo = self.mo.pack(dtype=dtype)
    #    l1 = self.l1 is not None else [None, None]
    #    l2 = self.l2 is not None else len(self.t2)*[None]
    #    projector = self.projector is not None else [None]
    #    data = (mo, *self.t1, *self.t2, *l1, *l2, *projector)
    #    pack = pack_arrays(*data, dtype=dtype)
    #    return pack

    #@classmethod
    #def unpack(cls, packed):
    #    """Unpack from a single array of data type `dtype`.

    #    Useful for communication via MPI."""
    #    mo, *unpacked = unpack_arrays(packed)
    #    mo = SpinOrbitals.unpack(mo)
    #    t1a, t1b, t2, l1, l2, projector =
    #    wf = cls(mo, t1, t2, l1=l1, l2=l2)
    #    if projector is not None:
    #        wf.projector = projector
    #    return wf
