import numpy as np

import pyscf
import pyscf.scf
import pyscf.mp
import pyscf.ci
import pyscf.cc
import pyscf.fci

import vayesta
from vayesta.core.util import *
from vayesta.core.types.orbitals import *
from vayesta.core.helper import pack_arrays, unpack_arrays

__all__ = [
        'WaveFunction',
        'HF_WaveFunction', 'RHF_WaveFunction', 'UHF_WaveFunction',
        'MP2_WaveFunction', 'RMP2_WaveFunction', 'UMP2_WaveFunction',
        'CCSD_WaveFunction', 'RCCSD_WaveFunction', 'UCCSD_WaveFunction',
        'CISD_WaveFunction', 'RCISD_WaveFunction', 'UCISD_WaveFunction',
        'FCI_WaveFunction', 'RFCI_WaveFunction', 'UFCI_WaveFunction',
        ]

class WaveFunction:

    def __init__(self, mo, projector=None):
        self.mo = mo
        self.projector = projector

    def __repr__(self):
        return "%s(norb= %r, nocc= %r, nvir=%r)" % (self.__class__.__name__, self.norb, self.nocc, self.nvir)

    @property
    def norb(self):
        return self.mo.norb

    @property
    def nocc(self):
        return self.mo.nocc

    @property
    def nvir(self):
        return self.mo.nvir

    @property
    def norba(self):
        return self.mo.norba

    @property
    def norbb(self):
        return self.mo.norbb

    @property
    def nocca(self):
        return self.mo.nocca

    @property
    def noccb(self):
        return self.mo.noccb

    @property
    def nvira(self):
        return self.mo.nvira

    @property
    def nvirb(self):
        return self.mo.nvirb

    @property
    def nelec(self):
        return self.mo.nelec

    def make_rdm1(self, *args, **kwargs):
        raise AbstractMethodError

    def make_rdm2(self, *args, **kwargs):
        raise AbstractMethodError

    @staticmethod
    def from_pyscf(obj, **kwargs):
        # MP2
        if isinstance(obj, pyscf.mp.ump2.UMP2):
            from pyscf.mp.ump2 import _ChemistsERIs
            eris = kwargs.get('eris', _ChemistsERIs()._common_init_(obj))
            mo = SpinOrbitals(eris.mo_coeff, energy=eris.mo_energy, occ=obj.nocc)
            wf = UMP2_WaveFunction(mo, obj.t2)
            return wf
        if isinstance(obj, pyscf.mp.mp2.MP2):
            from pyscf.mp.mp2 import _ChemistsERIs
            eris = kwargs.get('eris', _ChemistsERIs()._common_init_(obj))
            mo = SpatialOrbitals(eris.mo_coeff, energy=eris.mo_energy, occ=obj.nocc)
            wf = RMP2_WaveFunction(mo, obj.t2)
            return wf
        # CCSD
        if isinstance(obj, pyscf.cc.uccsd.UCCSD):
            from pyscf.cc.uccsd import _ChemistsERIs
            eris = kwargs.get('eris', _ChemistsERIs()._common_init_(obj))
            mo = SpinOrbitals(eris.mo_coeff, energy=eris.mo_energy, occ=obj.nocc)
            wf = UCCSD_WaveFunction(mo, obj.t1, obj.t2, l1=obj.l1, l2=obj.l2)
            return wf
        if isinstance(obj, pyscf.cc.ccsd.CCSD):
            from pyscf.cc.ccsd import _ChemistsERIs
            eris = kwargs.get('eris', _ChemistsERIs()._common_init_(obj))
            mo = SpatialOrbitals(eris.mo_coeff, energy=eris.mo_energy, occ=obj.nocc)
            wf = RCCSD_WaveFunction(mo, obj.t1, obj.t2, l1=obj.l1, l2=obj.l2)
            return wf
        # CISD
        if isinstance(obj, pyscf.ci.ucisd.UCISD):
            from pyscf.cc.uccsd import _ChemistsERIs
            eris = kwargs.get('eris', _ChemistsERIs()._common_init_(obj))
            mo = SpinOrbitals(eris.mo_coeff, energy=eris.mo_energy, occ=obj.nocc)
            c0, c1, c2 = obj.cisdvec_to_amplitudes(obj.ci)
            wf = UCISD_WaveFunction(mo, c0, c1, c2)
            return wf
        if isinstance(obj, pyscf.ci.cisd.CISD):
            from pyscf.cc.ccsd import _ChemistsERIs
            eris = kwargs.get('eris', _ChemistsERIs()._common_init_(obj))
            mo = SpatialOrbitals(eris.mo_coeff, energy=eris.mo_energy, occ=obj.nocc)
            c0, c1, c2 = obj.cisdvec_to_amplitudes(obj.ci)
            wf = RCISD_WaveFunction(mo, c0, c1, c2)
            return wf
        # FCI
        if isinstance(obj, pyscf.fci.direct_uhf.FCISolver):
            mo = kwargs['mo']
            wf = UFCI_WaveFunction(mo, obj.ci)
            return wf
        if isinstance(obj, pyscf.fci.direct_spin1.FCISolver):
            mo = kwargs['mo']
            wf = RFCI_WaveFunction(mo, obj.ci)
            return wf
        raise NotImplementedError

# --- HF

class RHF_WaveFunction(WaveFunction):

    def make_rdm1(self, mo_coeff=None, mo_occ=None, ao_basis=True):
        if mo_occ is None: mo_occ = self.mo.occ
        if not ao_basis:
            return np.diag(mo_occ)
        if mo_coeff is None: mo_coeff = self.mo.coeff
        occ = (mo_occ > 0)
        return np.dot(mo_coeff[:,occ]*mo_occ[occ], mo_coeff[:,occ].T)

    def make_rdm2(self, mo_coeff=None, mo_occ=None, ao_basis=True):
        dm1 = self.make_rdm1(mo_coeff=mo_coeff, mo_occ=mo_occ, ao_basis=ao_basis)
        dm2 = einsum('ij,kl->ijkl', dm1, dm1) - einsum('ij,kl->iklj', dm1, dm1)/2
        return dm2

    def as_restricted(self):
        return self

    def as_unrestricted(self):
        raise NotImplementedError

class UHF_WaveFunction(RHF_WaveFunction):

    def make_rdm1(self, mo_coeff=None, mo_occ=None, ao_basis=True):
        if mo_coeff is None: mo_coeff = self.mo.coeff
        if mo_occ is None: mo_occ = self.mo.occ
        dm1a = super().make_rdm1(mo_coeff=mo_coeff[0], mo_occ=mo_occ[0], ao_basis=ao_basis)
        dm1b = super().make_rdm1(mo_coeff=mo_coeff[1], mo_occ=mo_occ[1], ao_basis=ao_basis)
        return (dm1a, dm1b)

    def make_rdm2(self, mo_coeff=None, mo_occ=None, ao_basis=True):
        dm1a, dm1b = self.make_rdm1(mo_coeff=mo_coeff, mo_occ=mo_occ, ao_basis=ao_basis)
        dm2aa = einsum('ij,kl->ijkl', dm1a, dm1a) - einsum('ij,kl->iklj', dm1a, dm1a)
        dm2bb = einsum('ij,kl->ijkl', dm1b, dm1b) - einsum('ij,kl->iklj', dm1b, dm1b)
        dm2ab = einsum('ij,kl->ijkl', dm1a, dm1b)
        return (dm2aa, dm2ab, dm2bb)

    def to_restricted(self):
        raise NotImplementedError

    def as_unrestricted(self):
        return self

def HF_WaveFunction(mo):
    if mo.nspin == 1:
        cls = RHF_WaveFunction
    elif mo.nspin == 2:
        cls = UHF_WaveFunction
    return cls(mo)

# --- Helper

def project_c1(c1, p):
    if c1 is None: return None
    if p is None: return c1
    return np.dot(p, c1)

def project_c2(c2, p):
    if c2 is None: return None
    if p is None: return c2
    return np.tensordot(p, c2, axes=1)

def project_uc1(c1, p):
    if c1 is None: return None
    if p is None: return c1
    return (project_c1(c1[0], p[0]),
            project_c1(c1[1], p[1]))

def project_uc2(c2, p):
    if c2 is None: return None
    if p is None: return c2
    c2ba = (c2[2] if len(c2) == 4 else c2[1].transpose(1,0,3,2))
    return (project_c2(c2[0], p[0]),
            project_c2(c2[1], p[0]),
            #einsum('xi,ij...->ix...', p[1], c2[1]),
            project_c2(c2ba, p[1]),
            project_c2(c2[-1], p[1]))

def symmetrize_c2(c2, inplace=True):
    if not inplace:
        c2 = c2.copy()
    c2 += c2.transpose(1,0,3,2)
    c2 /= 2
    return c2

def symmetrize_uc2(c2, inplace=True):
    if not inplace:
        c2 = tuple(x.copy() for x in c2)

    # alpha-alpha:
    #c2[0][:] += c2[0].transpose(1,0,3,2) - c2[0].transpose(1,0,2,3) - c2[0].transpose(0,1,3,2)
    #c2[0][:] /= 4
    ## beta-beta:
    #c2[-1][:] += c2[-1].transpose(1,0,3,2) - c2[-1].transpose(1,0,2,3) - c2[-1].transpose(0,1,3,2)
    #c2[-1][:] /= 4
    # alpha-alpha:
    c2[0][:] += c2[0].transpose(1,0,3,2)
    c2[0][:] /= 2
    # beta-beta:
    c2[-1][:] += c2[-1].transpose(1,0,3,2)
    c2[-1][:] /= 2
    # alpha-beta and beta-alpha:
    if len(c2) == 4:
        c2ab = (c2[1] + c2[2].transpose(1,0,3,2))/2
        #c2 = (c2[0], c2ab, c2ab.transpose(1,0,3,2), c2[3])
        c2 = (c2[0], c2ab, c2[3])
    return c2

# --- MP2

class RMP2_WaveFunction(WaveFunction):

    def __init__(self, mo, t2, projector=None):
        super().__init__(mo, projector=projector)
        self.t2 = t2

    def as_restricted(self):
        return self

    def as_unrestricted(self):
        mo = self.mo.to_spin_orbitals()
        t2 = self.t2.copy()
        t2aa = (t2 - t2.transpose(0,1,3,2))
        t2 = (t2aa, t2, t2aa)
        return UMP2_WaveFunction(mo, t2)

    def project(self, projector, inplace=False):
        wf = self if inplace else self.copy()
        wf.t2 = project_c2(wf.t2, projector)
        wf.projector = projector
        return wf

    def restore(self, projector=None, inplace=False, sym=True):
        if projector is None: projector = self.projector
        wf = self.project(projector.T, inplace=inplace)
        if not sym:
            return wf
        wf.t2 = symmetrize_c2(wf.t2)
        wf.projector = None
        return wf

    def as_mp2(self):
        return self

    def as_ccsd(self):
        nocc1 = self.t2.shape[0]
        t1 = np.zeros((nocc1, self.nvir))
        #return CCSD_WaveFunction(self.mo, t1, self.t2, projector=self.projector)
        return CCSD_WaveFunction(self.mo, t1, self.t2, l1=t1, l2=self.t2, projector=self.projector)

    def to_cisd(self, c0=1.0):
        nocc1 = self.t2.shape[0]
        c1 = np.zeros((nocc1, self.nvir))
        c2 = c0*self.t2
        return RCISD_WaveFunction(self.mo, c0, c1, c2, projector=self.projector)

    def as_fci(self):
        raise NotImplementedError

    def copy(self):
        return RMP2_WaveFunction(self.mo.copy(), self.t2.copy())

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
        wf = cls(mo, t2, projector=projector)
        return wf

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
        return self.t2ab.transpose(1,0,3,2)

    @property
    def t2bb(self):
        return self.t2[-1]

    def project(self, projector, inplace=False):
        wf = self if inplace else self.copy()
        wf.t2 = project_uc2(wf.t2, projector)
        wf.projector = projector
        return wf

    def restore(self, projector=None, inplace=False, sym=True):
        if projector is None: projector = self.projector
        wf = self.project((projector[0].T, projector[1].T), inplace=inplace)
        if not sym:
            return wf
        wf.t2 = symmetrize_uc2(wf.t2)
        wf.projector = None
        return wf

    def as_mp2(self):
        return self

    def as_ccsd(self):
        nocc1a = self.t2aa.shape[0]
        nocc1b = self.t2bb.shape[0]
        t1 = (np.zeros((nocc1a, self.nvira)),
              np.zeros((nocc1b, self.nvirb)))
        return UCCSD_WaveFunction(self.mo, t1, self.t2, l1=t1, l2=self.t2, projector=self.projector)

    def to_cisd(self, c0=1.0):
        nocc1a = self.t2aa.shape[0]
        nocc1b = self.t2bb.shape[0]
        c1 = (np.zeros((nocc1a, self.nvira)),
              np.zeros((nocc1b, self.nvirb)))
        c2aa = c0*self.t2aa
        c2ab = c0*self.t2ab
        c2bb = c0*self.t2bb
        if len(self.t2) == 3:
            c2 = (c2aa, c2ab, c2bb)
        elif len(self.t2) == 4:
            c2ba = c0*self.t2ba
            c2 = (c2aa, c2ab, c2ba, c2bb)
        return UCISD_WaveFunction(self.mo, c0, c1, c2, projector=self.projector)

    def as_fci(self):
        return NotImplementedError

    def copy(self):
        t2 = tuple(t.copy() for t in self.t2)
        return UMP2_WaveFunction(self.mo.copy(), t2)

def MP2_WaveFunction(mo, t2, **kwargs):
    if mo.nspin == 1:
        cls = RMP2_WaveFunction
    elif mo.nspin == 2:
        cls = UMP2_WaveFunction
    return cls(mo, t2, **kwargs)

# CCSD

class RCCSD_WaveFunction(WaveFunction):

    _make_rdm1_backend = pyscf.cc.ccsd_rdm.make_rdm1
    _make_rdm2_backend = pyscf.cc.ccsd_rdm.make_rdm2

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

    def make_rdm2(self, t_as_lambda=False, with_dm1=True, ao_basis=False):
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
        return dm2

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
        if not sym:
            return wf
        wf.t2 = symmetrize_c2(wf.t2)
        if wf.l2 is None:
            return wf
        wf.l2 = symmetrize_c2(wf.l2)
        wf.projector = None
        return wf

    def copy(self):
        t1 = self.t1.copy()
        t2 = self.t2.copy()
        l1 = l2 = None
        if self.l1 is not None:
            l1 = self.l1.copy()
        if self.l2 is not None:
            l2 = self.l2.copy()
        return RCCSD_WaveFunction(self.mo.copy(), t1, t2, l1=l1, l2=l2)

    def as_unestricted(self):
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

    def to_mp2(self):
        raise NotImplementedError

    def as_ccsd(self):
        return self

    def to_cisd(self, c0=1.0):
        """In intermediate normalization."""
        c1 = c0*self.t1
        c2 = c0*(self.t2 + einsum('ia,jb->ijab', self.t1, self.t1))
        return RCISD_WaveFunction(self.mo, c0, c1, c2, projector=self.projector)

    def to_fci(self):
        raise NotImplementedError


class UCCSD_WaveFunction(RCCSD_WaveFunction):

    _make_rdm1_backend = pyscf.cc.uccsd_rdm.make_rdm1
    _make_rdm2_backend = pyscf.cc.uccsd_rdm.make_rdm2

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
        if not sym:
            return wf
        wf.t2 = symmetrize_uc2(wf.t2)
        if self.l2 is None:
            return wf
        wf.l2 = symmetrize_uc2(wf.l2)
        wf.projector = None
        return wf

    def copy(self):
        t1 = tuple(t.copy() for t in self.t1)
        t2 = tuple(t.copy() for t in self.t2)
        l1 = l2 = None
        if self.l1 is not None:
            l1 = tuple(t.copy() for t in self.l1)
        if self.l2 is not None:
            l2 = tuple(t.copy() for t in self.l2)
        return UCCSD_WaveFunction(self.mo.copy(), t1, t2, l1=l1, l2=l2)

    def to_mp2(self):
        raise NotImplementedError

    def as_ccsd(self):
        return self

    def to_cisd(self, c0=1.0):
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
        return UCISD_WaveFunction(self.mo, c0, c1, c2, projector=self.projector)

    def to_fci(self):
        raise NotImplementedError

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


def CCSD_WaveFunction(mo, t1, t2, **kwargs):
    if mo.nspin == 1:
        cls = RCCSD_WaveFunction
    elif mo.nspin == 2:
        cls = UCCSD_WaveFunction
    return cls(mo, t1, t2, **kwargs)

# --- CISD

class RCISD_WaveFunction(WaveFunction):

    def __init__(self, mo, c0, c1, c2, projector=None):
        super().__init__(mo, projector=projector)
        self.c0 = c0
        self.c1 = c1
        self.c2 = c2

    def project(self, projector, inplace=False):
        wf = self if inplace else self.copy()
        wf.c1 = project_c1(wf.c1, projector)
        wf.c2 = project_c2(wf.c2, projector)
        wf.projector = projector
        return wf

    def restore(self, projector=None, inplace=False, sym=True):
        if projector is None: projector = self.projector
        wf = self.project(projector.T, inplace=inplace)
        if not sym:
            return wf
        wf.c2 = symmetrize_c2(wf.c2)
        wf.projector = None
        return wf

    def copy(self):
        return RCISD_WaveFunction(self.mo.copy(), self.c0, self.c1.copy(), self.c2.copy())

    def as_mp2(self):
        raise NotImplementedError

    def as_ccsd(self):
        t1 = self.c1/self.c0
        t2 = self.c2/self.c0 - einsum('ia,jb->ijab', t1, t1)
        return RCCSD_Wavefunction(self.mo, t1, t2, projector=self.projector)

    def as_cisd(self, c0=None):
        if c0 is None:
            return self
        c1 = self.c1 * c0/self.c0
        c2 = self.c2 * c0/self.c0
        return RCISD_WaveFunction(self.mo, c0, c1, c2, projector=self.projector)

    def as_fci(self):
        raise NotImplementedError

class UCISD_WaveFunction(RCISD_WaveFunction):

    @property
    def c1a(self):
        return self.c1[0]

    @property
    def c1b(self):
        return self.c1[1]

    @property
    def c2aa(self):
        return self.c2[0]

    @property
    def c2ab(self):
        return self.c2[1]

    @property
    def c2ba(self):
        if len(self.c2) == 4:
            return self.c2[2]
        return self.c2ab.transpose(1,0,3,2)

    @property
    def c2bb(self):
        return self.c2[-1]

    def project(self, projector, inplace=False):
        wf = self if inplace else self.copy()
        wf.c1 = project_uc1(wf.c1, projector)
        wf.c2 = project_uc2(wf.c2, projector)
        wf.projector = projector
        return wf

    def restore(self, projector=None, inplace=False, sym=True):
        if projector is None: projector = self.projector
        wf = self.project((projector[0].T, projector[1].T), inplace=inplace)
        if not sym:
            return wf
        wf.c2 = symmetrize_uc2(wf.c2)
        wf.projector = None
        return wf

    def copy(self):
        c1 = tuple(t.copy() for t in self.c1)
        c2 = tuple(t.copy() for t in self.c2)
        return UCISD_WaveFunction(self.mo.copy(), self.c0, c1, c2)

    def as_mp2(self):
        raise NotImplementedError

    def as_ccsd(self):
        t1a = self.c1a/self.c0
        t1b = self.c1b/self.c0
        t1 = (t1a, t1b)
        t2aa = self.c2aa/self.c0 - einsum('ia,jb->ijab', t1a, t1a) + einsum('ib,ja->ijab', t1a, t1a)
        t2bb = self.c2bb/self.c0 - einsum('ia,jb->ijab', t1b, t1b) + einsum('ib,ja->ijab', t1b, t1b)
        t2ab = self.c2ab/self.c0 - einsum('ia,jb->ijab', t1a, t1b)
        if len(self.c2) == 3:
            t2 = (t2aa, t2ab, t2bb)
        elif len(self.c2) == 4:
            t2ba = self.c2ab/self.c0 - einsum('ia,jb->ijab', t1b, t1a)
            t2 = (t2aa, t2ab, t2ba, t2bb)
        return UCCSD_WaveFunction(self.mo, t1, t2, projector=self.projector)

    def as_cisd(self, c0=None):
        if c0 is None:
            return self
        c1a = self.c1a * c0/self.c0
        c1b = self.c1b * c0/self.c0
        c2aa = self.c2aa * c0/self.c0
        c2ab = self.c2ab * c0/self.c0
        c2bb = self.c2bb * c0/self.c0
        c1 = (c1a, c1b)
        if len(self.c2) == 3:
            c2 = (c2aa, c2ab, c2bb)
        elif len(self.c2) == 4:
            c2ba = self.c2ba * c0/self.c0
            c2 = (c2aa, c2ab, c2ba, c2bb)
        return UCISD_WaveFunction(self.mo, c0, c1, c2, projector=self.projector)

    def as_fci(self):
        raise NotImplementedError


def CISD_WaveFunction(mo, c0, c1, c2, **kwargs):
    if mo.nspin == 1:
        cls = RCISD_WaveFunction
    elif mo.nspin == 2:
        cls = UCISD_WaveFunction
    return cls(mo, c0, c1, c2, **kwargs)

# --- FCI

class RFCI_WaveFunction(WaveFunction):

    _make_rdm1_backend = pyscf.fci.direct_spin1.make_rdm1
    _make_rdm2_backend = pyscf.fci.direct_spin1.make_rdm12

    def __init__(self, mo, ci, projector=None):
        super().__init__(mo, projector=projector)
        self.ci = ci

    def make_rdm1(self):
        return type(self)._make_rdm1_backend(self.ci, self.norb, self.nelec)

    def make_rdm2(self):
        return type(self)._make_rdm2_backend(self.ci, self.norb, self.nelec)[1]

    def project(self, projector, inplace=False):
        raise NotImplementedError

    def restore(self, projector=None, inplace=False):
        raise NotImplementedError

    @property
    def c0(self):
        return self.ci[0,0]

    def as_unrestricted(self):
        mo = self.mo.to_spin_orbitals()
        return UFCI_WaveFunction(mo, self.ci)

    def as_mp2(self):
        raise self.as_cisd().as_mp2()

    def as_ccsd(self):
        return self.as_cisd().as_ccsd()

    def as_cisd(self, c0=None):
        norb, nocc, nvir = self.norb, self.nocc, self.nvir
        t1addr, t1sign = pyscf.ci.cisd.t1strs(norb, nocc)
        c1 = self.ci[0,t1addr] * t1sign
        c2 = einsum('i,j,ij->ij', t1sign, t1sign, self.ci[t1addr[:,None],t1addr])
        c1 = c1.reshape(nocc,nvir)
        c2 = c2.reshape(nocc,nvir,nocc,nvir).transpose(0,2,1,3)
        if c0 is None:
            c0 = self.c0
        else:
            c1 *= c0/self.c0
            c2 *= c0/self.c0
        return RCISD_WaveFunction(self.mo, c0, c1, c2, projector=self.projector)

    def as_fci(self):
        return self

class UFCI_WaveFunction(RFCI_WaveFunction):

    _make_rdm1_backend = pyscf.fci.direct_spin1.make_rdm1s
    _make_rdm2_backend = pyscf.fci.direct_spin1.make_rdm12s

    def make_rdm1(self):
        assert (self.norb[0] == self.norb[1])
        return type(self)._make_rdm1_backend(self.ci, self.norb[0], self.nelec)

    def make_rdm2(self):
        assert (self.norb[0] == self.norb[1])
        return type(self)._make_rdm2_backend(self.ci, self.norb[0], self.nelec)[1]

    def as_cisd(self, c0=None):
        norba, norbb = self.norb
        nocca, noccb = self.nocc
        nvira, nvirb = self.nvir

        t1addra, t1signa = pyscf.ci.cisd.tn_addrs_signs(norba, nocca, 1)
        t1addrb, t1signb = pyscf.ci.cisd.tn_addrs_signs(norbb, noccb, 1)
        t2addra, t2signa = pyscf.ci.cisd.tn_addrs_signs(norba, nocca, 2)
        t2addrb, t2signb = pyscf.ci.cisd.tn_addrs_signs(norbb, noccb, 2)
        na = pyscf.fci.cistring.num_strings(norba, nocca)
        nb = pyscf.fci.cistring.num_strings(norbb, noccb)

        ci = self.ci.reshape(na,nb)
        c1a = (self.ci[t1addra,0] * t1signa).reshape(nocca,nvira)
        c1b = (self.ci[0,t1addrb] * t1signb).reshape(noccb,nvirb)

        nocca_comp = nocca*(nocca-1)//2
        noccb_comp = noccb*(noccb-1)//2
        nvira_comp = nvira*(nvira-1)//2
        nvirb_comp = nvirb*(nvirb-1)//2
        c2aa = (self.ci[t2addra,0] * t2signa).reshape(nocca_comp, nvira_comp)
        c2bb = (self.ci[0,t2addrb] * t2signb).reshape(noccb_comp, nvirb_comp)
        c2aa = pyscf.cc.ccsd._unpack_4fold(c2aa, nocca, nvira)
        c2bb = pyscf.cc.ccsd._unpack_4fold(c2bb, noccb, nvirb)
        c2ab = einsum('i,j,ij->ij', t1signa, t1signb, self.ci[t1addra[:,None],t1addrb])
        c2ab = c2ab.reshape(nocca,nvira,noccb,nvirb).transpose(0,2,1,3)
        if c0 is None:
            c0 = self.c0
        else:
            c1a *= c0/self.c0
            c1b *= c0/self.c0
            c2aa *= c0/self.c0
            c2ab *= c0/self.c0
            c2bb *= c0/self.c0
        c1 = (c1a, c1b)
        c2 = (c2aa, c2ab, c2bb)
        return UCISD_WaveFunction(self.mo, c0, c1, c2, projector=self.projector)

def FCI_WaveFunction(mo, ci, **kwargs):
    if mo.nspin == 1:
        cls = RFCI_WaveFunction
    elif mo.nspin == 2:
        cls = UFCI_WaveFunction
    return cls(mo, ci, **kwargs)


if __name__ == '__main__':

    import pyscf.gto

    mol = pyscf.gto.Mole()
    mol.atom = 'H 0 0 0; H 0 0 0.74'
    mol.basis = 'cc-pVDZ'
    mol.build()

    rhf = pyscf.scf.RHF(mol)
    rhf.kernel()

    uhf = rhf.to_uhf()

    def test_rmp2():
        mp = pyscf.mp.MP2(rhf)
        mp.kernel()
        wf = WaveFunction.from_pyscf(mp)

    def test_ump2():
        mp = pyscf.mp.UMP2(uhf)
        mp.kernel()
        wf = WaveFunction.from_pyscf(mp)

    def test_rccsd():
        cc = pyscf.cc.CCSD(rhf)
        cc.kernel()
        wf = WaveFunction.from_pyscf(cc)
        return wf

    def test_uccsd():
        cc = pyscf.cc.UCCSD(uhf)
        cc.kernel()
        wf = WaveFunction.from_pyscf(cc)

    #test_rmp2()
    #test_ump2()
    wf = test_rccsd()
    array = wf.pack()
    #wf2 = RCCSD_WaveFunction.unpack(array)
    wf2 = wf.copy()

    print(id(wf.mo))
    print(id(wf2.mo))
    print(np.all(wf.mo.coeff == wf2.mo.coeff))
    1/0

    print(np.all(wf.t1 == wf2.t1))
    print(np.all(wf.t2 == wf2.t2))
    print(wf.l1)
    print(wf2.l1)

    #test_uccsd()
    1/0

    norb = 6
    nocc = 2
    nvir = norb - nocc
    mo_energy = np.random.rand(norb)
    mo_occ = np.asarray(nocc*[2] + nvir*[0])
    mo_coeff = np.random.rand(norb, norb)

    mo = SpatialOrbitals(mo_coeff, mo_energy, mo_occ)


    so = SpinOrbitals(2*[mo_energy], 2*[mo_coeff], 2*[np.asarray(mo_occ)/2])
    rhf = HF_WaveFunction(mo)
    uhf = HF_WaveFunction(so)

    print(mo)
    print(so)
    print(rhf)
    print(uhf)


    t1 = np.random.rand(nocc,nvir)
    t2 = np.random.rand(nocc,nocc,nvir,nvir)
    psi = CCSD_WaveFunction(mo, t1, t2)
    print(psi)
    psi = psi.to_cisd()
    print(psi)

    t1 = (t1, t1)
    t2 = (t2, t2, t2)
    psi = CCSD_WaveFunction(so, t1, t2)
    print(psi)
    psi = psi.to_cisd()
    print(psi)
