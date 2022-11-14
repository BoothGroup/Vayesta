import importlib
import numpy as np
import pyscf
import pyscf.scf
import pyscf.mp
import pyscf.ci
import pyscf.cc
import pyscf.fci
import vayesta
from vayesta.core.util import *
from vayesta.core.types import wf
from vayesta.core.types.orbitals import *


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
        # HF
        # TODO
        def eris_init(obj, mod):
            if 'eris' in kwargs:
                return kwargs['eris']
            eris = importlib.import_module(mod.__name__)._ChemistsERIs()
            eris._common_init_(obj)
            return eris
        # MP2
        if isinstance(obj, pyscf.mp.ump2.UMP2):
            eris = eris_init(obj, pyscf.mp.ump2)
            mo = SpinOrbitals(eris.mo_coeff, energy=eris.mo_energy, occ=obj.nocc)
            return wf.UMP2_WaveFunction(mo, obj.t2)
        if isinstance(obj, pyscf.mp.mp2.MP2):
            eris = eris_init(obj, pyscf.mp.mp2)
            mo = SpatialOrbitals(eris.mo_coeff, energy=eris.mo_energy, occ=obj.nocc)
            return wf.RMP2_WaveFunction(mo, obj.t2)
        # CCSD
        if isinstance(obj, pyscf.cc.uccsd.UCCSD):
            eris = eris_init(obj, pyscf.cc.uccsd)
            mo = SpinOrbitals(eris.mo_coeff, energy=eris.mo_energy, occ=obj.nocc)
            return wf.UCCSD_WaveFunction(mo, obj.t1, obj.t2, l1=obj.l1, l2=obj.l2)
        if isinstance(obj, pyscf.cc.ccsd.CCSD):
            eris = eris_init(obj, pyscf.cc.ccsd)
            mo = SpatialOrbitals(eris.mo_coeff, energy=eris.mo_energy, occ=obj.nocc)
            return wf.RCCSD_WaveFunction(mo, obj.t1, obj.t2, l1=obj.l1, l2=obj.l2)
        # CISD
        if isinstance(obj, pyscf.ci.ucisd.UCISD):
            eris = eris_init(obj, pyscf.cc.uccsd)
            mo = SpinOrbitals(eris.mo_coeff, energy=eris.mo_energy, occ=obj.nocc)
            c0, c1, c2 = obj.cisdvec_to_amplitudes(obj.ci)
            return wf.UCISD_WaveFunction(mo, c0, c1, c2)
        if isinstance(obj, pyscf.ci.cisd.CISD):
            eris = eris_init(obj, pyscf.cc.ccsd)
            mo = SpatialOrbitals(eris.mo_coeff, energy=eris.mo_energy, occ=obj.nocc)
            c0, c1, c2 = obj.cisdvec_to_amplitudes(obj.ci)
            return wf.RCISD_WaveFunction(mo, c0, c1, c2)
        # FCI
        if isinstance(obj, pyscf.fci.direct_uhf.FCISolver):
            mo = kwargs['mo']
            return wf.UFCI_WaveFunction(mo, obj.ci)
        if isinstance(obj, pyscf.fci.direct_spin1.FCISolver):
            mo = kwargs['mo']
            if isinstance(mo, np.ndarray):
                nelec = sum(obj.nelec)
                assert (nelec % 2 == 0)
                nocc = nelec // 2
                mo = SpatialOrbitals(mo, occ=nocc)
            return wf.RFCI_WaveFunction(mo, obj.ci)
        raise NotImplementedError
