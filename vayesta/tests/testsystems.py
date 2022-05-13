try:
    from functools import cache
except ImportError:
    from functools import lru_cache
    cache = lru_cache(maxsize=None)
import numpy as np

import pyscf
# Open boundary
import pyscf.gto
import pyscf.scf
import pyscf.mp
import pyscf.cc
import pyscf.fci
# Periodic boundary
import pyscf.pbc
import pyscf.pbc.gto
import pyscf.pbc.scf
import pyscf.pbc.cc
import pyscf.pbc.tools
from pyscf.pbc.scf.addons import kconj_symmetry_

import vayesta
from vayesta.misc import molecules
from vayesta.misc import solids
#from vayesta.core import fold_scf


class TestMolecule:

    def __init__(self, atom, basis, auxbasis=None, verbose=0, **kwargs):
        super().__init__()
        mol = pyscf.gto.Mole()
        mol.atom = atom
        mol.basis = basis
        for key, val in kwargs.items():
            setattr(mol, key, val)
        mol.build()
        self.auxbasis = auxbasis
        self.mol = mol

    # --- Mean-field

    @cache
    def rhf(self):
        rhf = pyscf.scf.RHF(self.mol)
        if self.auxbasis is not None:
            rhf = rhf.density_fit(auxbasis=self.auxbasis)
        rhf.kernel()
        return rhf

    @cache
    def uhf(self):
        uhf = pyscf.scf.UHF(self.mol)
        if self.auxbasis is not None:
            uhf = uhf.density_fit(auxbasis=self.auxbasis)
        uhf.kernel()
        return uhf

    # --- MP2

    @cache
    def rmp2(self):
        rmp2 = pyscf.mp.MP2(self.rhf())
        rmp2.kernel()
        return rmp2

    @cache
    def ump2(self):
        ump2 = pyscf.mp.UMP2(self.uhf())
        ump2.kernel()
        return ump2

    # --- CCSD

    @cache
    def rccsd(self):
        rccsd = pyscf.cc.RCCSD(self.rhf())
        rccsd.kernel()
        rccsd.solve_lambda()
        return rccsd

    @cache
    def uccsd(self):
        uccsd = pyscf.cc.UCCSD(self.uhf())
        uccsd.kernel()
        uccsd.solve_lambda()
        return uccsd

    # --- FCI

    @cache
    def rfci(self):
        rfci = pyscf.fci.FCI(self.rhf())
        rfci.kernel()
        return rfci

    @cache
    def ufci(self):
        ufci = pyscf.fci.FCI(self.uhf())
        ufci.kernel()
        return ufci


class TestSolid:

    def __init__(self, a, atom, basis, kmesh=None, auxbasis=None, supercell=None, verbose=0, **kwargs):
        super().__init__()
        mol = pyscf.pbc.gto.Cell()
        mol.a = a
        mol.atom = atom
        mol.basis = basis
        for key, val in kwargs.items():
            setattr(mol, key, val)
        mol.build()
        if supercell is not None:
            mol = pyscf.pbc.tools.super_cell(mol, supercell)
        self.supercell = supercell
        self.mol = mol
        #kmesh = (kmesh or supercell)
        self.kpts = self.mol.make_kpts(kmesh) if kmesh is not None else None
        self.auxbasis = auxbasis

    # --- Mean-field

    @cache
    def rhf(self):
        if self.kpts is None:
            rhf = pyscf.pbc.scf.RHF(self.mol)
        else:
            rhf = pyscf.pbc.scf.KRHF(self.mol, self.kpts)
        rhf = rhf.density_fit(auxbasis=self.auxbasis)
        if self.kpts is not None:
            rhf = kconj_symmetry_(rhf)
        rhf.conv_tol = 1e-10
        rhf.kernel()
        #if self.supercell is not None:
        #    rhf = fold_scf(rhf)
        #    self.mol = rhf.mol
        #    self.kpts = None
        return rhf

    @cache
    def uhf(self):
        if self.kpts is None:
            uhf = pyscf.pbc.scf.UHF(self.mol)
        else:
            uhf = pyscf.pbc.scf.KUHF(self.mol, self.kpts)
        uhf = uhf.density_fit(auxbasis=self.auxbasis)
        if self.kpts is not None:
            uhf = kconj_symmetry_(uhf)
        uhf.conv_tol = 1e-10
        uhf.kernel()
        #if self.supercell is not None:
        #    uhf = fold_scf(uhf)
        #    self.mol = uhf.mol
        #    self.kpts = None
        return uhf

    # --- MP2

    @cache
    def rmp2(self):
        mf = self.rhf()
        if self.kpts is None:
            rmp2 = pyscf.pbc.mp.RMP2(mf)
        else:
            rmp2 = pyscf.pbc.mp.KRMP2(mf)
        rmp2.kernel()
        return rmp2

    @cache
    def ump2(self):
        mf = self.uhf()
        if self.kpts is None:
            ump2 = pyscf.pbc.mp.UMP2(mf)
        else:
            ump2 = pyscf.pbc.mp.KUMP2(mf)
        ump2.kernel()
        return ump2

    # --- CCSD

    @cache
    def rccsd(self):
        mf = self.rhf()
        if self.kpts is None:
            rccsd = pyscf.pbc.cc.RCCSD(mf)
        else:
            rccsd = pyscf.pbc.cc.KRCCSD(mf)
        rccsd.conv_tol = 1e-10
        rccsd.conv_tol_normt = 1e-8
        rccsd.kernel()
        if self.kpts is None:
            rccsd.solve_lambda()
        return rccsd

    @cache
    def uccsd(self):
        mf = self.uhf()
        if self.kpts is None:
            uccsd = pyscf.pbc.cc.UCCSD(mf)
        else:
            uccsd = pyscf.pbc.cc.KUCCSD(mf)
        uccsd.conv_tol = 1e-10
        uccsd.conv_tol_normt = 1e-8
        uccsd.kernel()
        if self.kpts is None:
            uccsd.solve_lambda()
        return uccsd

# --- Test Systems

# Molecules

h2_dz = TestMolecule(
    atom = "H 0 0 0 ; H 0 0 0.74",
    basis = 'cc-pVDZ')

h2anion_dz = TestMolecule(
    atom = "H 0 0 0 ; H 0 0 0.74",
    basis = 'cc-pVDZ',
    charge=-1, spin=1)

#lih_dz = TestMolecule(
#    atom = "Li 0 0 0 ; H 0 0 1.595",
#    basis = 'cc-pVDZ')
#

water_sto3g = TestMolecule(atom=molecules.water(), basis='sto3g')
water_cation_sto3g = TestMolecule(atom=molecules.water(), basis='sto3g', charge=1, spin=1)

water_631g = TestMolecule(atom=molecules.water(), basis='6-31G')
water_631g_df = TestMolecule(atom=molecules.water(), basis='6-31G', auxbasis='6-31G')
water_cation_631g = TestMolecule(atom=molecules.water(), basis='6-31G', charge=1, spin=1)
water_cation_631g_df = TestMolecule(atom=molecules.water(), basis='6-31G', auxbasis='6-31G', charge=1, spin=1)

# Solids

a = 2*np.eye(3)
a[2,2] = 4
nk = 3
opts = dict(basis='sto-3g', auxbasis='sto-3g', exp_to_discard=0.1)

# 3D
h2_sto3g_k311 = TestSolid(a=a, atom='H 0 0 0 ; H 0 0 0.74', kmesh=(nk,1,1), **opts)
h2_sto3g_s311 = TestSolid(a=a, atom='H 0 0 0 ; H 0 0 0.74', supercell=(nk,1,1), **opts)

h3_sto3g_k311 = TestSolid(a=a, atom='H 0 0 0 ; H 0 0 1 ; H 0 0 2', kmesh=(nk,1,1), spin=3, **opts)
h3_sto3g_s311 = TestSolid(a=a, atom='H 0 0 0 ; H 0 0 1 ; H 0 0 2', supercell=(nk,1,1), spin=3, **opts)

# 2D
a = a.copy()
a[2,2] = 15.0
opts['dimension'] = 2
h2_sto3g_k31 = TestSolid(a=a, atom='H 0 0 0 ; H 0 0 0.74', kmesh=(nk,1,1), **opts)
h2_sto3g_s31 = TestSolid(a=a, atom='H 0 0 0 ; H 0 0 0.74', supercell=(nk,1,1), **opts)

h3_sto3g_k31 = TestSolid(a=a, atom='H 0 0 0 ; H 0 0 1 ; H 0 0 2', kmesh=(nk,1,1), spin=3, **opts)
h3_sto3g_s31 = TestSolid(a=a, atom='H 0 0 0 ; H 0 0 1 ; H 0 0 2', supercell=(nk,1,1), spin=3, **opts)


# --- Diamond cc-pVDZ
a, atom = solids.diamond()

opts = dict(basis='sto-3g', auxbasis='sto-3g', exp_to_discard=0.1)
mesh = (2,1,1)
diamond_sto3g_k211 = TestSolid(a=a, atom=atom, kmesh=mesh, **opts)
diamond_sto3g_s211 = TestSolid(a=a, atom=atom, supercell=mesh, **opts)

opts = dict(basis='sto3g', auxbasis='sto3g', exp_to_discard=0.1)
mesh = (3,3,3)
diamond_sto3g_k333 = TestSolid(a=a, atom=atom, kmesh=mesh, **opts)
diamond_sto3g_s333 = TestSolid(a=a, atom=atom, supercell=mesh, **opts)

#opts = dict(basis='cc-pVDZ', auxbasis='cc-pVDZ-ri', exp_to_discard=0.1)
#mesh = (3,3,3)
#diamond_dz_k333 = TestSolid(a=a, atom=atom, kmesh=mesh, **opts)
#diamond_dz_s333 = TestSolid(a=a, atom=atom, supercell=mesh, **opts)


# 2D
a, atom = solids.graphene()
opts = dict(basis='def2-svp', auxbasis='def2-svp-ri', exp_to_discard=0.1)
graphene_k22 = TestSolid(a=a, atom=atom, kmesh=(2,2,1), **opts)
graphene_s22 = TestSolid(a=a, atom=atom, supercell=(2,2,1), **opts)
