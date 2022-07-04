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
import pyscf.tools.ring
# Periodic boundary
import pyscf.pbc
import pyscf.pbc.gto
import pyscf.pbc.scf
import pyscf.pbc.cc
import pyscf.pbc.tools
from pyscf.pbc.scf.addons import kconj_symmetry_

from vayesta.lattmod import latt
from vayesta.misc import molecules
from vayesta.misc import solids


class TestMolecule:
    """Molecular test system.
    """

    def __init__(self, atom, basis, auxbasis=None, verbose=0, **kwargs):
        super().__init__()
        mol = pyscf.gto.Mole()
        mol.atom = atom
        mol.basis = basis
        for key, val in kwargs.items():
            setattr(mol, key, val)
        mol.verbose = verbose
        mol.build()
        self.auxbasis = auxbasis
        self.mol = mol

    # --- Mean-field

    @cache
    def rhf(self):
        rhf = pyscf.scf.RHF(self.mol)
        if self.auxbasis is not None:
            rhf = rhf.density_fit(auxbasis=self.auxbasis)
        rhf.conv_tol = 1e-10
        rhf.conv_tol_grad = 1e-8
        rhf.kernel()
        assert rhf.converged
        return rhf

    @cache
    def uhf(self):
        uhf = pyscf.scf.UHF(self.mol)
        if self.auxbasis is not None:
            uhf = uhf.density_fit(auxbasis=self.auxbasis)
        uhf.conv_tol = 1e-10
        uhf.conv_tol_grad = 1e-8
        uhf.kernel()
        assert uhf.converged
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
        rccsd.conv_tol = 1e-10
        rccsd.conv_tol_normt = 1e-8
        rccsd.kernel()
        assert rccsd.converged
        rccsd.solve_lambda()
        assert rccsd.converged_lambda
        return rccsd

    @cache
    def uccsd(self):
        uccsd = pyscf.cc.UCCSD(self.uhf())
        uccsd.conv_tol = 1e-10
        uccsd.conv_tol_normt = 1e-8
        uccsd.kernel()
        assert uccsd.converged
        uccsd.solve_lambda()
        assert uccsd.converged_lambda
        return uccsd

    # --- FCI

    @cache
    def rfci(self):
        rfci = pyscf.fci.FCI(self.rhf())
        rfci.threads = 1
        rfci.kernel()
        return rfci

    @cache
    def ufci(self):
        ufci = pyscf.fci.FCI(self.uhf())
        ufci.threads = 1
        ufci.kernel()
        return ufci


class TestSolid:
    """Solid test system.
    """

    def __init__(
            self,
            a,
            atom,
            basis,
            kmesh=None,
            auxbasis=None,
            supercell=None,
            exxdiv='ewald',
            verbose=0,
            df='gdf',
            **kwargs,
    ):
        super().__init__()
        mol = pyscf.pbc.gto.Cell()
        mol.a = a
        mol.atom = atom
        mol.basis = basis
        for key, val in kwargs.items():
            setattr(mol, key, val)
        mol.verbose = verbose
        mol.build()
        if supercell is not None:
            mol = pyscf.pbc.tools.super_cell(mol, supercell)
        self.supercell = supercell
        self.mol = mol
        self.kpts = self.mol.make_kpts(kmesh) if kmesh is not None else None
        self.df = df
        self.auxbasis = auxbasis
        self.exxdiv = exxdiv

    # --- Mean-field

    @cache
    def rhf(self):
        if self.kpts is None:
            rhf = pyscf.pbc.scf.RHF(self.mol)
        else:
            rhf = pyscf.pbc.scf.KRHF(self.mol, self.kpts)
        if self.df == 'gdf':
            rhf = rhf.density_fit(auxbasis=self.auxbasis)
        elif self.df == 'rsgdf':
            rhf = rhf.rs_density_fit(auxbasis=self.auxbasis)
        if self.kpts is not None:
            rhf = kconj_symmetry_(rhf)
        rhf.conv_tol = 1e-10
        rhf.conv_tol_grad = 1e-6
        rhf.exxdiv = self.exxdiv
        rhf.kernel()
        assert rhf.converged
        return rhf

    @cache
    def uhf(self):
        if self.kpts is None:
            uhf = pyscf.pbc.scf.UHF(self.mol)
        else:
            uhf = pyscf.pbc.scf.KUHF(self.mol, self.kpts)
        if self.df == 'gdf':
            uhf = uhf.density_fit(auxbasis=self.auxbasis)
        elif self.df == 'rsgdf':
            uhf = uhf.rs_density_fit(auxbasis=self.auxbasis)
        if self.kpts is not None:
            uhf = kconj_symmetry_(uhf)
        uhf.conv_tol = 1e-10
        uhf.conv_tol_grad = 1e-8
        uhf.exxdiv = self.exxdiv
        uhf.kernel()
        assert uhf.converged
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
        assert rccsd.converged
        if self.kpts is None:
            rccsd.solve_lambda()
            assert rccsd.converged_lambda
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
        assert uccsd.converged
        if self.kpts is None:
            uccsd.solve_lambda()
            assert uccsd.converged_lambda
        return uccsd


class TestLattice:
    """Lattice test system.
    """

    def __init__(
            self,
            nsite,
            nelectron=None,
            spin=0,
            order=None,
            boundary="pbc",
            tiles=(1, 1),
            verbose=0,
            with_df=False,
            **kwargs,
    ):
        super().__init__()
        if isinstance(nsite, int):
            mol = latt.Hubbard1D(
                    nsite,
                    nelectron=nelectron,
                    spin=spin,
                    order=order,
                    boundary=boundary,
            )
        else:
            mol = latt.Hubbard2D(
                    nsite,
                    nelectron=nelectron,
                    spin=spin,
                    order=order,
                    tiles=tiles,
                    boundary=boundary,
            )
        mol.verbose = verbose
        for key, val in kwargs.items():
            setattr(mol, key, val)
        self.mol = mol
        self.with_df = with_df

    # --- Mean-field

    @cache
    def rhf(self):
        rhf = latt.LatticeRHF(self.mol)
        if self.with_df:
            rhf = rhf.density_fit()
        rhf.conv_tol = 1e-12
        rhf.kernel()
        return rhf

    @cache
    def uhf(self):
        uhf = latt.LatticeRHF(self.mol)
        if self.with_df:
            uhf = uhf.density_fit()
        uhf.conv_tol = 1e-12
        uhf.kernel()
        return uhf


# --- Test Systems

# Molecules

h2_dz = TestMolecule("H 0 0 0; H 0 0 0.74", basis="cc-pvdz")
h2anion_dz = TestMolecule("H 0 0 0; H 0 0 0.74", basis="cc-pvdz", charge=-1, spin=1)

h6_sto6g = TestMolecule(
        atom=["H %f %f %f" % xyz for xyz in pyscf.tools.ring.make(6, 1.0)],
        basis="sto6g",
)
h6_sto6g_df = TestMolecule(
        atom=["H %f %f %f" % xyz for xyz in pyscf.tools.ring.make(6, 1.0)],
        basis="sto6g",
        auxbasis="weigend",
)

water_sto3g = TestMolecule(atom=molecules.water(), basis='sto3g')
water_cation_sto3g = TestMolecule(atom=molecules.water(), basis='sto3g', charge=1, spin=1)

water_631g = TestMolecule(atom=molecules.water(), basis='6-31G')
water_631g_df = TestMolecule(atom=molecules.water(), basis='6-31G', auxbasis='6-31G')
water_cation_631g = TestMolecule(atom=molecules.water(), basis='6-31G', charge=1, spin=1)
water_cation_631g_df = TestMolecule(atom=molecules.water(), basis='6-31G', auxbasis='6-31G', charge=1, spin=1)

water_ccpvdz = TestMolecule(atom=molecules.water(), basis="cc-pvdz")
water_ccpvdz_df = TestMolecule(atom=molecules.water(), basis="cc-pvdz", auxbasis="cc-pvdz-jkfit")

ethanol_ccpvdz = TestMolecule(atom=molecules.ethanol(), basis="cc-pvdz")

lih_ccpvdz = TestMolecule(atom="Li 0 0 0; H 0 0 1.4", basis="cc-pvdz")

h2_ccpvdz = TestMolecule(atom="H1 0 0 0; H2 0 0 1.0", basis="cc-pvdz")
h2_ccpvdz_df = TestMolecule(atom="H1 0 0 0; H2 0 0 1.0", basis="cc-pvdz", auxbasis="cc-pvdz-jkfit")

h3_ccpvdz = TestMolecule(atom="H1 0 0 0; H2 0 0 1.0; H3 0 1.0 0", basis="cc-pvdz", spin=1)
h3_ccpvdz_df = TestMolecule(atom="H1 0 0 0; H2 0 0 1.0; H3 0 1.0 0", basis="cc-pvdz", auxbasis="cc-pvdz-jkfit", spin=1)

n2_ccpvdz_df = TestMolecule("N1 0 0 0; N2 0 0 1.1", basis="cc-pvdz", auxbasis="cc-pvdz-jkfit")


# Solids

a = 2*np.eye(3)
a[2,2] = 4
nk = 3
opts = dict(basis='sto-3g', auxbasis='sto-3g', exp_to_discard=0.1)

h2_sto3g_k311 = TestSolid(a=a, atom='H 0 0 0 ; H 0 0 0.74', kmesh=(nk,1,1), **opts)
h2_sto3g_s311 = TestSolid(a=a, atom='H 0 0 0 ; H 0 0 0.74', supercell=(nk,1,1), **opts)

h3_sto3g_k311 = TestSolid(a=a, atom='H 0 0 0 ; H 0 0 1 ; H 0 0 2', kmesh=(nk,1,1), spin=3, **opts)
h3_sto3g_s311 = TestSolid(a=a, atom='H 0 0 0 ; H 0 0 1 ; H 0 0 2', supercell=(nk,1,1), spin=3, **opts)

a = a.copy()
a[2,2] = 15.0
opts['dimension'] = 2
h2_sto3g_k31 = TestSolid(a=a, atom='H 0 0 0 ; H 0 0 0.74', kmesh=(nk,1,1), **opts)
h2_sto3g_s31 = TestSolid(a=a, atom='H 0 0 0 ; H 0 0 0.74', supercell=(nk,1,1), **opts)

h3_sto3g_k31 = TestSolid(a=a, atom='H 0 0 0 ; H 0 0 1 ; H 0 0 2', kmesh=(nk,1,1), spin=3, **opts)
h3_sto3g_s31 = TestSolid(a=a, atom='H 0 0 0 ; H 0 0 1 ; H 0 0 2', supercell=(nk,1,1), spin=3, **opts)

a, atom = solids.diamond()
opts = dict(basis='sto-3g', auxbasis='sto-3g', exp_to_discard=0.1)
mesh = (2,1,1)
diamond_sto3g_k211 = TestSolid(a=a, atom=atom, kmesh=mesh, **opts)
diamond_sto3g_s211 = TestSolid(a=a, atom=atom, supercell=mesh, **opts)

a, atom = solids.diamond()
opts = dict(basis='sto3g', auxbasis='sto3g', exp_to_discard=0.1)
mesh = (3,3,3)
diamond_sto3g_k333 = TestSolid(a=a, atom=atom, kmesh=mesh, **opts)
diamond_sto3g_s333 = TestSolid(a=a, atom=atom, supercell=mesh, **opts)

a = np.eye(3) * 3.0
a[2, 2] = 20.0
he_k32 = TestSolid(a, atom="He 0 0 0", dimension=2, basis="def2-svp", auxbasis="def2-svp-ri", kmesh=(3, 2, 1))
he_s32 = TestSolid(a, atom="He 0 0 0", dimension=2, basis="def2-svp", auxbasis="def2-svp-ri", supercell=(3, 2, 1))

a = np.eye(3) * 3.0
a[1, 1] = a[2, 2] = 30.0
he_k3 = TestSolid(a, atom="He 0 0 0", dimension=1, basis="def2-svp", auxbasis="def2-svp-ri", kmesh=(3, 1, 1))
he_s3 = TestSolid(a, atom="He 0 0 0", dimension=1, basis="def2-svp", auxbasis="def2-svp-ri", supercell=(3, 1, 1))

a = np.eye(3) * 5
opts = dict(basis="6-31g", mesh=[11, 11, 11])
he2_631g_k222 = TestSolid(a=a, atom="He 3 2 3; He 1 1 1", kmesh=(2, 2, 2), **opts)
he2_631g_s222 = TestSolid(a=a, atom="He 3 2 3; He 1 1 1", supercell=(2, 2, 2), **opts)

a, atom = solids.rocksalt(atoms=["Li", "H"])
lih_k221 = TestSolid(a=a, atom=atom, basis="def2-svp", auxbasis="def2-svp-ri", kmesh=(2, 1, 1), exp_to_discard=0.1)
lih_s221 = TestSolid(a=a, atom=atom, basis="def2-svp", auxbasis="def2-svp-ri", supercell=(2, 1, 1), exp_to_discard=0.1)

a = np.eye(3) * 3.0
he_k321 = TestSolid(a, atom="He 0 0 0", basis="def2-svp", auxbasis="def2-svp-ri", kmesh=(3, 2, 1))
he_s321 = TestSolid(a, atom="He 0 0 0", basis="def2-svp", auxbasis="def2-svp-ri", supercell=(3, 2, 1))


# Lattices

hubb_6_u0 = TestLattice(6, hubbard_u=0.0, nelectron=6)
hubb_10_u2 = TestLattice(10, hubbard_u=2.0, nelectron=10)
hubb_10_u4 = TestLattice(10, hubbard_u=4.0, nelectron=16, boundary='apbc')
hubb_14_u4 = TestLattice(14, hubbard_u=4.0, nelectron=14, boundary='pbc')
hubb_14_u4_df = TestLattice(14, hubbard_u=4.0, nelectron=14, boundary='pbc', with_df=True)

hubb_6x6_u0_1x1imp = TestLattice((6, 6), hubbard_u=0.0, nelectron=26, tiles=(1, 1), boundary='pbc')
hubb_6x6_u2_1x1imp = TestLattice((6, 6), hubbard_u=2.0, nelectron=26, tiles=(1, 1), boundary='pbc')
hubb_6x6_u6_1x1imp = TestLattice((6, 6), hubbard_u=6.0, nelectron=26, tiles=(1, 1), boundary='pbc')
hubb_8x8_u2_2x2imp = TestLattice((8, 8), hubbard_u=2.0, nelectron=50, tiles=(2, 2), boundary='pbc')
hubb_8x8_u2_2x1imp = TestLattice((8, 8), hubbard_u=2.0, nelectron=50, tiles=(2, 1), boundary='pbc')
