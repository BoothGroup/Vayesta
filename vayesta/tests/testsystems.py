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


class TestMolecule:

    def __init__(self, atom, basis, charge=0, spin=0):
        super().__init__()
        mol = pyscf.gto.Mole()
        mol.atom = atom
        mol.basis = basis
        mol.charge = charge
        mol.spin = spin
        mol.build()
        self.mol = mol

    # --- Mean-field

    @cache
    def rhf(self):
        rhf = pyscf.scf.RHF(self.mol)
        rhf.kernel()
        return rhf

    @cache
    def uhf(self):
        uhf = pyscf.scf.UHF(self.mol)
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

    def __init__(self, a, atom, basis, charge=0, spin=0, kmesh=None, auxbasis=None):
        super().__init__()
        mol = pyscf.pbc.gto.Cell()
        mol.a = a
        mol.atom = atom
        mol.basis = basis
        mol.charge = charge
        mol.spin = spin
        mol.build()
        self.mol = mol
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
        rhf.kernel()
        return rhf

    @cache
    def uhf(self):
        if self.kpts is None:
            uhf = pyscf.pbc.scf.UHF(self.mol)
        else:
            uhf = pyscf.pbc.scf.KUHF(self.mol, self.kpts)
        uhf = rhf.density_fit(auxbasis=self.auxbasis)
        uhf.kernel()
        return uhf

    # --- CCSD

    @cache
    def rccsd(self):
        if self.kpts is None:
            rccsd = pyscf.pbc.cc.RCCSD(self.rhf())
        else:
            rccsd = pyscf.pbc.cc.KRCCSD(self.rhf())
        rccsd.kernel()
        rccsd.solve_lambda()
        return rccsd

    @cache
    def uccsd(self):
        if self.kpts is None:
            uccsd = pyscf.pbc.cc.UCCSD(self.uhf())
        else:
            uccsd = pyscf.pbc.cc.KUCCSD(self.uhf())
        uccsd.kernel()
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

lih_dz = TestMolecule(
    atom = "Li 0 0 0 ; H 0 0 1.595",
    basis = 'cc-pVDZ')

h2o_dz = TestMolecule(
    atom = """
        O  0.0000   0.0000   0.1173
        H  0.0000   0.7572  -0.4692
        H  0.0000  -0.7572  -0.4692
        """,
    basis = 'cc-pVDZ')

# Solids

he_431g_gamma = TestSolid(
        a=3*np.eye(3), atom='He 0 0 0', basis='431g')

he_431g_k222 = TestSolid(
        a=3*np.eye(3), atom='He 0 0 0', basis='431g', kmesh=(2,2,2))
