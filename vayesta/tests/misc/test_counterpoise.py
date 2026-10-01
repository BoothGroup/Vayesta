import pytest
import numpy as np
import pyscf.lib

from vayesta.misc.counterpoise import make_cp_mol
from vayesta.tests import testsystems
from vayesta.tests.common import TestCase


def count_ghosts(mol):
    return sum(mol.atom_symbol(atm).lower().startswith("ghost") for atm in range(mol.natm))


@pytest.mark.fast
class TestCounterpoiseMolecule(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mol = testsystems.water_sto3g.mol

    def test_atom_only(self):
        mol_cp = make_cp_mol(self.mol, 0, rmax=0)
        self.assertEqual(mol_cp.natm, 1)
        self.assertEqual(mol_cp.atom_symbol(0), "O")
        self.assertEqual(mol_cp.nelectron, 8)
        self.assertAllclose(mol_cp.atom_coord(0), self.mol.atom_coord(0))

    def test_ghost_atoms(self):
        mol_cp = make_cp_mol(self.mol, 0, rmax=10.0)
        self.assertEqual(mol_cp.natm, self.mol.natm)
        self.assertEqual(count_ghosts(mol_cp), self.mol.natm - 1)
        self.assertEqual(mol_cp.nelectron, 8)
        # Basis functions of all atoms are present:
        self.assertEqual(mol_cp.nao, self.mol.nao)

    def test_rmax(self):
        d_oh = np.linalg.norm(self.mol.atom_coord(0) - self.mol.atom_coord(1))
        d_hh = np.linalg.norm(self.mol.atom_coord(1) - self.mol.atom_coord(2))
        self.assertLess(d_oh, d_hh)
        rmax = (d_oh + d_hh) / 2
        # Hydrogen atom: only the oxygen is within rmax
        mol_cp = make_cp_mol(self.mol, 1, rmax=rmax, unit="B", spin=1)
        self.assertEqual(mol_cp.natm, 2)
        self.assertEqual(mol_cp.atom_symbol(1).lower(), "ghost-o")
        # Same with rmax in Angstrom
        mol_cp = make_cp_mol(self.mol, 1, rmax=rmax * pyscf.lib.param.BOHR, unit="A", spin=1)
        self.assertEqual(mol_cp.natm, 2)

    def test_invalid_unit(self):
        with self.assertRaises(ValueError):
            make_cp_mol(self.mol, 0, rmax=1.0, unit="X")


@pytest.mark.fast
class TestCounterpoiseSolid(TestCase):
    def test_images(self):
        # 2D lattice of H2 molecules (along z) with lattice constant of 2 Angstrom in x and y
        cell = testsystems.h2_sto3g_k31.mol
        self.assertEqual(cell.dimension, 2)
        mol_cp = make_cp_mol(cell, 0, rmax=1.01, nimages=1, unit="a0", spin=1)
        # Open boundary conditions
        self.assertEqual(mol_cp.dimension, 0)
        self.assertEqual(mol_cp.nelectron, 1)
        # Ghost atoms: H2 partner in the home cell, and the H atoms in the four neighboring cells in x and y
        self.assertEqual(mol_cp.natm, 6)
        self.assertEqual(count_ghosts(mol_cp), 5)
        coords = np.asarray([mol_cp.atom_coord(atm, unit="ANG") for atm in range(mol_cp.natm)])
        dist = np.sort(np.linalg.norm(coords - coords[0], axis=1))
        self.assertAllclose(dist, [0, 0.74, 2, 2, 2, 2])
