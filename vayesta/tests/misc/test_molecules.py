import pytest
import numpy as np
import pyscf.gto

from vayesta.misc import molecules
from vayesta.tests.common import TestCase


def coords(atom):
    return np.asarray([a[1] for a in atom], dtype=float)


def symbols(atom):
    return [a[0] for a in atom]


def distance(atom, i, j):
    return np.linalg.norm(coords(atom)[i] - coords(atom)[j])


def min_distances(atom, sym1, sym2):
    """For each atom of type sym1, the distance to the closest atom of type sym2."""
    r = coords(atom)
    idx1 = [i for i, s in enumerate(symbols(atom)) if s.startswith(sym1)]
    idx2 = [i for i, s in enumerate(symbols(atom)) if s.startswith(sym2)]
    out = []
    for i in idx1:
        d = [np.linalg.norm(r[i] - r[j]) for j in idx2 if j != i]
        out.append(min(d))
    return np.asarray(out)


@pytest.mark.fast
class TestMolecules(TestCase):
    def test_alkane(self):
        for n in range(1, 5):
            atom = molecules.alkane(n)
            self.assertEqual(len(atom), 3 * n + 2)
            self.assertEqual(symbols(atom).count("C"), n)
            self.assertAllclose(min_distances(atom, "H", "C"), 1.09)
            if n > 1:
                self.assertAllclose(min_distances(atom, "C", "C"), 1.54)

    def test_alkane_numbering(self):
        atom = molecules.alkane(2, numbering="atom")
        self.assertEqual(symbols(atom), ["C0", "H1", "H2", "H3", "C4", "H5", "H6", "H7"])
        atom = molecules.alkane(2, numbering="unit")
        self.assertEqual(symbols(atom), ["C0", "H0", "H0", "H0", "C1", "H1", "H1", "H1"])
        with self.assertRaises(AssertionError):
            molecules.alkane(2, numbering="invalid")

    def test_alkene(self):
        for n in (2, 4, 6):
            atom = molecules.alkene(n)
            self.assertEqual(len(atom), 2 * n + 2)
            self.assertAllclose(min_distances(atom, "C", "C"), 1.33)
            self.assertAllclose(min_distances(atom, "H", "C"), 1.09)
        with self.assertRaises(ValueError):
            molecules.alkene(1)
        with self.assertRaises(NotImplementedError):
            molecules.alkene(3)

    def test_arene(self):
        atom = molecules.arene(6)
        self.assertEqual(len(atom), 12)
        self.assertAllclose(min_distances(atom, "C", "C"), 1.39)
        self.assertAllclose(min_distances(atom, "H", "C"), 1.09)

    def test_ring_and_chain(self):
        atom = molecules.ring("H", 6, bond_length=1.2)
        self.assertEqual(len(atom), 6)
        self.assertAllclose(min_distances(atom, "H", "H"), 1.2)
        atom = molecules.ring(["H", "Li"], 4, radius=2.0, z=1.0, numbering=1)
        self.assertEqual(symbols(atom), ["H1", "Li2", "H3", "Li4"])
        self.assertAllclose(np.linalg.norm(coords(atom)[:, :2], axis=1), 2.0)
        self.assertAllclose(coords(atom)[:, 2], 1.0)
        atom = molecules.chain("H", 4, 1.5)
        self.assertAllclose(coords(atom)[:, 0], [0, 1.5, 3.0, 4.5])

    def test_ethanol(self):
        atom = molecules.ethanol(oh_bond=1.2)
        self.assertAlmostEqual(distance(atom, 2, 3), 1.2)
        # Scaling
        atom0 = molecules.ethanol()
        atom2 = molecules.ethanol(scale=2)
        self.assertAllclose(coords(atom2), 2 * coords(atom0))

    def test_ketene(self):
        atom0 = molecules.ketene()
        atom = molecules.ketene(cc_bond=1.5)
        self.assertAlmostEqual(distance(atom, 0, 1), 1.5)
        # C=O bond is unchanged
        self.assertAlmostEqual(distance(atom, 1, 2), distance(atom0, 1, 2))

    def test_ferrocene(self):
        atom = molecules.ferrocene()
        self.assertEqual(len(atom), 21)
        self.assertEqual(symbols(atom).count("C"), 10)
        # Distance of iron to the centers of the cyclopentadienyl rings
        r = coords(atom)
        for ring in (r[1:6], r[11:16]):
            self.assertAlmostEqual(np.linalg.norm(ring.mean(axis=0) - r[0]), 1.648)
        self.assertAllclose(min_distances(atom, "H", "C"), 1.079)
        atom = molecules.ferrocene(numbering=True)
        self.assertEqual(symbols(atom)[:3], ["Fe1", "C2", "C3"])
        self.assertEqual(symbols(atom)[-1], "H21")
        with self.assertRaises(NotImplementedError):
            molecules.ferrocene(conformation="staggered")

    def test_datafiles(self):
        """All molecules read from data files can be built by PySCF."""
        for name in (
            "water",
            "no2",
            "acetic_acid",
            "ferrocene_b3lyp",
            "propyl",
            "phenyl",
            "propanol",
            "chloroethanol",
            "neopentane",
            "boronene",
            "coronene",
            "glycine",
        ):
            atom = getattr(molecules, name)()
            mol = pyscf.gto.M(atom=atom, basis="sto-3g", spin=None, verbose=0)
            self.assertEqual(mol.natm, len(atom))
            # No overlapping atoms
            self.assertGreater(min_distances(atom, "", "").min(), 0.5)
