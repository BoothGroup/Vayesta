import pytest
import unittest
import numpy as np
from pyscf.lib.parameters import BOHR

from vayesta import lattmod
from vayesta.tests.common import TestCase


@pytest.mark.fast
class Hubbard2DTests_6_0(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mol = lattmod.Hubbard2D((6, 6), hubbard_u=0.0, nelectron=6 * 6)
        cls.mf = lattmod.LatticeMF(cls.mol)
        cls.mf.kernel()
        cls.known_values = {"e_tot": -59.7128129211020}

    @classmethod
    def tearDownClass(cls):
        del cls.mol, cls.mf

    def test_energy(self):
        self.assertAlmostEqual(self.mf.e_tot, self.known_values["e_tot"])

    def test_fock(self):
        f0 = self.mf.get_fock()
        f0 = np.einsum("pq,pi,qj->ij", f0, self.mf.mo_coeff.conj(), self.mf.mo_coeff)
        f1 = np.diag(self.mf.mo_energy)
        self.assertAlmostEqual(np.max(np.abs(f0 - f1)), 0.0, 8)


@pytest.mark.fast
class Hubbard2DTests_10_0(Hubbard2DTests_6_0):
    @classmethod
    def setUpClass(cls):
        cls.mol = lattmod.Hubbard2D((10, 10), hubbard_u=0.0, nelectron=10 * 10)
        cls.mf = lattmod.LatticeMF(cls.mol)
        cls.mf.kernel()
        cls.known_values = {"e_tot": -163.45383275624567}


@pytest.mark.fast
class Hubbard2DTests_6_4(Hubbard2DTests_6_0):
    @classmethod
    def setUpClass(cls):
        cls.mol = lattmod.Hubbard2D((6, 6), hubbard_u=4.0, nelectron=6 * 6)
        cls.mf = lattmod.LatticeMF(cls.mol)
        cls.mf.kernel()
        cls.known_values = {"e_tot": -23.712812921102035}


@pytest.mark.fast
class Hubbard2DTests_6_8(Hubbard2DTests_6_0):
    @classmethod
    def setUpClass(cls):
        cls.mol = lattmod.Hubbard2D((6, 6), hubbard_u=8.0, nelectron=6 * 6)
        cls.mf = lattmod.LatticeMF(cls.mol)
        cls.mf.kernel()
        cls.known_values = {"e_tot": 12.28718707889798}


@pytest.mark.fast
class Hubbard2DOrderingTests(TestCase):
    """The site ordering of the hopping matrix has to agree with the geometry, also for rectangular lattices."""

    @staticmethod
    def get_h1e_ref(mol):
        """Hopping matrix from the nearest-neighbour distances between the sites and their periodic images."""
        nx, ny = mol.nsites
        bx, by = mol.bfac
        # Coordinates in units of the lattice spacing:
        coords = mol.atom_coords() * BOHR
        h1e = np.zeros((mol.nsite, mol.nsite))
        for sx in (-1, 0, 1):
            for sy in (-1, 0, 1):
                shift = np.asarray([sx * nx, sy * ny, 0])
                dist = np.linalg.norm(coords[:, None] - coords[None, :] + shift, axis=-1)
                h1e -= mol.hubbard_t * bx ** abs(sx) * by ** abs(sy) * (abs(dist - 1) < 1e-10)
        return h1e

    @staticmethod
    def get_mo_energy_ref(mol):
        """Tight-binding band energies of the nx x ny lattice."""
        nx, ny = mol.nsites
        shifts = {1: 0.0, -1: 0.5}
        kx = 2 * np.pi * (np.arange(nx) + shifts[mol.bfac[0]]) / nx
        ky = 2 * np.pi * (np.arange(ny) + shifts[mol.bfac[1]]) / ny
        return np.sort((-2 * mol.hubbard_t * (np.cos(kx)[:, None] + np.cos(ky)[None, :])).ravel())

    def _test_h1e(self, nsites, boundary, **kwargs):
        mol = lattmod.Hubbard2D(nsites, hubbard_u=4.0, nelectron=nsites[0] * nsites[1], boundary=boundary, **kwargs)
        self.assertAllclose(mol.h1e, self.get_h1e_ref(mol))
        if "open" not in boundary:
            self.assertAllclose(np.linalg.eigvalsh(mol.h1e), self.get_mo_energy_ref(mol))

    def test_h1e_square(self):
        self._test_h1e((4, 4), ("pbc", "pbc"))
        self._test_h1e((4, 4), ("pbc", "apbc"))
        self._test_h1e((6, 6), ("apbc", "pbc"))

    def test_h1e_rectangular(self):
        self._test_h1e((2, 4), ("pbc", "pbc"))
        self._test_h1e((4, 2), ("pbc", "pbc"))
        self._test_h1e((3, 5), ("pbc", "apbc"))
        self._test_h1e((6, 4), ("apbc", "pbc"))
        self._test_h1e((4, 6), ("apbc", "apbc"))
        self._test_h1e((4, 6), ("open", "pbc"))

    def test_h1e_rectangular_tiles(self):
        self._test_h1e((4, 6), ("pbc", "pbc"), tiles=(2, 3))
        self._test_h1e((6, 4), ("pbc", "apbc"), tiles=(3, 2))
        self._test_h1e((6, 4), ("pbc", "pbc"), tiles=(2, 1))

    def _test_tiles_contiguous(self, nsites, tiles):
        mol = lattmod.Hubbard2D(nsites, hubbard_u=4.0, nelectron=nsites[0] * nsites[1], boundary="pbc", tiles=tiles)
        coords = mol.atom_coords() * BOHR
        tsize = tiles[0] * tiles[1]
        self.assertEqual(sorted(mol.order), list(range(mol.nsite)))
        for tile in range(mol.nsite // tsize):
            tile_coords = coords[tile * tsize : (tile + 1) * tsize]
            # Each tile is a contiguous block of tiles[0] x tiles[1] sites:
            self.assertAllclose(tile_coords.max(axis=0) - tile_coords.min(axis=0), [tiles[0] - 1, tiles[1] - 1, 0])
            self.assertEqual(len({tuple(c) for c in np.round(tile_coords, 8)}), tsize)

    def test_tiles_contiguous(self):
        self._test_tiles_contiguous((4, 6), (2, 3))
        self._test_tiles_contiguous((6, 4), (3, 2))
        self._test_tiles_contiguous((6, 4), (2, 1))
        self._test_tiles_contiguous((4, 4), (2, 2))


if __name__ == "__main__":
    print("Running %s" % __file__)
    unittest.main()
