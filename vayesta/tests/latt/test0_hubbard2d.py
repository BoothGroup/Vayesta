import unittest
import numpy as np

from vayesta import lattmod
from vayesta.tests.common import TestCase


class Hubbard2DTests_6_0(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mol = lattmod.Hubbard2D((6, 6), hubbard_u=0.0, nelectron=6*6)
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
        self.assertAlmostEqual(np.max(np.abs(f0-f1)), 0.0, 8)


class Hubbard2DTests_10_0(Hubbard2DTests_6_0):
    @classmethod
    def setUpClass(cls):
        cls.mol = lattmod.Hubbard2D((10, 10), hubbard_u=0.0, nelectron=10*10)
        cls.mf = lattmod.LatticeMF(cls.mol)
        cls.mf.kernel()
        cls.known_values = {"e_tot": -163.45383275624567}


class Hubbard2DTests_6_4(Hubbard2DTests_6_0):
    @classmethod
    def setUpClass(cls):
        cls.mol = lattmod.Hubbard2D((6, 6), hubbard_u=4.0, nelectron=6*6)
        cls.mf = lattmod.LatticeMF(cls.mol)
        cls.mf.kernel()
        cls.known_values = {"e_tot": -23.712812921102035}


class Hubbard2DTests_6_8(Hubbard2DTests_6_0):
    @classmethod
    def setUpClass(cls):
        cls.mol = lattmod.Hubbard2D((6, 6), hubbard_u=8.0, nelectron=6*6)
        cls.mf = lattmod.LatticeMF(cls.mol)
        cls.mf.kernel()
        cls.known_values = {"e_tot": 12.28718707889798}


if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
