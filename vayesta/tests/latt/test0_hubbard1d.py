import unittest
import numpy as np

from vayesta import lattmod
from vayesta.tests.common import TestCase


class Hubbard1DTests_10_0(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mol = lattmod.Hubbard1D(10, hubbard_u=0.0, nelectron=10)
        cls.mf = lattmod.LatticeMF(cls.mol)
        cls.mf.kernel()
        cls.known_values = {"e_tot": -12.94427190999916}

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


class Hubbard1DTests_20_0(Hubbard1DTests_10_0):
    @classmethod
    def setUpClass(cls):
        cls.mol = lattmod.Hubbard1D(20, hubbard_u=0.0, nelectron=20)
        cls.mf = lattmod.LatticeMF(cls.mol)
        cls.mf.kernel()
        cls.known_values = {"e_tot": -25.569812885998644}


class Hubbard1DTests_10_4(Hubbard1DTests_10_0):
    @classmethod
    def setUpClass(cls):
        cls.mol = lattmod.Hubbard1D(10, hubbard_u=4.0, nelectron=10)
        cls.mf = lattmod.LatticeMF(cls.mol)
        cls.mf.kernel()
        cls.known_values = {"e_tot": -2.944271909999157}


class Hubbard1DTests_10_8(Hubbard1DTests_10_0):
    @classmethod
    def setUpClass(cls):
        cls.mol = lattmod.Hubbard1D(10, hubbard_u=8.0, nelectron=10)
        cls.mf = lattmod.LatticeMF(cls.mol)
        cls.mf.kernel()
        cls.known_values = {"e_tot": 7.0557280900008275}


if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
