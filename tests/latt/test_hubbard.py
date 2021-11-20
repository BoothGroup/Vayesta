import unittest
import numpy as np
from vayesta import lattmod


class HubbardTests:

    @classmethod
    def setUpClass(cls):
        cls.mol = None
        cls.mf = None
        cls.known_values = {}

    @classmethod
    def tearDownClass(cls):
        del cls.mol, cls.mf, cls.known_values

    def test_energy(self):
        self.assertAlmostEqual(self.mf.e_tot, self.known_values['e_tot'])

    def test_fock(self):
        f0 = self.mf.get_fock()
        f0 = np.einsum('pq,pi,qj->ij', f0, self.mf.mo_coeff.conj(), self.mf.mo_coeff)
        f1 = np.diag(self.mf.mo_energy)
        self.assertAlmostEqual(np.max(np.abs(f0-f1)), 0.0, 8)


class Hubbard_1D_N10_U0_halffilling_Tests(unittest.TestCase, HubbardTests):
    @classmethod
    def setUpClass(cls):
        cls.mol = lattmod.Hubbard1D(10, hubbard_u=0.0, nelectron=10)
        cls.mf = lattmod.LatticeMF(cls.mol)
        cls.mf.kernel()
        cls.known_values = {'e_tot': -12.94427190999916}


class Hubbard_1D_N20_U0_halffilling_Tests(unittest.TestCase, HubbardTests):
    @classmethod
    def setUpClass(cls):
        cls.mol = lattmod.Hubbard1D(20, hubbard_u=0.0, nelectron=20)
        cls.mf = lattmod.LatticeMF(cls.mol)
        cls.mf.kernel()
        cls.known_values = {'e_tot': -25.569812885998644}


class Hubbard_1D_N10_U4_halffilling_Tests(unittest.TestCase, HubbardTests):
    @classmethod
    def setUpClass(cls):
        cls.mol = lattmod.Hubbard1D(10, hubbard_u=4.0, nelectron=10)
        cls.mf = lattmod.LatticeMF(cls.mol)
        cls.mf.kernel()
        cls.known_values = {'e_tot': -2.944271909999157}


class Hubbard_1D_N10_U4_halffilling_Tests(unittest.TestCase, HubbardTests):
    @classmethod
    def setUpClass(cls):
        cls.mol = lattmod.Hubbard1D(10, hubbard_u=8.0, nelectron=10)
        cls.mf = lattmod.LatticeMF(cls.mol)
        cls.mf.kernel()
        cls.known_values = {'e_tot': 7.0557280900008275}


class Hubbard_2D_N6_U0_halffilling_Tests(unittest.TestCase, HubbardTests):
    @classmethod
    def setUpClass(cls):
        cls.mol = lattmod.Hubbard2D((6, 6), hubbard_u=0.0, nelectron=36)
        cls.mf = lattmod.LatticeMF(cls.mol)
        cls.mf.kernel()
        cls.known_values = {'e_tot': -59.71281292110203}


class Hubbard_2D_N10_U0_halffilling_Tests(unittest.TestCase, HubbardTests):
    @classmethod
    def setUpClass(cls):
        cls.mol = lattmod.Hubbard2D((10, 10), hubbard_u=0.0, nelectron=100)
        cls.mf = lattmod.LatticeMF(cls.mol)
        cls.mf.kernel()
        cls.known_values = {'e_tot': -163.45383275624567}


class Hubbard_2D_N6_U4_halffilling_Tests(unittest.TestCase, HubbardTests):
    @classmethod
    def setUpClass(cls):
        cls.mol = lattmod.Hubbard2D((6, 6), hubbard_u=4.0, nelectron=36)
        cls.mf = lattmod.LatticeMF(cls.mol)
        cls.mf.kernel()
        cls.known_values = {'e_tot': -23.712812921102035}


class Hubbard_2D_N6_U8_halffilling_Tests(unittest.TestCase, HubbardTests):
    @classmethod
    def setUpClass(cls):
        cls.mol = lattmod.Hubbard2D((6, 6), hubbard_u=8.0, nelectron=36)
        cls.mf = lattmod.LatticeMF(cls.mol)
        cls.mf.kernel()
        cls.known_values = {'e_tot': 12.287187078897988}


if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
