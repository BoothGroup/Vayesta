import unittest
import numpy as np
from vayesta.lattmod import bethe


class BetheTests:

    @classmethod
    def setUpClass(cls):
        cls.t = None
        cls.u = None
        cls.known_values = {}

    @classmethod
    def tearDownClass(cls):
        del cls.t, cls.u, cls.known_values

    def test_energy(self):
        e = bethe.hubbard1d_bethe_energy(self.t, self.u)
        self.assertAlmostEqual(e, self.known_values['energy'])

    def test_double_occ(self):
        d = bethe.hubbard1d_bethe_docc(self.t, self.u)
        self.assertAlmostEqual(d, self.known_values['docc'])


class Bethe_T1_U0_Tests(unittest.TestCase, BetheTests):
    @classmethod
    def setUpClass(cls):
        cls.t = 1.0
        cls.u = 0.0
        cls.known_values = {
                'energy': -1.2732565954632262,
                'docc': 0.2499999972419595,
        }


class Bethe_T1_U0_Tests(unittest.TestCase, BetheTests):
    @classmethod
    def setUpClass(cls):
        cls.t = 2.0
        cls.u = 4.0
        cls.known_values = {
                'energy': -1.688748682251278,
                'docc': 0.17545254464835117,
        }


if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
