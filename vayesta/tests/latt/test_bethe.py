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
        self.assertAlmostEqual(e, self.known_values['energy'], 8)

    def test_double_occ(self):
        d = bethe.hubbard1d_bethe_docc(self.t, self.u)
        self.assertAlmostEqual(d, self.known_values['docc'], 8)

    def test_double_occ_numdiff(self):
        d1 = bethe.hubbard1d_bethe_docc_numdiff(self.t, self.u, du=1e-12, order=1)
        d2 = bethe.hubbard1d_bethe_docc_numdiff(self.t, self.u, du=1e-12, order=2)
        self.assertAlmostEqual(d1, self.known_values['docc-numdiff1'], 8)
        self.assertAlmostEqual(d2, self.known_values['docc-numdiff2'], 8)


class Bethe_T1_U0_Tests(unittest.TestCase, BetheTests):
    @classmethod
    def setUpClass(cls):
        cls.t = 1.0
        cls.u = 0.0
        cls.known_values = {
                'energy': -1.2732565954632262,
                'docc': 0.2499999972419595,
                'docc-numdiff1': 0.25002222514558525,
                'docc-numdiff2': 0.2500916140846243,
        }


class Bethe_T2_U4_Tests(unittest.TestCase, BetheTests):
    @classmethod
    def setUpClass(cls):
        cls.t = 2.0
        cls.u = 4.0
        cls.known_values = {
                'energy': -1.688748682251278,
                'docc': 0.17545254464835117,
                'docc-numdiff1': 0.17530421558831222,
                'docc-numdiff2': 0.17510992655900282,
        }


if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
