import unittest
import numpy as np
from vayesta import lattmod


def make_test(n, nelectron, U, known_values):

    class Hubbard1DTests(unittest.TestCase):

        @classmethod
        def setUpClass(cls):
            cls.mol = lattmod.Hubbard1D(n, hubbard_u=U, nelectron=nelectron)
            cls.mf = lattmod.LatticeMF(cls.mol)
            cls.mf.kernel()

        @classmethod
        def tearDownClass(cls):
            del cls.mol, cls.mf

        def test_energy(self):
            self.assertAlmostEqual(self.mf.e_tot, known_values['e_tot'])

        def test_fock(self):
            f0 = self.mf.get_fock()
            f0 = np.einsum('pq,pi,qj->ij', f0, self.mf.mo_coeff.conj(), self.mf.mo_coeff)
            f1 = np.diag(self.mf.mo_energy)
            self.assertAlmostEqual(np.max(np.abs(f0-f1)), 0.0, 8)

    return Hubbard1DTests


N10_U0_halffilling_Test = make_test(10, 10, 0.0, {'e_tot': -12.94427190999916})
N20_U0_halffilling_Test = make_test(20, 20, 0.0, {'e_tot': -25.569812885998644})
N10_U4_halffilling_Test = make_test(10, 10, 4.0, {'e_tot': -2.944271909999157})
N10_U8_halffilling_Test = make_test(10, 10, 8.0, {'e_tot': 7.0557280900008275})



if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
