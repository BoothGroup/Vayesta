import unittest
import numpy as np
from vayesta import lattmod


def make_test(n, nelectron, U, known_values):

    class Hubbard2DTests(unittest.TestCase):

        @classmethod
        def setUpClass(cls):
            cls.mol = lattmod.Hubbard2D(n, hubbard_u=U, nelectron=nelectron)
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

    return Hubbard2DTests


N10_U0_halffilling_Test = make_test((6, 6),   6*6,   0.0, {'e_tot': -59.71281292110203})
N20_U0_halffilling_Test = make_test((10, 10), 10*10, 0.0, {'e_tot': -163.45383275624567})
N10_U4_halffilling_Test = make_test((6, 6),   6*6,   4.0, {'e_tot': -23.712812921102035})
N10_U8_halffilling_Test = make_test((6, 6),   6*6,   8.0, {'e_tot': 12.287187078897988})



if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()

