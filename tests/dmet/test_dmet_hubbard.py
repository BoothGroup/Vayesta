import unittest
from vayesta import lattmod, dmet


def make_test_hub1d(n, nelectron, U, nimp, known_values):

    class Hubbard1DTests(unittest.TestCase):

        @classmethod
        def setUpClass(cls):
            cls.mol = lattmod.Hubbard1D(n, hubbard_u=U, nelectron=nelectron)
            cls.mf = lattmod.LatticeMF(cls.mol)
            cls.mf.kernel()
            cls.dmet = dmet.DMET(cls.mf, solver='FCI', fragment_type='Site', charge_consistent = False, maxiter=20)
            # Ensure that we don't spam with output.
            cls.dmet.log.setLevel(50)
            f = cls.dmet.make_atom_fragment(list(range(nimp)))
            symfrags = f.make_tsymmetric_fragments(tvecs=[n//nimp, 1, 1])
            cls.dmet.kernel()

        @classmethod
        def tearDownClass(cls):
            del cls.mol, cls.mf, cls.dmet

        def test_energy(self):
            self.assertAlmostEqual(self.dmet.e_tot, known_values['e_tot'])

    return Hubbard1DTests

def make_test_hub2d(n, nelectron, U, impshape, boundary, known_values):

    class Hubbard2DTests(unittest.TestCase):

        @classmethod
        def setUpClass(cls):
            cls.mol = lattmod.Hubbard2D(n, hubbard_u=U, nelectron=nelectron, tiles = impshape, boundary=boundary)
            cls.mf = lattmod.LatticeMF(cls.mol)
            cls.mf.kernel()
            cls.dmet = dmet.DMET(cls.mf, solver='FCI', fragment_type='Site', charge_consistent = False, maxiter=20)
            # Ensure that we don't spam with output.
            cls.dmet.log.setLevel(50)
            f = cls.dmet.make_atom_fragment(list(range(impshape[0] * impshape[1])))
            symfrags = f.make_tsymmetric_fragments(tvecs=[n[0] // impshape[0], n[1] // impshape[1], 1])
            cls.dmet.kernel()

        @classmethod
        def tearDownClass(cls):
            del cls.mol, cls.mf, cls.dmet

        def test_energy(self):
            self.assertAlmostEqual(self.dmet.e_tot, known_values['e_tot'])

    return Hubbard2DTests


dmet_N10_U0_1imp_halffilling_Test = make_test_hub1d(10, 10, 0.0, 1, {'e_tot': -12.944271909999166})
dmet_N10_U0_2imp_halffilling_Test = make_test_hub1d(10, 10, 0.0, 2, {'e_tot': -12.944271909999154})
dmet_N20_U0_2imp_halffilling_Test = make_test_hub1d(20, 20, 0.0, 2, {'e_tot': -25.56981288599864})
dmet_N10_U4_1imp_halffilling_Test = make_test_hub1d(10, 10, 4.0, 1, {'e_tot': -6.109901024380408})
dmet_N10_U4_2imp_halffilling_Test = make_test_hub1d(10, 10, 4.0, 2, {'e_tot': -5.982495317229596})
dmet_N10_U4_5imp_halffilling_Test = make_test_hub1d(10, 10, 4.0, 5, {'e_tot': -5.834322635766257})
dmet_N12_U8_2imp_halffilling_Test = make_test_hub1d(12, 12, 8.0, 2, {'e_tot': -4.022943661738756})


dmet_N6x6_U1_2x2imp_halffilling_Test = make_test_hub2d((6,6), 26, 1.0, (2,2), "PBC", {'e_tot': -51.744964087157626})
dmet_N6x6_U6_2x2imp_halffilling_Test = make_test_hub2d((6,6), 26, 6.0, (2,2), "PBC", {'e_tot': -39.51961645103434})
dmet_N8x8_U4_1x1imp_halffilling_Test = make_test_hub2d((8,8), 50, 4.0, (1,1), "PBC", {'e_tot': -74.93240779775238})
dmet_N8x8_U3_2x2imp_halffilling_Test = make_test_hub2d((8,8), 50, 3.0, (2,2), "PBC", {'e_tot': -79.06298997526869})


if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
