import unittest
from vayesta import lattmod, edmet


def make_test_hub1d(n, nelectron, U, nimp, max_boson_occ, known_values):

    class Hubbard1DTests(unittest.TestCase):

        @classmethod
        def setUpClass(cls):
            cls.mol = lattmod.Hubbard1D(n, hubbard_u=U, nelectron=nelectron)
            cls.mf = lattmod.LatticeMF(cls.mol)
            cls.mf.kernel()
            cls.edmet = edmet.EDMET(cls.mf, solver='EBFCI', fragment_type='Site', bos_occ_cutoff=max_boson_occ)
            # Ensure that we don't spam with output.
            cls.edmet.log.setLevel(50)
            f = cls.edmet.make_atom_fragment(list(range(nimp)))
            symfrags = f.make_tsymmetric_fragments(tvecs=[n//nimp, 1, 1])
            cls.edmet.kernel()

        @classmethod
        def tearDownClass(cls):
            del cls.mol, cls.mf, cls.edmet

        def test_energy(self):
            self.assertAlmostEqual(self.edmet.e_tot, known_values['e_tot'])

    return Hubbard1DTests

def make_test_hub2d(n, nelectron, U, impshape, boundary, max_boson_occ, known_values):

    class Hubbard2DTests(unittest.TestCase):

        @classmethod
        def setUpClass(cls):
            cls.mol = lattmod.Hubbard2D(n, hubbard_u=U, nelectron=nelectron, tiles = impshape, boundary=boundary)
            cls.mf = lattmod.LatticeMF(cls.mol)
            cls.mf.kernel()
            cls.edmet = edmet.EDMET(cls.mf, solver='EBFCI', fragment_type='Site', bos_occ_cutoff=max_boson_occ)
            # Ensure that we don't spam with output.
            cls.edmet.log.setLevel(50)
            f = cls.edmet.make_atom_fragment(list(range(impshape[0] * impshape[1])))
            symfrags = f.make_tsymmetric_fragments(tvecs=[n[0] // impshape[0], n[1] // impshape[1], 1])
            cls.edmet.kernel()

        @classmethod
        def tearDownClass(cls):
            del cls.mol, cls.mf, cls.edmet

        def test_energy(self):
            self.assertAlmostEqual(self.edmet.e_tot, known_values['e_tot'], places = 6)

    return Hubbard2DTests


dmet_N10_U4_1imp_1occ_halffilling_Test = make_test_hub1d(10, 10, 4.0, 1, 1, {'e_tot': -6.118248173618246})
dmet_N10_U4_1imp_10occ_halffilling_Test = make_test_hub1d(10, 10, 4.0, 1, 10, {'e_tot': -6.1181667874754035})
dmet_N10_U4_2imp_2occ_halffilling_Test = make_test_hub1d(10, 10, 0.5, 2, 2, {'e_tot': -11.74139024719132})

dmet_N6x6_U1_2x2imp_halffilling_Test = make_test_hub2d((6,6), 26, 4.0, (2,1), "PBC", 2, {'e_tot': -43.387417397200636})
dmet_N6x6_U6_2x2imp_halffilling_Test = make_test_hub2d((6,6), 26, 6.0, (1,1), "PBC", 5, {'e_tot': -41.295421512183374})
dmet_N8x8_U4_1x1imp_halffilling_Test = make_test_hub2d((8,8), 50, 2.0, (1,2), "PBC", 5, {'e_tot': -84.96249789942502})

if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
