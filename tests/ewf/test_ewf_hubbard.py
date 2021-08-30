import unittest
import numpy as np
from vayesta import lattmod, ewf


def make_hubbard1d_test(n, nelectron, U, nimp, kwargs, known_values, name=None):

    class Hubbard1DTests(unittest.TestCase):

        shortDescription = lambda self: name

        @classmethod
        def setUpClass(cls):
            cls.mol = lattmod.Hubbard1D(n, hubbard_u=U, nelectron=nelectron)
            cls.mf = lattmod.LatticeMF(cls.mol)
            cls.mf.kernel()
            cls.ewf = ewf.EWF(cls.mf, fragment_type='site', solver_options={'conv_tol': 1e-10}, **kwargs)
            f = cls.ewf.make_atom_fragment(list(range(nimp)))
            f.make_tsymmetric_fragments(tvecs=[n//nimp, 1, 1])
            cls.ewf.kernel()

        @classmethod
        def tearDownClass(cls):
            del cls.mol, cls.mf, cls.ewf

        def test_energy(self):
            self.assertAlmostEqual(self.ewf.e_tot, known_values['e_tot'], 8)

    return Hubbard1DTests


def make_hubbard2d_test(n, nelectron, U, impshape, boundary, kwargs, known_values, name=None):

    class Hubbard2DTests(unittest.TestCase):

        shortDescription = lambda self: name

        @classmethod
        def setUpClass(cls):
            cls.mol = lattmod.Hubbard2D(n, hubbard_u=U, nelectron=nelectron, tiles=impshape, boundary=boundary)
            cls.mf = lattmod.LatticeMF(cls.mol)
            cls.mf.kernel()
            cls.ewf = ewf.EWF(cls.mf, fragment_type='site', solver_options={'conv_tol': 1e-10}, **kwargs)
            f = cls.ewf.make_atom_fragment(list(range(impshape[0] * impshape[1])))
            f.make_tsymmetric_fragments(tvecs=[n[0] // impshape[0], n[1] // impshape[1], 1])
            cls.ewf.kernel()

        @classmethod
        def tearDownClass(cls):
            del cls.mol, cls.mf, cls.ewf

        def test_energy(self):
            self.assertAlmostEqual(self.ewf.e_tot, known_values['e_tot'], 7)

    return Hubbard2DTests


N10_U0_1imp_halffilling_Test = make_hubbard1d_test(
        10, 10, 0.0, 1,
        {'bno_threshold': 1e-8},
        {'e_tot': -12.94427190999916},
        name='N10_U0_1imp_halffilling_Test',
)

N10_U0_2imp_halffilling_Test = make_hubbard1d_test(
        10, 10, 0.0, 2,
        {'bno_threshold': 1e-8},
        {'e_tot': -12.94427190999916},
        name='N10_U0_2imp_halffilling_Test',
)

N10_U4_1imp_halffilling_Test = make_hubbard1d_test(
        10, 10, 4.0, 1,
        {'bno_threshold': 1e-8},
        {'e_tot': -6.133885588519993},
        name='N10_U4_1imp_halffilling_Test',
)

N6_U6_2imp_halffilling_sc_Test = make_hubbard1d_test(
        6, 6, 6.0, 2,
        {'bno_threshold': 1e-6, 'sc_mode': 1, 'sc_energy_tol': 1e-9},
        {'e_tot': -3.1985807202795167},
        name='N6_U6_2imp_halffilling_sc_Test',
)

#N6x6_U1_2x2imp_halffilling_Test = make_hubbard2d_test(
#        (6, 6), 36, 1.0, (2, 2), "PBC",
#        {'bno_threshold': 1e-6},
#        {'e_tot': -46.639376232193776},
#        name='N6x6_U1_2x2imp_halffilling_Test',
#)



if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
