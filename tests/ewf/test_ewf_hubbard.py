import unittest
import numpy as np
from vayesta import lattmod, ewf

# Use default conv_tol
EWF_CONV_TOL = None

class HubbardEWFTest:
    ''' Abstract base class for Hubbard model EWF tests.
    '''

    @classmethod
    def setUpClass(cls):
        cls.mol = None
        cls.mf = None
        cls.ewf = None
        cls.known_values = None

    @classmethod
    def teardownClass(cls):
        del cls.mol, cls.mf, cls.ewf, cls.known_values

    def test_energy(self):
        self.assertAlmostEqual(self.ewf.e_tot, self.known_values['e_tot'], 8)


class HubbardEWFTest_N10_U0_1imp(unittest.TestCase, HubbardEWFTest):
    @classmethod
    def setUpClass(cls):
        cls.mol = lattmod.Hubbard1D(10, hubbard_u=0.0, nelectron=10)
        cls.mf = lattmod.LatticeMF(cls.mol)
        cls.mf.kernel()
        cls.ewf = ewf.EWF(
                cls.mf,
                bno_threshold=1e-8,
                solver_options={
                    'conv_tol': EWF_CONV_TOL,
                },
        )
        cls.ewf.site_fragmentation()
        f = cls.ewf.add_atomic_fragment(0)
        f.make_tsymmetric_fragments(tvecs=[10, 1, 1])
        cls.ewf.kernel()
        cls.known_values = {'e_tot': -12.94427190999916}


class HubbardEWFTest_N10_U0_2imp(unittest.TestCase, HubbardEWFTest):
    @classmethod
    def setUpClass(cls):
        cls.mol = lattmod.Hubbard1D(10, hubbard_u=0.0, nelectron=10)
        cls.mf = lattmod.LatticeMF(cls.mol)
        cls.mf.kernel()
        cls.ewf = ewf.EWF(
                cls.mf,
                bno_threshold=1e-8,
                solver_options={
                    'conv_tol': EWF_CONV_TOL,
                },
        )
        cls.ewf.site_fragmentation()
        f = cls.ewf.add_atomic_fragment([0, 1])
        f.make_tsymmetric_fragments(tvecs=[5, 1, 1])
        cls.ewf.kernel()
        cls.known_values = {'e_tot': -12.94427190999916}


class HubbardEWFTest_N10_U4_1imp(unittest.TestCase, HubbardEWFTest):
    @classmethod
    def setUpClass(cls):
        cls.mol = lattmod.Hubbard1D(10, hubbard_u=4.0, nelectron=10)
        cls.mf = lattmod.LatticeMF(cls.mol)
        cls.mf.kernel()
        cls.ewf = ewf.EWF(
                cls.mf,
                bno_threshold=1e-8,
                solver_options={
                    'conv_tol': EWF_CONV_TOL,
                },
        )
        cls.ewf.site_fragmentation()
        f = cls.ewf.add_atomic_fragment(0)
        f.make_tsymmetric_fragments(tvecs=[10, 1, 1])
        cls.ewf.kernel()
        cls.known_values = {'e_tot': -6.133885588519993}


class HubbardEWFTest_N6_U6_2imp(unittest.TestCase, HubbardEWFTest):
    @classmethod
    def setUpClass(cls):
        cls.mol = lattmod.Hubbard1D(6, hubbard_u=6.0, nelectron=6)
        cls.mf = lattmod.LatticeMF(cls.mol)
        cls.mf.kernel()
        cls.ewf = ewf.EWF(
                cls.mf,
                bno_threshold=1e-6,
                sc_mode=1,
                sc_energy_tol=1e-9,
                solver_options={
                    'conv_tol': EWF_CONV_TOL,
                },
        )
        cls.ewf.site_fragmentation()
        f = cls.ewf.add_atomic_fragment([0, 1])
        f.make_tsymmetric_fragments(tvecs=[3, 1, 1])
        cls.ewf.kernel()
        cls.known_values = {'e_tot': -3.1985807202795167}


if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
