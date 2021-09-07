import unittest
from vayesta import lattmod, dmet


class HubbardDMETTest:
    ''' Abstract base class for Hubbard model DMET tests.
    '''

    @classmethod
    def setUpClass(cls):
        cls.mol = None
        cls.mf = None
        cls.dmet = None
        cls.known_values = None

    @classmethod
    def tearDownClass(cls):
        del cls.mol, cls.mf, cls.dmet, cls.known_values

    def test_converged(self):
        self.assertTrue(self.dmet.converged)

    def test_energy(self):
        self.assertAlmostEqual(self.dmet.e_tot, self.known_values['e_tot'], 6)


class HubbardDMETTest_N10_U0_1imp(unittest.TestCase, HubbardDMETTest):
    @classmethod
    def setUpClass(cls):
        cls.mol = lattmod.Hubbard1D(10, hubbard_u=0.0, nelectron=10)
        cls.mf = lattmod.LatticeMF(cls.mol)
        cls.mf.kernel()
        cls.dmet = dmet.DMET(cls.mf, solver='FCI', fragment_type='site', charge_consistent=False)
        f = cls.dmet.make_atom_fragment(0)
        symfrags = f.make_tsymmetric_fragments(tvecs=[10, 1, 1])
        cls.dmet.kernel()
        cls.known_values = {'e_tot': -12.944271909999166}


class HubbardDMETTest_N10_U0_2imp(unittest.TestCase, HubbardDMETTest):
    @classmethod
    def setUpClass(cls):
        cls.mol = lattmod.Hubbard1D(10, hubbard_u=0.0, nelectron=10)
        cls.mf = lattmod.LatticeMF(cls.mol)
        cls.mf.kernel()
        cls.dmet = dmet.DMET(cls.mf, solver='FCI', fragment_type='site', charge_consistent=False)
        f = cls.dmet.make_atom_fragment([0, 1])
        symfrags = f.make_tsymmetric_fragments(tvecs=[5, 1, 1])
        cls.dmet.kernel()
        cls.known_values = {'e_tot': -12.944271909999154}


class HubbardDMETTest_N20_U0_2imp(unittest.TestCase, HubbardDMETTest):
    @classmethod
    def setUpClass(cls):
        cls.mol = lattmod.Hubbard1D(20, hubbard_u=0.0, nelectron=20)
        cls.mf = lattmod.LatticeMF(cls.mol)
        cls.mf.kernel()
        cls.dmet = dmet.DMET(cls.mf, solver='FCI', fragment_type='site', charge_consistent=False)
        f = cls.dmet.make_atom_fragment([0, 1])
        symfrags = f.make_tsymmetric_fragments(tvecs=[10, 1, 1])
        cls.dmet.kernel()
        cls.known_values = {'e_tot': -25.56981288599864}


class HubbardDMETTest_N10_U4_1imp(unittest.TestCase, HubbardDMETTest):
    @classmethod
    def setUpClass(cls):
        cls.mol = lattmod.Hubbard1D(10, hubbard_u=4.0, nelectron=10)
        cls.mf = lattmod.LatticeMF(cls.mol)
        cls.mf.kernel()
        cls.dmet = dmet.DMET(cls.mf, solver='FCI', fragment_type='site', charge_consistent=False)
        f = cls.dmet.make_atom_fragment(0)
        symfrags = f.make_tsymmetric_fragments(tvecs=[10, 1, 1])
        cls.dmet.kernel()
        cls.known_values = {'e_tot': -6.109901024380408}


class HubbardDMETTest_N10_U4_2imp(unittest.TestCase, HubbardDMETTest):
    @classmethod
    def setUpClass(cls):
        cls.mol = lattmod.Hubbard1D(10, hubbard_u=4.0, nelectron=10)
        cls.mf = lattmod.LatticeMF(cls.mol)
        cls.mf.kernel()
        cls.dmet = dmet.DMET(cls.mf, solver='FCI', fragment_type='site', charge_consistent=False)
        f = cls.dmet.make_atom_fragment([0, 1])
        symfrags = f.make_tsymmetric_fragments(tvecs=[5, 1, 1])
        cls.dmet.kernel()
        cls.known_values = {'e_tot': -5.982495317229596}


class HubbardDMETTest_N10_U4_5imp(unittest.TestCase, HubbardDMETTest):
    @classmethod
    def setUpClass(cls):
        cls.mol = lattmod.Hubbard1D(10, hubbard_u=4.0, nelectron=10)
        cls.mf = lattmod.LatticeMF(cls.mol)
        cls.mf.kernel()
        cls.dmet = dmet.DMET(cls.mf, solver='FCI', fragment_type='site', charge_consistent=False)
        f = cls.dmet.make_atom_fragment(list(range(5)))
        symfrags = f.make_tsymmetric_fragments(tvecs=[2, 1, 1])
        cls.dmet.kernel()
        cls.known_values = {'e_tot': -5.834322635766257}


class HubbardDMETTest_N12_U8_2imp(unittest.TestCase, HubbardDMETTest):
    @classmethod
    def setUpClass(cls):
        cls.mol = lattmod.Hubbard1D(12, hubbard_u=8.0, nelectron=12)
        cls.mf = lattmod.LatticeMF(cls.mol)
        cls.mf.kernel()
        cls.dmet = dmet.DMET(cls.mf, solver='FCI', fragment_type='site', charge_consistent=False)
        f = cls.dmet.make_atom_fragment([0, 1])
        symfrags = f.make_tsymmetric_fragments(tvecs=[6, 1, 1])
        cls.dmet.kernel()
        cls.known_values = {'e_tot': -4.022943661738756}


class HubbardDMETTest_N6x6_U1_2x2imp(unittest.TestCase, HubbardDMETTest):
    @classmethod
    def setUpClass(cls):
        cls.mol = lattmod.Hubbard2D((6, 6), hubbard_u=1.0, nelectron=26, tiles=(2, 2), boundary="PBC")
        cls.mf = lattmod.LatticeMF(cls.mol)
        cls.mf.kernel()
        cls.dmet = dmet.DMET(cls.mf, solver='FCI', fragment_type='site', charge_consistent=False)
        f = cls.dmet.make_atom_fragment(list(range(4)))
        symfrags = f.make_tsymmetric_fragments(tvecs=[3, 3, 1])
        cls.dmet.kernel()
        cls.known_values = {'e_tot': -51.744964087157626}


class HubbardDMETTest_N6x6_U6_2x2imp(unittest.TestCase, HubbardDMETTest):
    @classmethod
    def setUpClass(cls):
        cls.mol = lattmod.Hubbard2D((6, 6), hubbard_u=6.0, nelectron=26, tiles=(2, 2), boundary="PBC")
        cls.mf = lattmod.LatticeMF(cls.mol)
        cls.mf.kernel()
        cls.dmet = dmet.DMET(cls.mf, solver='FCI', fragment_type='site', charge_consistent=False)
        f = cls.dmet.make_atom_fragment(list(range(4)))
        symfrags = f.make_tsymmetric_fragments(tvecs=[3, 3, 1])
        cls.dmet.kernel()
        cls.known_values = {'e_tot': -39.51961645103434}


class HubbardDMETTest_N8x8_U4_1x1imp(unittest.TestCase, HubbardDMETTest):
    @classmethod
    def setUpClass(cls):
        cls.mol = lattmod.Hubbard2D((8, 8), hubbard_u=4.0, nelectron=50, tiles=(1, 1), boundary="PBC")
        cls.mf = lattmod.LatticeMF(cls.mol)
        cls.mf.kernel()
        cls.dmet = dmet.DMET(cls.mf, solver='FCI', fragment_type='site', charge_consistent=False)
        f = cls.dmet.make_atom_fragment(0)
        symfrags = f.make_tsymmetric_fragments(tvecs=[8, 8, 1])
        cls.dmet.kernel()
        cls.known_values = {'e_tot': -74.93240779775238}


class HubbardDMETTest_N8x8_U3_2x2imp(unittest.TestCase, HubbardDMETTest):
    @classmethod
    def setUpClass(cls):
        cls.mol = lattmod.Hubbard2D((8, 8), hubbard_u=3.0, nelectron=50, tiles=(2, 2), boundary="PBC")
        cls.mf = lattmod.LatticeMF(cls.mol)
        cls.mf.kernel()
        cls.dmet = dmet.DMET(cls.mf, solver='FCI', fragment_type='site', charge_consistent=False)
        f = cls.dmet.make_atom_fragment(list(range(4)))
        symfrags = f.make_tsymmetric_fragments(tvecs=[4, 4, 1])
        cls.dmet.kernel()
        cls.known_values = {'e_tot': -79.06298997526869}


if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
