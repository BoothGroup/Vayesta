import unittest
from vayesta import lattmod, edmet


class HubbardEDMETTest:
    ''' Abstract base class for Hubbard model EDMET tests.
    '''

    @classmethod
    def setUpClass(cls):
        cls.mol = None
        cls.mf = None
        cls.edmet = None
        cls.known_values = None

    @classmethod
    def tearDownClass(cls):
        del cls.mol, cls.mf, cls.edmet, cls.known_values

    def test_energy(self):
        self.assertAlmostEqual(self.edmet.e_tot, self.known_values['e_tot'], 6)


class HubbardEDMETTest_N10_U4_1imp_1occ(unittest.TestCase, HubbardEDMETTest):
    @classmethod
    def setUpClass(cls):
        cls.mol = lattmod.Hubbard1D(10, hubbard_u=4.0, nelectron=10)
        cls.mf = lattmod.LatticeMF(cls.mol)
        cls.mf.kernel()
        cls.edmet = edmet.EDMET(cls.mf, solver='EBFCI', bos_occ_cutoff=1)
        cls.edmet.site_fragmentation()
        f = cls.edmet.add_atomic_fragment(0)
        symfrags = f.make_tsymmetric_fragments(tvecs=[10, 1, 1])
        cls.edmet.kernel()
        cls.known_values = {'e_tot': -6.118248173618246}


class HubbardEDMETTest_N10_U4_1imp_10occ(unittest.TestCase, HubbardEDMETTest):
    @classmethod
    def setUpClass(cls):
        cls.mol = lattmod.Hubbard1D(10, hubbard_u=4.0, nelectron=10)
        cls.mf = lattmod.LatticeMF(cls.mol)
        cls.mf.kernel()
        cls.edmet = edmet.EDMET(cls.mf, solver='EBFCI', bos_occ_cutoff=10)
        cls.edmet.site_fragmentation()
        f = cls.edmet.add_atomic_fragment(0)
        symfrags = f.make_tsymmetric_fragments(tvecs=[10, 1, 1])
        cls.edmet.kernel()
        cls.known_values = {'e_tot': -6.1181667874754035}


class HubbardEDMETTest_N10_U4_2imp_2occ(unittest.TestCase, HubbardEDMETTest):
    @classmethod
    def setUpClass(cls):
        cls.mol = lattmod.Hubbard1D(10, hubbard_u=0.5, nelectron=10)
        cls.mf = lattmod.LatticeMF(cls.mol)
        cls.mf.kernel()
        cls.edmet = edmet.EDMET(cls.mf, solver='EBFCI', bos_occ_cutoff=2)
        cls.edmet.site_fragmentation()
        f = cls.edmet.add_atomic_fragment([0, 1])
        symfrags = f.make_tsymmetric_fragments(tvecs=[5, 1, 1])
        cls.edmet.kernel()
        cls.known_values = {'e_tot': -11.74139024719132}


class HubbardEDMETTest_N6x6_U4_2x1imp_2occ(unittest.TestCase, HubbardEDMETTest):
    @classmethod
    def setUpClass(cls):
        cls.mol = lattmod.Hubbard2D((6, 6), hubbard_u=4.0, nelectron=26, tiles=(2,1), boundary="PBC")
        cls.mf = lattmod.LatticeMF(cls.mol)
        cls.mf.kernel()
        cls.edmet = edmet.EDMET(cls.mf, solver='EBFCI', bos_occ_cutoff=2)
        cls.edmet.site_fragmentation()
        f = cls.edmet.add_atomic_fragment([0, 1])
        symfrags = f.make_tsymmetric_fragments(tvecs=[3, 6, 1])
        cls.edmet.kernel()
        cls.known_values = {'e_tot': -43.387417397200636}


class HubbardEDMETTest_N6x6_U6_1x1imp_5occ(unittest.TestCase, HubbardEDMETTest):
    @classmethod
    def setUpClass(cls):
        cls.mol = lattmod.Hubbard2D((6, 6), hubbard_u=6.0, nelectron=26, tiles=(1,1), boundary="PBC")
        cls.mf = lattmod.LatticeMF(cls.mol)
        cls.mf.kernel()
        cls.edmet = edmet.EDMET(cls.mf, solver='EBFCI', bos_occ_cutoff=5)
        cls.edmet.site_fragmentation()
        f = cls.edmet.add_atomic_fragment(0)
        symfrags = f.make_tsymmetric_fragments(tvecs=[6, 6, 1])
        cls.edmet.kernel()
        cls.known_values = {'e_tot': -41.295421512183374}


class HubbardEDMETTest_N8x8_U2_1x2imp_5occ(unittest.TestCase, HubbardEDMETTest):
    @classmethod
    def setUpClass(cls):
        cls.mol = lattmod.Hubbard2D((8, 8), hubbard_u=2.0, nelectron=50, tiles=(1,2), boundary="PBC")
        cls.mf = lattmod.LatticeMF(cls.mol)
        cls.mf.kernel()
        cls.edmet = edmet.EDMET(cls.mf, solver='EBFCI', bos_occ_cutoff=5)
        cls.edmet.site_fragmentation()
        f = cls.edmet.add_atomic_fragment([0, 1])
        symfrags = f.make_tsymmetric_fragments(tvecs=[8, 4, 1])
        cls.edmet.kernel()
        cls.known_values = {'e_tot': -84.96249789942502}


if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
