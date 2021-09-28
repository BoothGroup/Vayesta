import unittest
from vayesta import lattmod, rpa


class HubbardRPATest:
    ''' Abstract base class for Hubbard model DMET tests.
    '''
    @classmethod
    def setUpClass(cls):
        cls.mol = None
        cls.mf = None
        cls.rpa = None
        cls.rpa2 = None
        cls.known_values = None

    @classmethod
    def tearDownClass(cls):
        del cls.mol, cls.mf, cls.rpa, cls.rpa2, cls.known_values

    def test_energy(self):
        self.assertAlmostEqual(self.rpa.e_tot, self.known_values['e_tot'], 6)

class HubbarddRPATest(HubbardRPATest):

    def test_approaches_match(self):
        self.assertAlmostEqual(self.rpa.e_tot, self.rpa2.e_tot)



def set_up_1D_Hubbard(nsite, hubbard_u, nelectron):
    mol = lattmod.Hubbard1D(nsite, hubbard_u=hubbard_u, nelectron=nelectron)
    mf = lattmod.LatticeMF(mol)
    mf.kernel()
    return mol, mf

def set_up_2D_Hubbard(nsites, hubbard_u, nelectron):
    mol = lattmod.Hubbard2D(nsites, hubbard_u=hubbard_u, nelectron=nelectron, boundary="PBC")
    mf = lattmod.LatticeMF(mol)
    mf.kernel()
    return mol, mf

class HubbardRPAXTest_N10_U1(unittest.TestCase, HubbardRPATest):
    @classmethod
    def setUpClass(cls):
        cls.mol, cls.mf = set_up_1D_Hubbard(22, 1.0, 22)
        cls.rpa = rpa.RPA(cls.mf)
        cls.rpa.kernel("rpax")
        cls.known_values = {"e_tot": -23.027359444167207}

class HubbarddRPATest_N30_U8(unittest.TestCase, HubbarddRPATest):
    @classmethod
    def setUpClass(cls):
        cls.mol, cls.mf = set_up_1D_Hubbard(30, 8.0, 30)
        cls.rpa = rpa.RPA(cls.mf)
        cls.rpa.kernel("drpa")
        cls.rpa2 = rpa.ssRPA(cls.mf)
        cls.rpa2.kernel()
        cls.known_values = {"e_tot": -1.2875037735538157}

class HubbardRPAXTest_N8x8_U2_50elec(unittest.TestCase, HubbardRPATest):
    @classmethod
    def setUpClass(cls):
        cls.mol, cls.mf = set_up_2D_Hubbard((8,8), 2.0, 50)
        cls.rpa = rpa.RPA(cls.mf)
        cls.rpa.kernel("rpax")
        cls.known_values = {"e_tot": -85.05266742812762}

class HubbarddRPAXTest_N6x6_U2_26elec(unittest.TestCase, HubbarddRPATest):
    @classmethod
    def setUpClass(cls):
        cls.mol, cls.mf = set_up_2D_Hubbard((6,6), 6.0, 26)
        cls.rpa = rpa.RPA(cls.mf)
        cls.rpa.kernel("drpa")
        cls.rpa2 = rpa.ssRPA(cls.mf)
        cls.rpa2.kernel()
        cls.known_values = {"e_tot": -41.069807438783826}



if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
