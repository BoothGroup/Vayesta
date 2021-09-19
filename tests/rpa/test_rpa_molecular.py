import unittest
from vayesta import lattmod, rpa
from pyscf import gto, scf

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


class MoleculeRPAXTest_LiH_ccpvdz(unittest.TestCase, HubbardRPATest):
    @classmethod
    def setUpClass(cls):
        cls.mol = gto.Mole()
        cls.mol.atom = 'Li 0 0 0; H 0 0 1.4'
        cls.mol.basis = 'cc-pvdz'
        cls.mol.verbose = 0
        cls.mol.max_memory = 1e9
        cls.mol.build()

        cls.mf = scf.RHF(cls.mol)
        cls.mf.conv_tol = 1e-12
        cls.mf.kernel()

        cls.rpa = rpa.RPA(cls.mf)
        cls.rpa.kernel("rpax")
        cls.known_values = {"e_tot":-8.021765236699629}


class MoleculedRPATest_LiH_ccpvdz(unittest.TestCase, HubbarddRPATest):
    @classmethod
    def setUpClass(cls):
        cls.mol = gto.Mole()
        cls.mol.atom = 'Li 0 0 0; H 0 0 3.0'
        cls.mol.basis = 'cc-pvtz'
        cls.mol.verbose = 0
        cls.mol.max_memory = 1e9
        cls.mol.build()

        cls.mf = scf.RHF(cls.mol)
        cls.mf.conv_tol = 1e-12
        cls.mf.kernel()

        cls.rpa = rpa.RPA(cls.mf)
        cls.rpa.kernel("drpa")
        cls.rpa2 = rpa.dRPA(cls.mf)
        cls.rpa2.kernel()
        cls.known_values = {"e_tot":-7.980883936038881}

if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
