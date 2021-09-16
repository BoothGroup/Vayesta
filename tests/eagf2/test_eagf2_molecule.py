import unittest
import numpy as np
from pyscf import gto, scf, lib
from vayesta.eagf2 import eagf2, ragf2


class MoleculeEAGF2Test:
    ''' Abstract base class for molecular EAGF2 tests.
    '''

    @classmethod
    def setUpClass(cls):
        cls.mol = None
        cls.mf = None
        cls.agf2 = None
        cls.eagf2 = None
        cls.known_values = None

    @classmethod
    def tearDownClass(cls):
        del cls.mol, cls.mf, cls.agf2, cls.eagf2, cls.known_values


class KnownMoleculeEAGF2Test(MoleculeEAGF2Test):
    def test_energy(self):
        self.assertAlmostEqual(self.eagf2.e_tot, cls.known_values['e_tot'], 8)

    def test_ip(self):
        self.assertAlmostEqual(self.eagf2.e_ip, cls.known_values['e_ip'], 8)

    def test_ea(self):
        self.assertAlmostEqual(self.eagf2.e_ea, cls.known_values['e_ea'], 8)


class ExactMoleculeEAGF2Test(MoleculeEAGF2Test):
    def test_ip(self):
        self.assertAlmostEqual(self.eagf2.e_ip, self.agf2.e_ip, 8)

    def test_ea(self):
        self.assertAlmostEqual(self.eagf2.e_ea, self.agf2.e_ea, 8)

    def test_gf_moms(self):
        self.assertAlmostEqual(np.max(np.abs(self.eagf2.results.gf.moment(0) - self.agf2.gf.moment(0))), 0, 8)
        self.assertAlmostEqual(np.max(np.abs(self.eagf2.results.gf.moment(1) - self.agf2.gf.moment(1))), 0, 8)


class LiH_ccpvdz_Lowdin_all_exact_Test(ExactMoleculeEAGF2Test, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mol = gto.Mole()
        cls.mol.atom = 'Li 0 0 0; H 0 0 1.4'
        cls.mol.basis = 'cc-pvdz'
        cls.mol.max_memory = 1e9
        cls.mol.verbose = 0
        cls.mol.build()

        cls.mf = scf.RHF(cls.mol)
        cls.mf.conv_tol = 1e-14
        cls.mf.kernel()

        options = {
                'conv_tol': 1e-10,
                'conv_tol_rdm1': 1e-14,
                'conv_tol_nelec': 1e-12,
                'conv_tol_nelec_factor': 1e-4,
                'weight_tol': 0.0,
        }

        cls.eagf2 = eagf2.EAGF2(cls.mf, fragment_type='Lowdin-AO', bath_type='all', **options)
        cls.eagf2.make_all_atom_fragments()
        cls.eagf2.kernel()

        cls.agf2 = ragf2.RAGF2(cls.mf, **options)
        cls.agf2.kernel()

        cls.known_values = None


class N2_631g_Lowdin_power_exact_Test(ExactMoleculeEAGF2Test, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mol = gto.Mole()
        cls.mol.atom = 'N 0 0 0; N 0 0 1.1'
        cls.mol.basis = '6-31g'
        cls.mol.max_memory = 1e9
        cls.mol.verbose = 0
        cls.mol.build()

        cls.mf = scf.RHF(cls.mol)
        cls.mf.conv_tol = 1e-14
        cls.mf.kernel()

        options = {
                'conv_tol': 1e-10,
                'conv_tol_rdm1': 1e-14,
                'conv_tol_nelec': 1e-12,
                'conv_tol_nelec_factor': 1e-4,
                'weight_tol': 0.0,
        }

        cls.eagf2 = eagf2.EAGF2(cls.mf, fragment_type='Lowdin-AO', bath_type='power', max_bath_order=100, **options)
        cls.eagf2.make_all_atom_fragments()
        cls.eagf2.kernel()

        cls.agf2 = ragf2.RAGF2(cls.mf, **options)
        cls.agf2.kernel()

        cls.known_values = None


if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
