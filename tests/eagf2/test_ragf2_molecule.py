import unittest
import numpy as np
from pyscf import gto, scf, lib, agf2
from vayesta.eagf2 import ragf2


class MoleculeRAGF2Test:
    ''' Abstract base class for molecular RAGF2 tests.
    '''

    @classmethod
    def setUpClass(cls):
        cls.mol = None
        cls.mf = None
        cls.ragf2 = None
        cls.ragf2_ref = None
        cls.known_values = None

    @classmethod
    def tearDownClass(cls):
        del cls.mol, cls.mf, cls.ragf2, cls.ragf2_ref, cls.known_values


class KnownMoleculeRAGF2Test(MoleculeRAGF2Test):
    def test_energy(self):
        self.assertAlmostEqual(self.ragf2.e_tot, self.known_values['e_tot'], 8)

    def test_ip(self):
        self.assertAlmostEqual(self.ragf2.e_ip, self.known_values['e_ip'], 8)

    def test_ea(self):
        self.assertAlmostEqual(self.ragf2.e_ea, self.known_values['e_ea'], 8)


class ComparedMoleculeRAGF2Test(MoleculeRAGF2Test):
    def test_energy(self):
        self.assertAlmostEqual(self.ragf2.e_tot, self.ragf2_ref.e_tot, 8)

    def test_ip(self):
        self.assertAlmostEqual(self.ragf2.e_ip, self.ragf2_ref.ipagf2()[0][0], 8)

    def test_ea(self):
        self.assertAlmostEqual(self.ragf2.e_ea, self.ragf2_ref.eaagf2()[0][0], 8)

    def test_gf_moms(self):
        self.assertAlmostEqual(np.max(np.abs(self.ragf2.gf.moment([0, 1]) - self.ragf2_ref.gf.moment([0, 1]))), 0, 8)

    def test_se_moms(self):
        self.assertAlmostEqual(np.max(np.abs(self.ragf2.se.moment([0, 1]) - self.ragf2_ref.se.moment([0, 1]))), 0, 8)


class LiH_ccpvdz_Test(KnownMoleculeRAGF2Test, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mol = gto.Mole()
        cls.mol.atom = 'Li 0 0 0; H 0 0 1.4'
        cls.mol.basis = 'cc-pvdz'
        cls.mol.max_memory = 1e9
        cls.mol.verbose = 0
        cls.mol.build()

        cls.mf = scf.RHF(cls.mol)
        cls.mf.conv_tol = 1e-12
        cls.mf.kernel()

        cls.ragf2 = ragf2.RAGF2(cls.mf)
        cls.ragf2.opts.conv_tol = 1e-10
        cls.ragf2.opts.conv_tol_rdm1 = 1e-12
        cls.ragf2.opts.conv_tol_nelec = 1e-12
        cls.ragf2.opts.conv_tol_nelec_factor = 1e-4
        cls.ragf2.kernel()

        cls.ragf2_ref = None

        cls.known_values = {
                'e_tot': -7.998013606039682,
                'e_ip': 0.3068625478488375,
                'e_ea': 0.0022960662769273463,
        }


class N2_ccpvdz_DF_Test(ComparedMoleculeRAGF2Test, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mol = gto.Mole()
        cls.mol.atom = 'N 0 0 0; N 0 0 1.1'
        cls.mol.basis = 'cc-pvdz'
        cls.mol.max_memory = 1e9
        cls.mol.verbose = 0
        cls.mol.build()

        cls.mf = scf.RHF(cls.mol)
        cls.mf = cls.mf.density_fit()
        cls.mf.conv_tol = 1e-12
        cls.mf.kernel()

        cls.ragf2 = ragf2.RAGF2(cls.mf)
        cls.ragf2.opts.conv_tol = 1e-10
        cls.ragf2.opts.conv_tol_rdm1 = 1e-12
        cls.ragf2.opts.conv_tol_nelec = 1e-12
        cls.ragf2.opts.conv_tol_nelec_factor = 1e-4
        cls.ragf2.kernel()

        cls.ragf2_ref = agf2.RAGF2(cls.mf)
        cls.ragf2_ref.conv_tol = 1e-10
        cls.ragf2_ref.conv_tol_rdm1 = 1e-12
        cls.ragf2_ref.conv_tol_nelec = 1e-12
        cls.ragf2_ref.kernel()

        cls.known_values = None


if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
