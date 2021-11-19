import unittest

import numpy as np

import pyscf
import pyscf.scf
import pyscf.cc

import vayesta
import vayesta.ewf


class T_Amplitudes(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.mol = pyscf.gto.Mole()
        cls.mol.atom = """
        O  0.0000   0.0000   0.1173
        H  0.0000   0.7572  -0.4692
        H  0.0000  -0.7572  -0.4692
        """
        cls.mol.basis = 'cc-pVDZ'
        cls.mol.build()
        # RHF
        cls.mf = pyscf.scf.RHF(cls.mol)
        cls.mf.conv_tol = 1e-12
        cls.mf.kernel()
        assert cls.mf.converged
        # CCSD
        cls.ccsd = pyscf.cc.CCSD(cls.mf)
        cls.ccsd.kernel()
        assert cls.ccsd.converged
        cls.dm1 = cls.ccsd.make_rdm1()
        cls.dm2 = cls.ccsd.make_rdm2()
        # Emb-CCSD
        cls.ecc = vayesta.ewf.EWF(cls.mf, bath_type='full', make_rdm1=True, make_rdm2=True)
        cls.ecc.iao_fragmentation()
        cls.ecc.add_all_atomic_fragments()
        cls.ecc.kernel()

    @classmethod
    def tearDownClass(cls):
        del cls.mol, cls.mf, cls.ccsd, cls.ecc, cls.dm1, cls.dm2

    def test_t1(self):
        atol = 1e-8
        t1 = self.ecc.get_t1()
        self.assertIsNone(np.testing.assert_allclose(t1, self.ccsd.t1, atol=atol))

    def test_t2(self):
        atol = 1e-8
        t2 = self.ecc.get_t2()
        self.assertIsNone(np.testing.assert_allclose(t2, self.ccsd.t2, atol=atol))

    def test_t12(self):
        atol = 1e-8
        t1 = self.ecc.get_t1()
        t2 = self.ecc.get_t2()
        self.assertIsNone(np.testing.assert_allclose(t1, self.ccsd.t1, atol=atol))
        self.assertIsNone(np.testing.assert_allclose(t2, self.ccsd.t2, atol=atol))

    def test_dm1(self):
        atol = 1e-8
        dm1 = self.ecc.make_rdm1_ccsd()
        self.assertIsNone(np.testing.assert_allclose(dm1, self.dm1, atol=atol))

    def test_dm2(self):
        atol = 1e-8
        dm2 = self.ecc.make_rdm2_ccsd()
        self.assertIsNone(np.testing.assert_allclose(dm2, self.dm2, atol=atol))


class T_Amplitudes_UHF(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.mol = pyscf.gto.Mole()
        cls.mol.atom = """
        H 0 0 -2.5
        H 0 0 +2.5
        """
        cls.mol.basis = 'cc-pVDZ'
        cls.mol.build()
        # UHF
        cls.mf = pyscf.scf.UHF(cls.mol)
        dma, dmb = cls.mf.get_init_guess()
        dma[0,0] += 0.1
        dmb[0,0] -= 0.1
        cls.mf.conv_tol = 1e-12
        cls.mf.kernel(dm0=(dma, dmb))
        assert cls.mf.converged
        assert cls.mf.spin_square()[0] > 1e-2
        # CCSD
        cls.ccsd = pyscf.cc.UCCSD(cls.mf)
        cls.ccsd.conv_tol = 1e-12
        cls.ccsd.conv_tol_normt = 1e-6
        cls.ccsd.kernel()
        assert cls.ccsd.converged
        cls.dm1 = cls.ccsd.make_rdm1()
        cls.dm2 = cls.ccsd.make_rdm2()
        # Emb-UCCSD
        cls.ecc = vayesta.ewf.EWF(cls.mf, bath_type='full',
                solver_options={'conv_tol' : 1e-12, 'conv_tol_normt' : 1e-6})
        cls.ecc.iao_fragmentation()
        cls.ecc.add_all_atomic_fragments()
        cls.ecc.kernel()

    @classmethod
    def tearDownClass(cls):
        del cls.mol, cls.mf, cls.ccsd, cls.ecc, cls.dm1, cls.dm2

    def test_t1(self):
        atol = 1e-6
        t1a, t1b = self.ecc.get_t1()
        self.assertIsNone(np.testing.assert_allclose(t1a, self.ccsd.t1[0], atol=atol))
        self.assertIsNone(np.testing.assert_allclose(t1b, self.ccsd.t1[1], atol=atol))

    def test_t2(self):
        atol = 1e-6
        t2aa, t2ab, t2bb = self.ecc.get_t2()
        self.assertIsNone(np.testing.assert_allclose(t2aa, self.ccsd.t2[0], atol=atol))
        self.assertIsNone(np.testing.assert_allclose(t2ab, self.ccsd.t2[1], atol=atol))
        self.assertIsNone(np.testing.assert_allclose(t2bb, self.ccsd.t2[2], atol=atol))

    def test_t12(self):
        atol = 1e-6
        (t1a, t1b) = self.ecc.get_t1()
        (t2aa, t2ab, t2bb) = self.ecc.get_t2()
        self.assertIsNone(np.testing.assert_allclose(t1a, self.ccsd.t1[0], atol=atol))
        self.assertIsNone(np.testing.assert_allclose(t1b, self.ccsd.t1[1], atol=atol))
        self.assertIsNone(np.testing.assert_allclose(t2aa, self.ccsd.t2[0], atol=atol))
        self.assertIsNone(np.testing.assert_allclose(t2ab, self.ccsd.t2[1], atol=atol))
        self.assertIsNone(np.testing.assert_allclose(t2bb, self.ccsd.t2[2], atol=atol))

    # TODO
    #def test_dm1(self):
    #    atol = 1e-8
    #    dm1a, dm1b = self.ecc.make_rdm1_ccsd()
    #    self.assertIsNone(np.testing.assert_allclose(dm1a, self.dm1[0], atol=atol))
    #    self.assertIsNone(np.testing.assert_allclose(dm1b, self.dm1[1], atol=atol))

    #def test_dm2(self):
    #    atol = 1e-8
    #    dm2aa, dm2ab, dm2bb = self.ecc.make_rdm2_ccsd()
    #    self.assertIsNone(np.testing.assert_allclose(dm2aa, self.dm2[0], atol=atol))
    #    self.assertIsNone(np.testing.assert_allclose(dm2ab, self.dm2[1], atol=atol))
    #    self.assertIsNone(np.testing.assert_allclose(dm2bb, self.dm2[2], atol=atol))

if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
