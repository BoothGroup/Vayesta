import unittest

import numpy as np

import pyscf
import pyscf.scf
import pyscf.cc

import vayesta
import vayesta.ewf


class RHF_H2O(unittest.TestCase):

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
        cls.ecc = vayesta.ewf.EWF(cls.mf, bath_type='full', solve_lambda=True)
        cls.ecc.iao_fragmentation()
        cls.ecc.add_all_atomic_fragments()
        cls.ecc.kernel()

    @classmethod
    def tearDownClass(cls):
        del cls.mol, cls.mf, cls.ccsd, cls.ecc, cls.dm1, cls.dm2

    def test_t1(self):
        atol = 1e-8
        t1 = self.ecc.get_global_t1()
        self.assertIsNone(np.testing.assert_allclose(t1, self.ccsd.t1, atol=atol))

    def test_t2(self):
        atol = 1e-8
        t2 = self.ecc.get_global_t2()
        self.assertIsNone(np.testing.assert_allclose(t2, self.ccsd.t2, atol=atol))

    def test_dm1(self):
        atol = 1e-8
        dm1 = self.ecc.make_rdm1_ccsd()
        self.assertAlmostEqual(np.trace(dm1), self.mol.nelectron)
        e0, e1 = np.linalg.eigh(dm1)[0][[0,-1]]
        self.assertGreater(e0, -atol)
        self.assertLess(e1, 2+atol)
        self.assertIsNone(np.testing.assert_allclose(dm1, self.dm1, atol=atol))

    #def test_dm1_new2(self):
    #    atol = 1e-8
    #    dm1 = self.ecc.make_rdm1_ccsd_new2()
    #    self.assertAlmostEqual(np.trace(dm1), self.mol.nelectron)
    #    e0, e1 = np.linalg.eigh(dm1)[0][[0,-1]]
    #    self.assertGreater(e0, -atol)
    #    self.assertLess(e1, 2+atol)
    #    self.assertIsNone(np.testing.assert_allclose(dm1, self.dm1, atol=atol))

    def test_dm2(self):
        atol = 1e-8
        dm2 = self.ecc.make_rdm2_ccsd()
        self.assertIsNone(np.testing.assert_allclose(dm2, self.dm2, atol=atol))

class RHF_vs_UHF_H2O(unittest.TestCase):

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
        # UHF
        cls.rhf = pyscf.scf.RHF(cls.mol)
        cls.rhf.conv_tol = 1e-12
        cls.rhf.kernel()
        assert cls.rhf.converged
        cls.uhf = cls.rhf.to_uhf()

        # Emb-CCSD
        cc_conv_tol = 1e-11
        cc_conv_tol_normt = 1e-9

        #bno = -1
        bno = 1e-4
        opts = dict(solve_lambda=True)
        solver_opts = dict(conv_tol=cc_conv_tol, conv_tol_normt=cc_conv_tol_normt)
        cls.remb = vayesta.ewf.EWF(cls.rhf, solver_options=solver_opts, **opts)
        cls.remb.iao_fragmentation()
        cls.remb.add_all_atomic_fragments()
        cls.remb.kernel(bno_threshold=bno)

        cls.uemb = vayesta.ewf.EWF(cls.uhf, solver_options=solver_opts, **opts)
        cls.uemb.iao_fragmentation()
        cls.uemb.add_all_atomic_fragments()
        cls.uemb.kernel(bno_threshold=bno)


    @classmethod
    def tearDownClass(cls):
        del cls.mol, cls.rhf, cls.uhf, cls.remb, cls.uemb

    def test_etot(self):
        atol = 1e-7
        self.assertIsNone(np.testing.assert_allclose(self.remb.e_tot, self.uemb.e_tot, atol=atol))

    def test_t1(self):
        atol = 1e-7
        t1 = self.remb.get_global_t1()
        t1a, t1b = self.uemb.get_global_t1()
        self.assertIsNone(np.testing.assert_allclose(t1, t1a, atol=atol))
        self.assertIsNone(np.testing.assert_allclose(t1, t1b, atol=atol))

    def test_t2(self):
        atol = 1e-7
        t2 = self.remb.get_global_t2()
        t2aa, t2ab, t2bb = self.uemb.get_global_t2()
        self.assertIsNone(np.testing.assert_allclose(t2aa, t2 - t2.transpose(0,1,3,2), atol=atol))
        self.assertIsNone(np.testing.assert_allclose(t2ab, t2, atol=atol))
        self.assertIsNone(np.testing.assert_allclose(t2bb, t2 - t2.transpose(0,1,3,2), atol=atol))

    def test_dm1(self):
        # TODO, currently fails - reason asymmetric use of alpha and beta fragment projector in UHF?
        atol = 1e-8
        rdm1 = self.remb.make_rdm1_ccsd()
        dm1a, dm1b = self.uemb.make_rdm1_ccsd()
        udm1 = (dm1a + dm1b)
        self.assertAlmostEqual(np.trace(udm1), self.mol.nelectron)
        self.assertAlmostEqual(np.trace(dm1a-dm1b), self.mol.spin)

        e0, e1 = np.linalg.eigh(dm1a)[0][[0,-1]]
        self.assertGreater(e0, -atol)
        self.assertLess(e1, 1+atol)
        e0, e1 = np.linalg.eigh(dm1b)[0][[0,-1]]
        self.assertGreater(e0, -atol)
        self.assertLess(e1, 1+atol)

        self.assertIsNone(np.testing.assert_allclose(rdm1, udm1, atol=atol))


class UHF_NO2(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.mol = pyscf.gto.Mole()
        cls.mol.atom = """
        N   0.0000   0.0000	0.0000
        O   0.0000   1.0989	0.4653
        O   0.0000  -1.0989     0.4653
        """
        cls.mol.basis = '6-31G'
        cls.mol.spin = 1
        cls.mol.build()
        # UHF
        cls.mf = pyscf.scf.UHF(cls.mol)
        cls.mf.conv_tol = 1e-12
        cls.mf.kernel()
        assert cls.mf.converged
        # CCSD
        cls.ccsd = pyscf.cc.UCCSD(cls.mf)
        cc_conv_tol = 1e-11
        cc_conv_tol_normt = 1e-9
        cls.ccsd.conv_tol = cc_conv_tol
        cls.ccsd.conv_tol_normt = cc_conv_tol_normt
        cls.ccsd.kernel()
        assert cls.ccsd.converged
        cls.dm1 = cls.ccsd.make_rdm1()
        #cls.dm2 = cls.ccsd.make_rdm2()
        # Emb-UCCSD
        opts = dict(solve_lambda=True)
        solver_opts= dict(conv_tol=cc_conv_tol, conv_tol_normt=cc_conv_tol_normt)
        #cls.ecc = vayesta.ewf.EWF(cls.mf, bath_type='full', store_l1=True, store_l2=True,
        #        solver_options={'conv_tol' : cc_conv_tol, 'conv_tol_normt' : cc_conv_tol_normt})
        cls.ecc = vayesta.ewf.EWF(cls.mf, solver_options=solver_opts, **opts)
        #cls.ecc.kernel(bno_threshold=1e-4)
        cls.ecc.kernel(bno_threshold=-1)

    @classmethod
    def tearDownClass(cls):
        del cls.mol, cls.mf, cls.ccsd, cls.ecc, cls.dm1

    def test_t1(self):
        atol = 1e-7
        t1a, t1b = self.ecc.get_global_t1()
        self.assertIsNone(np.testing.assert_allclose(t1a, self.ccsd.t1[0], atol=atol))
        self.assertIsNone(np.testing.assert_allclose(t1b, self.ccsd.t1[1], atol=atol))

    def test_t2(self):
        atol = 1e-7
        t2aa, t2ab, t2bb = self.ecc.get_global_t2()
        self.assertIsNone(np.testing.assert_allclose(t2aa, self.ccsd.t2[0], atol=atol))
        self.assertIsNone(np.testing.assert_allclose(t2ab, self.ccsd.t2[1], atol=atol))
        self.assertIsNone(np.testing.assert_allclose(t2bb, self.ccsd.t2[2], atol=atol))

    def test_dm1(self):
        atol = 1e-8
        dm1a, dm1b = self.ecc.make_rdm1_ccsd()

        nocca = np.count_nonzero(self.mf.mo_occ[0] > 0)
        noccb = np.count_nonzero(self.mf.mo_occ[1] > 0)
        occa, vira = np.s_[:nocca], np.s_[nocca:]
        occb, virb = np.s_[:noccb], np.s_[noccb:]

        # occ-occ
        norm = np.linalg.norm(dm1a[occa,occa] - self.dm1[0][occa,occa])
        print("a: occ-occ = %.3e" % norm)
        norm = np.linalg.norm(dm1b[occb,occb] - self.dm1[1][occb,occb])
        print("b: occ-occ = %.3e" % norm)

        # vir-vir
        norm = np.linalg.norm(dm1a[vira,vira] - self.dm1[0][vira,vira])
        print("a: vir-vir = %.3e" % norm)
        norm = np.linalg.norm(dm1b[virb,virb] - self.dm1[1][virb,virb])
        print("b: vir-vir = %.3e" % norm)

        # occ-vir
        norm = np.linalg.norm(dm1a[occa,vira] - self.dm1[0][occa,vira])
        print("a: occ-vir = %.3e" % norm)
        norm = np.linalg.norm(dm1b[occb,virb] - self.dm1[1][occb,virb])
        print("b: occ-vir = %.3e" % norm)


        self.assertIsNone(np.testing.assert_allclose(dm1a[occa,occa], self.dm1[0][occa,occa], atol=atol))
        self.assertIsNone(np.testing.assert_allclose(dm1a[occa,vira], self.dm1[0][occa,vira], atol=atol))
        self.assertIsNone(np.testing.assert_allclose(dm1a[vira,occa], self.dm1[0][vira,occa], atol=atol))
        self.assertIsNone(np.testing.assert_allclose(dm1a[vira,vira], self.dm1[0][vira,vira], atol=atol))

        self.assertIsNone(np.testing.assert_allclose(dm1b[occb,occb], self.dm1[1][occb,occb], atol=atol))
        self.assertIsNone(np.testing.assert_allclose(dm1b[occb,virb], self.dm1[1][occb,virb], atol=atol))
        self.assertIsNone(np.testing.assert_allclose(dm1b[virb,occb], self.dm1[1][virb,occb], atol=atol))
        self.assertIsNone(np.testing.assert_allclose(dm1b[virb,virb], self.dm1[1][virb,virb], atol=atol))

        self.assertIsNone(np.testing.assert_allclose(dm1a, self.dm1[0], atol=atol))
        self.assertIsNone(np.testing.assert_allclose(dm1b, self.dm1[1], atol=atol))


if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
