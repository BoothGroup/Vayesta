import unittest
import numpy as np
import vayesta
import vayesta.ewf
from vayesta.core.util import cache
from vayesta.tests import testsystems
from vayesta.tests.common import TestCase

class Test_MP2(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.mf = testsystems.h2_dz.rhf()
        cls.cc = testsystems.h2_dz.rmp2()

    @classmethod
    def tearDownClass(cls):
        del cls.mf
        del cls.cc
        cls.emb.cache_clear()

    @classmethod
    @cache
    def emb(cls, bno_threshold):
        emb = vayesta.ewf.EWF(cls.mf, bath_options=dict(threshold=bno_threshold), solver='MP2')
        emb.kernel()
        return emb

    # --- Full bath tests:

    def test_energy(self):
        emb = self.emb(-1)
        self.assertAllclose(emb.e_corr, self.cc.e_corr, rtol=0)
        self.assertAllclose(emb.e_tot, self.cc.e_tot, rtol=0)

    def test_dm_energy(self):
        emb = self.emb(-1)
        self.assertAllclose(emb.get_dm_energy(), self.cc.e_tot, rtol=0)
        self.assertAllclose(emb.get_dm_corr_energy(), self.cc.e_corr, rtol=0)

    def test_t_amplitudes(self):
        emb = self.emb(-1)
        t2 = emb.get_global_t2()
        self.assertAllclose(t2, self.cc.t2)

    def test_global_dm1(self):
        emb = self.emb(-1)
        dm1_exact = self.cc.make_rdm1()
        dm1 = emb.make_rdm1(slow=True)
        self.assertAllclose(dm1, dm1_exact)
        dm1 = emb.make_rdm1(late_t2_sym=True)
        self.assertAllclose(dm1, dm1_exact)
        dm1 = emb.make_rdm1(late_t2_sym=False)
        self.assertAllclose(dm1, dm1_exact)

class Test_CCSD(Test_MP2):

    @classmethod
    def setUpClass(cls):
        cls.mf = testsystems.h2_dz.rhf()
        cls.cc = testsystems.h2_dz.rccsd()

    @classmethod
    @cache
    def emb(cls, bno_threshold):
        solver_opts = dict(conv_tol= 1e-10, conv_tol_normt=1e-8)
        emb = vayesta.ewf.EWF(cls.mf, bath_options=dict(threshold=bno_threshold), solve_lambda=True,
                solver_options=solver_opts)
        emb.kernel()
        return emb

    def test_t_amplitudes(self):
        emb = self.emb(-1)
        t1 = emb.get_global_t1()
        t2 = emb.get_global_t2()
        self.assertAllclose(t1, self.cc.t1)
        self.assertAllclose(t2, self.cc.t2)

    def test_l_amplitudes(self):
        emb = self.emb(-1)
        l1 = emb.get_global_l1()
        l2 = emb.get_global_l2()
        self.assertAllclose(l1, self.cc.l1)
        self.assertAllclose(l2, self.cc.l2)

    def test_cluster_dm1(self):
        emb = self.emb(-1)
        t = np.dot(self.mf.get_ovlp(), self.mf.mo_coeff)
        dm1_exact = self.cc.make_rdm1(ao_repr=True)
        for x in emb.fragments:
            dm1 = x.results.wf.make_rdm1(ao_basis=True)
            self.assertAllclose(dm1, dm1_exact)

    def test_cluster_dm2(self):
        emb = self.emb(-1)
        t = np.dot(self.mf.get_ovlp(), self.mf.mo_coeff)
        dm2_exact = self.cc.make_rdm2(ao_repr=True)
        for x in emb.fragments:
            dm2 = x.results.wf.make_rdm2(ao_basis=True)
            self.assertAllclose(dm2, dm2_exact)

    def test_global_dm2(self):
        emb = self.emb(-1)
        dm2_exact = self.cc.make_rdm2()
        dm2 = emb._make_rdm2_ccsd_global_wf()
        self.assertAllclose(dm2, dm2_exact)


class Test_UMP2(Test_MP2):

    @classmethod
    def setUpClass(cls):
        cls.mf = testsystems.h2anion_dz.uhf()
        cls.cc = testsystems.h2anion_dz.ump2()

    def test_global_dm1(self):
        emb = self.emb(-1)
        dm1 = emb._make_rdm1_mp2()
        dm1_exact = self.cc.make_rdm1()
        self.assertAllclose(dm1, dm1_exact)


class Test_UCCSD(Test_CCSD):

    @classmethod
    def setUpClass(cls):
        cls.mf = testsystems.h2anion_dz.uhf()
        cls.cc = testsystems.h2anion_dz.uccsd()

    def test_global_dm2(self):
        pass

# --- FCI

class Test_FCI(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.mf = testsystems.h2_dz.rhf()
        cls.fci = testsystems.h2_dz.rfci()

    @classmethod
    def tearDownClass(cls):
        del cls.mf
        del cls.fci
        cls.emb.cache_clear()

    @classmethod
    @cache
    def emb(cls, bno_threshold):
        emb = vayesta.ewf.EWF(cls.mf, solver='FCI', bath_options=dict(threshold=bno_threshold))
        emb.kernel()
        return emb

    def test_energy(self):
        emb = self.emb(-1)
        self.assertAllclose(emb.e_tot, self.fci.e_tot, rtol=0)

class Test_UFCI(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.mf = testsystems.h2anion_dz.uhf()
        cls.fci = testsystems.h2anion_dz.ufci()

class Test_UFCI_dissoc(TestCase):

    @classmethod
    def setUpClass(cls):
        # TODO ensure this tests works if density fitting is used.
        cls.mf = testsystems.h2_sto3g_dissoc.uhf_stable()
        cls.fci = testsystems.h2_sto3g_dissoc.ufci()

    @classmethod
    def tearDownClass(cls):
        del cls.mf
        del cls.fci
        cls.emb.cache_clear()

    @classmethod
    @cache
    def emb(cls, bno_threshold):
        emb = vayesta.ewf.EWF(cls.mf, bath_options=dict(bathtype="dmet"), solver='FCI')
        emb.kernel()
        return emb

    def test_energy(self):
        emb = self.emb(-1)
        self.assertAllclose(emb.e_tot, self.fci.e_tot, rtol=0)


if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
