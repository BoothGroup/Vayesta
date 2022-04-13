import unittest

import numpy as np

import vayesta
import vayesta.ewf
from vayesta.core.util import cache
from vayesta.tests import testsystems
from vayesta.tests.common import TestCase

class TestH2(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.mf = testsystems.h2_dz.rhf()
        cls.cc = testsystems.h2_dz.rccsd()

    @classmethod
    def tearDownClass(cls):
        del cls.mf
        del cls.cc
        cls.emb.cache_clear()

    @classmethod
    @cache
    def emb(cls, bno_threshold):
        emb = vayesta.ewf.EWF(cls.mf, bno_threshold=bno_threshold, solve_lambda=True)
        emb.iao_fragmentation()
        emb.add_all_atomic_fragments()
        emb.kernel()
        return emb

    # --- Full bath tests:

    def test_energy(self):
        emb = self.emb(-1)
        self.assertAllclose(emb.e_tot, self.cc.e_tot, rtol=0)
        self.assertAllclose(emb.e_corr, self.cc.e_corr, rtol=0)

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

    def test_global_dm1(self):
        emb = self.emb(-1)
        dm1 = emb.make_rdm1_ccsd()
        dm1_exact = self.cc.make_rdm1()
        self.assertAllclose(dm1, dm1_exact)


class TestH2Anion(TestH2):

    @classmethod
    def setUpClass(cls):
        cls.mf = testsystems.h2anion_dz.uhf()
        cls.cc = testsystems.h2anion_dz.uccsd()


if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
