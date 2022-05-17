import unittest

import numpy as np

import vayesta
import vayesta.ewf
from vayesta.core.util import cache
from vayesta.tests import testsystems
from vayesta.tests.common import TestCase


class Test_Water(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.mf = testsystems.water_631g.rhf()
        cls.cc = testsystems.water_631g.rccsd()

    @classmethod
    def tearDownClass(cls):
        del cls.mf
        del cls.cc
        cls.emb.cache_clear()

    @classmethod
    @cache
    def emb(cls, bno_threshold):
        solver_opts = dict(conv_tol=1e-10, conv_tol_normt=1e-8)
        emb = vayesta.ewf.EWF(cls.mf, bno_threshold=bno_threshold, solve_lambda=True, solver_options=solver_opts)
        emb.sao_fragmentation()
        emb.add_all_atomic_fragments()
        emb.kernel()
        return emb

    def test_make_rdm1_demo(self):
        """Test full bath"""
        dm1_ref = self.cc.make_rdm1()
        emb = self.emb(-1)
        dm1 = emb.make_rdm1_demo()
        self.assertAllclose(dm1, dm1_ref)

    def test_make_rdm2_demo(self):
        """Test full bath"""
        dm2_ref = self.cc.make_rdm2()
        emb = self.emb(-1)
        dm2 = emb.make_rdm2_demo()
        self.assertAllclose(dm2, dm2_ref)

class Test_WaterCation(Test_Water):

    @classmethod
    def setUpClass(cls):
        cls.mf = testsystems.water_cation_631g.uhf()
        cls.cc = testsystems.water_cation_631g.uccsd()

    def test_make_rdm1_demo(self):
        """Test full bath"""
        dm1_ref = self.cc.make_rdm1()
        emb = self.emb(-1)
        dm1 = emb.make_rdm1_demo()
        for s in range(2):
            self.assertAllclose(dm1[s], dm1_ref[s])

    def test_make_rdm2_demo(self):
        """Test full bath"""
        dm2_ref = self.cc.make_rdm2()
        emb = self.emb(-1)

        dm2 = emb.make_rdm2_demo()
        for s in range(3):
            self.assertAllclose(dm2[s], dm2_ref[s])

        dm2 = emb.make_rdm2_demo(dmet_dm2=False)
        for s in range(3):
            self.assertAllclose(dm2[s], dm2_ref[s])

        dm2 = emb.make_rdm2_demo(approx_cumulant=False)
        for s in range(3):
            self.assertAllclose(dm2[s], dm2_ref[s])

class Test_H2Chain(Test_Water):

    @classmethod
    def setUpClass(cls):
        cls.mf = testsystems.h2_sto3g_s311.rhf()
        cls.kmf = testsystems.h2_sto3g_k311.rhf()
        cls.cc = testsystems.h2_sto3g_s311.rccsd()

    @classmethod
    def tearDownClass(cls):
        super(cls, Test_H2Chain).tearDownClass()
        del cls.kmf

    @classmethod
    @cache
    def emb_kpts(cls, bno_threshold):
        solver_opts = dict(conv_tol=1e-10, conv_tol_normt=1e-8)
        emb = vayesta.ewf.EWF(cls.kmf, bno_threshold=bno_threshold, solve_lambda=True, solver_options=solver_opts)
        emb.sao_fragmentation()
        emb.add_all_atomic_fragments()
        emb.kernel()
        return emb

    def test_make_rdm1_demo_kpts(self):
        """Test full bath"""
        dm1_ref = self.cc.make_rdm1(ao_repr=True)
        emb = self.emb_kpts(-1)
        dm1 = emb.make_rdm1_demo(ao_basis=True)
        self.assertAllclose(dm1, dm1_ref)

    def test_make_rdm2_demo_kpts(self):
        """Test full bath"""
        dm2_ref = self.cc.make_rdm2(ao_repr=True)
        emb = self.emb_kpts(-1)
        dm2 = emb.make_rdm2_demo(ao_basis=True)
        self.assertAllclose(dm2, dm2_ref)

if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
