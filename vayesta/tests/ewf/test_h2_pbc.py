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
        cls.mf = testsystems.h2_sto3g_k3.rhf()
        cls.cc = testsystems.h2_sto3g_s3.rccsd()

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
        nk = len(self.mf.kpts)
        self.assertAllclose(emb.e_corr, self.cc.e_corr/nk, rtol=0)
        e_exxdiv = emb.get_exxdiv()[0]  # PySCF's gamma-point CCSD solver misses exxdiv energy correction
        self.assertAllclose(emb.e_tot, self.cc.e_tot/nk + e_exxdiv, rtol=0)

    def _get_ref_ao_amplitudes(self, t1, t2):
        occ = self.cc._scf.mo_occ > 0
        vir = self.cc._scf.mo_occ == 0
        mo_occ = self.cc.mo_coeff[:,occ]
        mo_vir = self.cc.mo_coeff[:,vir]
        t1_ref = np.einsum('Ii,ia,Aa->IA', mo_occ, t1, mo_vir, optimize=True)
        t2_ref = np.einsum('Ii,Jj,ijab,Aa,Bb->IJAB', mo_occ, mo_occ, t2, mo_vir, mo_vir, optimize=True)
        return t1_ref, t2_ref

    def test_t_amplitudes(self):
        emb = self.emb(-1)
        t1 = emb.get_global_t1(ao_basis=True)
        t2 = emb.get_global_t2(ao_basis=True)
        t1_ref, t2_ref = self._get_ref_ao_amplitudes(self.cc.t1, self.cc.t2)
        self.assertAllclose(t1, t1_ref)
        self.assertAllclose(t2, t2_ref)

    def test_l_amplitudes(self):
        emb = self.emb(-1)
        l1 = emb.get_global_l1(ao_basis=True)
        l2 = emb.get_global_l2(ao_basis=True)
        l1_ref, l2_ref = self._get_ref_ao_amplitudes(self.cc.l1, self.cc.l2)
        self.assertAllclose(l1, l1_ref)
        self.assertAllclose(l2, l2_ref)

    #def test_cluster_dm1(self):
    #    emb = self.emb(-1)
    #    t = np.dot(self.mf.get_ovlp(), self.mf.mo_coeff)
    #    dm1_exact = self.cc.make_rdm1(ao_repr=True)
    #    for x in emb.fragments:
    #        dm1 = x.results.wf.make_rdm1(ao_basis=True)
    #        self.assertAllclose(dm1, dm1_exact)

    #def test_cluster_dm2(self):
    #    emb = self.emb(-1)
    #    t = np.dot(self.mf.get_ovlp(), self.mf.mo_coeff)
    #    dm2_exact = self.cc.make_rdm2(ao_repr=True)
    #    for x in emb.fragments:
    #        dm2 = x.results.wf.make_rdm2(ao_basis=True)
    #        self.assertAllclose(dm2, dm2_exact)

    #def test_global_dm1(self):
    #    emb = self.emb(-1)
    #    dm1 = emb.make_rdm1_ccsd()
    #    dm1_exact = self.cc.make_rdm1()
    #    self.assertAllclose(dm1, dm1_exact)

    def test_global_dm1_symmetry(self):
        emb = self.emb(-1)
        dm1 = emb.make_rdm1_ccsd_2p2l()
        dm1_sym = emb.make_rdm1_ccsd_2p2l(use_sym=True)
        self.assertAllclose(dm1[:2,:2], dm1_sym[:2,:2])
        self.assertAllclose(dm1, dm1_sym)

#class TestH2Anion(TestH2):
#
#    @classmethod
#    def setUpClass(cls):
#        cls.mf = testsystems.h2anion_dz.uhf()
#        cls.cc = testsystems.h2anion_dz.uccsd()
#
## --- MP2
#
#class TestH2_MP2(TestH2):
#
#    @classmethod
#    def setUpClass(cls):
#        cls.mf = testsystems.h2_dz.rhf()
#        cls.cc = testsystems.h2_dz.rmp2()
#
#    @classmethod
#    @cache
#    def emb(cls, bno_threshold):
#        emb = vayesta.ewf.EWF(cls.mf, solver='MP2', bno_threshold=bno_threshold)
#        emb.iao_fragmentation()
#        emb.add_all_atomic_fragments()
#        emb.kernel()
#        return emb
#
#    def test_t_amplitudes(self):
#        emb = self.emb(-1)
#        t2 = emb.get_global_t2()
#        self.assertAllclose(t2, self.cc.t2)
#
#    def test_l_amplitudes(self):
#        pass
#
#    def test_cluster_dm1(self):
#        pass
#
#    def test_cluster_dm2(self):
#        pass
#
#    def test_global_dm1(self):
#        emb = self.emb(-1)
#        dm1 = emb.make_rdm1_mp2()
#        dm1_exact = self.cc.make_rdm1()
#        self.assertAllclose(dm1, dm1_exact)
#
## --- FCI
#
#class TestH2_FCI(TestCase):
#
#    @classmethod
#    def setUpClass(cls):
#        cls.mf = testsystems.h2_dz.rhf()
#        cls.fci = testsystems.h2_dz.rfci()
#
#    @classmethod
#    def tearDownClass(cls):
#        del cls.mf
#        del cls.fci
#        cls.emb.cache_clear()
#
#    @classmethod
#    @cache
#    def emb(cls, bno_threshold):
#        emb = vayesta.ewf.EWF(cls.mf, solver='FCI', bno_threshold=bno_threshold)
#        emb.iao_fragmentation()
#        emb.add_all_atomic_fragments()
#        emb.kernel()
#        return emb
#
#    def test_energy(self):
#        emb = self.emb(-1)
#        self.assertAllclose(emb.e_tot, self.fci.e_tot, rtol=0)
#
#class TestH2Anion_FCI(TestCase):
#
#    @classmethod
#    def setUpClass(cls):
#        cls.mf = testsystems.h2anion_dz.uhf()
#        cls.fci = testsystems.h2anion_dz.ufci()


if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
