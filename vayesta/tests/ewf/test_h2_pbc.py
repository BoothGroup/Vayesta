import unittest

import numpy as np

import vayesta
import vayesta.ewf
from vayesta.core.util import cache
from vayesta.tests import testsystems
from vayesta.tests.common import TestCase

class TestMP2_H2_STO3G(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.mf = testsystems.h2_sto3g_k311.rhf()
        cls.cc = testsystems.h2_sto3g_s311.rmp2()

    @classmethod
    def tearDownClass(cls):
        del cls.mf
        del cls.cc
        cls.emb.cache_clear()

    @classmethod
    @cache
    def emb(cls, bno_threshold):
        emb = vayesta.ewf.EWF(cls.mf, bno_threshold=bno_threshold, solver='MP2')
        emb.kernel()
        return emb

    # --- Full bath tests:

    def test_energy(self):
        nk = len(self.mf.kpts)
        # Full bath
        emb = self.emb(-1)
        self.assertAllclose(emb.e_corr, self.cc.e_corr/nk, rtol=0)
        self.assertAllclose(emb.e_tot, self.cc.e_tot/nk, rtol=0)
        # Finite bath
        emb = self.emb(1e-3)
        self.assertAllclose(emb.e_corr, -0.009599078822158, rtol=0)
        self.assertAllclose(emb.e_tot, -1.277732258158756, rtol=0)

    def _get_ref_t1_ao(self, t1):
        occ = self.cc._scf.mo_occ > 0
        vir = self.cc._scf.mo_occ == 0
        mo_occ = self.cc.mo_coeff[:,occ]
        mo_vir = self.cc.mo_coeff[:,vir]
        t1_ref = np.einsum('Ii,ia,Aa->IA', mo_occ, t1, mo_vir, optimize=True)
        return t1_ref

    def _get_ref_t2_ao(self, t2):
        occ = self.cc._scf.mo_occ > 0
        vir = self.cc._scf.mo_occ == 0
        mo_occ = self.cc.mo_coeff[:,occ]
        mo_vir = self.cc.mo_coeff[:,vir]
        t2_ref = np.einsum('Ii,Jj,ijab,Aa,Bb->IJAB', mo_occ, mo_occ, t2, mo_vir, mo_vir, optimize=True)
        return t2_ref

    def test_t_amplitudes(self):
        emb = self.emb(-1)
        t2 = emb.get_global_t2(ao_basis=True)
        t2_ref = self._get_ref_t2_ao(self.cc.t2)
        self.assertAllclose(t2, t2_ref)

    def test_global_dm1(self):
        emb = self.emb(-1)
        dm1 = emb.make_rdm1()
        dm1_exact = self.cc.make_rdm1()
        self.assertAllclose(dm1, dm1_exact)
        dm1 = emb._make_rdm1_ccsd_2p2l()
        self.assertAllclose(dm1, dm1_exact)
        dm1 = emb._make_rdm1_ccsd_2p2l(late_t2_sym=False)
        self.assertAllclose(dm1, dm1_exact)
        dm1 = emb._make_rdm1_ccsd_2p2l(use_sym=False)
        self.assertAllclose(dm1, dm1_exact)
        dm1 = emb._make_rdm1_ccsd_2p2l(late_t2_sym=False, use_sym=False)
        self.assertAllclose(dm1, dm1_exact)

class TestCCSD_H2_STO3G(TestMP2_H2_STO3G):

    @classmethod
    def setUpClass(cls):
        cls.mf = testsystems.h2_sto3g_k311.rhf()
        cls.cc = testsystems.h2_sto3g_s311.rccsd()

    @classmethod
    @cache
    def emb(cls, bno_threshold):
        solver_opts = dict(conv_tol=1e-10, conv_tol_normt=1e-8)
        emb = vayesta.ewf.EWF(cls.mf, bno_threshold=bno_threshold, solve_lambda=True, solver_options=solver_opts)
        emb.iao_fragmentation()
        emb.add_all_atomic_fragments()
        emb.kernel()
        return emb

    def test_energy(self):
        nk = len(self.mf.kpts)
        # Full bath
        emb = self.emb(-1)
        e_exxdiv = emb.get_exxdiv()[0]  # PySCF's gamma-point CCSD solver misses exxdiv energy correction
        self.assertAllclose(emb.e_corr, self.cc.e_corr/nk, rtol=0)
        self.assertAllclose(emb.e_tot, self.cc.e_tot/nk + e_exxdiv, rtol=0)
        # Finite bath
        emb = self.emb(1e-3)
        self.assertAllclose(emb.e_corr, -0.0153692736073979, rtol=0)
        self.assertAllclose(emb.e_tot, -1.2835024529439953, rtol=0)

    def test_t_amplitudes(self):
        emb = self.emb(-1)
        t1 = emb.get_global_t1(ao_basis=True)
        t2 = emb.get_global_t2(ao_basis=True)
        t1_ref = self._get_ref_t1_ao(self.cc.t1)
        t2_ref = self._get_ref_t2_ao(self.cc.t2)
        self.assertAllclose(t1, t1_ref)
        self.assertAllclose(t2, t2_ref)

    def test_l_amplitudes(self):
        emb = self.emb(-1)
        l1 = emb.get_global_l1(ao_basis=True)
        l2 = emb.get_global_l2(ao_basis=True)
        l1_ref = self._get_ref_t1_ao(self.cc.l1)
        l2_ref = self._get_ref_t2_ao(self.cc.l2)
        self.assertAllclose(l1, l1_ref)
        self.assertAllclose(l2, l2_ref)

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


if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
