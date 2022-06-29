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
        cls.mf = testsystems.h2_sto3g_k311.rhf()
        cls.cc = testsystems.h2_sto3g_s311.rmp2()
        cls.ref_values = {
                ('e_corr', 1e-3) : -0.009599078822158,
                ('e_tot', 1e-3) : -1.277732258158756,
                }

    @classmethod
    def tearDownClass(cls):
        del cls.mf
        del cls.cc
        del cls.ref_values
        cls.emb.cache_clear()

    @classmethod
    @cache()
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
        self.assertAllclose(emb.e_corr, self.ref_values[('e_corr', 1e-3)], rtol=0)
        self.assertAllclose(emb.e_tot, self.ref_values[('e_tot', 1e-3)], rtol=0)

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

    def test_dm1(self):
        emb = self.emb(-1)
        dm1_exact = self.cc.make_rdm1(ao_repr=True)

        dm1 = emb._make_rdm1_ccsd_global_wf(ao_basis=True)
        self.assertAllclose(dm1, dm1_exact)
        dm1 = emb._make_rdm1_ccsd_global_wf(ao_basis=True, late_t2_sym=False)
        self.assertAllclose(dm1, dm1_exact)
        dm1 = emb._make_rdm1_ccsd_global_wf(ao_basis=True, use_sym=False)
        self.assertAllclose(dm1, dm1_exact)
        dm1 = emb._make_rdm1_ccsd_global_wf(ao_basis=True, late_t2_sym=False, use_sym=False)
        self.assertAllclose(dm1, dm1_exact)


class Test_CCSD(Test_MP2):

    @classmethod
    def setUpClass(cls):
        cls.mf = testsystems.h2_sto3g_k311.rhf()
        cls.cc = testsystems.h2_sto3g_s311.rccsd()
        cls.ref_values = {
                ('e_corr', 1e-3) : -0.0153692736073979,
                ('e_tot', 1e-3) : -1.2835024529439953,
                }

    @classmethod
    @cache()
    def emb(cls, bno_threshold):
        solver_opts = dict(conv_tol=1e-10, conv_tol_normt=1e-8)
        emb = vayesta.ewf.EWF(cls.mf, bno_threshold=bno_threshold, solve_lambda=True, solver_options=solver_opts)
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
        self.assertAllclose(emb.e_corr, self.ref_values[('e_corr', 1e-3)], rtol=0)
        self.assertAllclose(emb.e_tot, self.ref_values[('e_tot', 1e-3)], rtol=0)

    def test_dmet_energy(self):
        emb = self.emb(-1)
        nk = len(self.mf.kpts)
        e_exxdiv = emb.get_exxdiv()[0]  # PySCF's gamma-point CCSD solver misses exxdiv energy correction
        e_ref = self.cc.e_tot/nk + e_exxdiv

        etot_dmet = emb.get_dmet_energy()
        self.assertAllclose(etot_dmet, e_ref, rtol=0)
        etot_dmet = emb.get_dmet_energy(version=2)
        self.assertAllclose(etot_dmet, e_ref, rtol=0)
        # TODO: Why is this broken?!
        #etot_dmet = emb.get_dmet_energy(version=2, approx_cumulant=False)
        #self.assertAllclose(etot_dmet, e_ref, rtol=0)

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

    def test_dm1_demo(self):
        emb = self.emb(-1)
        dm1_exact = self.cc.make_rdm1(ao_repr=True)
        dm1 = emb.make_rdm1_demo(ao_basis=True)
        self.assertAllclose(dm1, dm1_exact)

    def test_dm2_demo(self):
        emb = self.emb(-1)
        dm2_exact = self.cc.make_rdm2(ao_repr=True)
        dm2 = emb.make_rdm2_demo(ao_basis=True)
        self.assertAllclose(dm2, dm2_exact)

# --- Unrestricted

class Test_UMP2(Test_MP2):

    @classmethod
    def setUpClass(cls):
        cls.mf = testsystems.h3_sto3g_k311.uhf()
        cls.cc = testsystems.h3_sto3g_s311.ump2()
        cls.ref_values = {
                ('e_corr', 1e-3) : -0.00820754179397088,
                ('e_tot', 1e-3) : -1.716742435416252
                }

    def _get_ref_t1_ao(self, t1):
        t1a, t1b = t1
        occa = self.cc._scf.mo_occ[0] > 0
        occb = self.cc._scf.mo_occ[1] > 0
        vira = self.cc._scf.mo_occ[0] == 0
        virb = self.cc._scf.mo_occ[1] == 0
        mo_occ_a = self.cc.mo_coeff[0][:,occa]
        mo_occ_b = self.cc.mo_coeff[1][:,occb]
        mo_vir_a = self.cc.mo_coeff[0][:,vira]
        mo_vir_b = self.cc.mo_coeff[1][:,virb]
        t1a_ref = np.einsum('Ii,ia,Aa->IA', mo_occ_a, t1a, mo_vir_a, optimize=True)
        t1b_ref = np.einsum('Ii,ia,Aa->IA', mo_occ_b, t1b, mo_vir_b, optimize=True)
        return (t1a_ref, t1b_ref)

    def _get_ref_t2_ao(self, t2):
        t2aa, t2ab, t2bb = t2
        occa = self.cc._scf.mo_occ[0] > 0
        occb = self.cc._scf.mo_occ[1] > 0
        vira = self.cc._scf.mo_occ[0] == 0
        virb = self.cc._scf.mo_occ[1] == 0
        mo_occ_a = self.cc.mo_coeff[0][:,occa]
        mo_occ_b = self.cc.mo_coeff[1][:,occb]
        mo_vir_a = self.cc.mo_coeff[0][:,vira]
        mo_vir_b = self.cc.mo_coeff[1][:,virb]
        t2aa_ref = np.einsum('Ii,Jj,ijab,Aa,Bb->IJAB', mo_occ_a, mo_occ_a, t2aa, mo_vir_a, mo_vir_a, optimize=True)
        t2ab_ref = np.einsum('Ii,Jj,ijab,Aa,Bb->IJAB', mo_occ_a, mo_occ_b, t2ab, mo_vir_a, mo_vir_b, optimize=True)
        t2bb_ref = np.einsum('Ii,Jj,ijab,Aa,Bb->IJAB', mo_occ_b, mo_occ_b, t2bb, mo_vir_b, mo_vir_b, optimize=True)
        return (t2aa_ref, t2ab_ref, t2bb_ref)

    # Not implemented:

    def test_dm1_2p2l(self):
        pass

    def test_dm1_demo(self):
        pass

    def test_dm2_demo(self):
        pass

class Test_UCCSD(Test_CCSD):

    @classmethod
    def setUpClass(cls):
        cls.mf = testsystems.h3_sto3g_k311.uhf()
        cls.cc = testsystems.h3_sto3g_s311.uccsd()
        cls.ref_values = {
                ('e_corr', 1e-3) : -0.01654717440912164,
                ('e_tot', 1e-3) : -1.7250820680314027,
                }

    def _get_ref_t1_ao(self, t1):
        t1a, t1b = t1
        occa = self.cc._scf.mo_occ[0] > 0
        occb = self.cc._scf.mo_occ[1] > 0
        vira = self.cc._scf.mo_occ[0] == 0
        virb = self.cc._scf.mo_occ[1] == 0
        mo_occ_a = self.cc.mo_coeff[0][:,occa]
        mo_occ_b = self.cc.mo_coeff[1][:,occb]
        mo_vir_a = self.cc.mo_coeff[0][:,vira]
        mo_vir_b = self.cc.mo_coeff[1][:,virb]
        t1a_ref = np.einsum('Ii,ia,Aa->IA', mo_occ_a, t1a, mo_vir_a, optimize=True)
        t1b_ref = np.einsum('Ii,ia,Aa->IA', mo_occ_b, t1b, mo_vir_b, optimize=True)
        return (t1a_ref, t1b_ref)

    def _get_ref_t2_ao(self, t2):
        t2aa, t2ab, t2bb = t2
        occa = self.cc._scf.mo_occ[0] > 0
        occb = self.cc._scf.mo_occ[1] > 0
        vira = self.cc._scf.mo_occ[0] == 0
        virb = self.cc._scf.mo_occ[1] == 0
        mo_occ_a = self.cc.mo_coeff[0][:,occa]
        mo_occ_b = self.cc.mo_coeff[1][:,occb]
        mo_vir_a = self.cc.mo_coeff[0][:,vira]
        mo_vir_b = self.cc.mo_coeff[1][:,virb]
        t2aa_ref = np.einsum('Ii,Jj,ijab,Aa,Bb->IJAB', mo_occ_a, mo_occ_a, t2aa, mo_vir_a, mo_vir_a, optimize=True)
        t2ab_ref = np.einsum('Ii,Jj,ijab,Aa,Bb->IJAB', mo_occ_a, mo_occ_b, t2ab, mo_vir_a, mo_vir_b, optimize=True)
        t2bb_ref = np.einsum('Ii,Jj,ijab,Aa,Bb->IJAB', mo_occ_b, mo_occ_b, t2bb, mo_vir_b, mo_vir_b, optimize=True)
        return (t2aa_ref, t2ab_ref, t2bb_ref)

    # Not implemented:

    def test_dm1_2p2l(self):
        pass

    def test_dm2_demo(self):
        pass

# --- 2D

class Test_MP2_2D(Test_MP2):

    @classmethod
    def setUpClass(cls):
        cls.mf = testsystems.h2_sto3g_k31.rhf()
        cls.cc = testsystems.h2_sto3g_s31.rmp2()
        cls.ref_values = {
                ('e_corr', 1e-3) : -0.013767085896414821,
                ('e_tot', 1e-3) : -1.3539205678917514,
                }

class Test_CCSD_2D(Test_CCSD):

    @classmethod
    def setUpClass(cls):
        cls.mf = testsystems.h2_sto3g_k31.rhf()
        cls.cc = testsystems.h2_sto3g_s31.rccsd()
        cls.ref_values = {
                ('e_corr', 1e-3) : -0.01982005986990425,
                ('e_tot', 1e-3) : -1.3599735418652414,
                }

if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
