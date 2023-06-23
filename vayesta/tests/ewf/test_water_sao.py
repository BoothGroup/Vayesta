import pytest
import unittest
import numpy as np
import vayesta
from vayesta.core.util import cache
from vayesta.tests.ewf import test_h2
from vayesta.tests import testsystems
from vayesta.tests.common import TestCase

class Test_MP2(test_h2.Test_MP2):

    @classmethod
    def setUpClass(cls):
        cls.mf = testsystems.water_631g.rhf()
        cls.cc = testsystems.water_631g.rmp2()

    @classmethod
    @cache
    def emb(cls, bno_threshold):
        emb = vayesta.ewf.EWF(cls.mf, solver='MP2', bath_options=dict(threshold=bno_threshold))
        with emb.sao_fragmentation() as f:
            f.add_all_atomic_fragments()
        emb.kernel()
        return emb

    def test_dm1_demo(self):
        emb = self.emb(-1)
        dm1_exact = self.cc.make_rdm1(ao_repr=True)
        dm1 = emb.make_rdm1_demo(ao_basis=True)
        self.assertAllclose(dm1, dm1_exact)

    def test_dm2_demo(self):
        emb = self.emb(-1)
        dm1_exact = self.cc.make_rdm1(ao_repr=True)
        dm1 = emb.make_rdm1_demo(ao_basis=True)
        self.assertAllclose(dm1, dm1_exact)

    def test_dmet_energy_full_bath(self):
        """Tests DMET energy."""
        emb = self.emb(-1)
        etot_dmet = emb.get_dmet_energy()
        self.assertAllclose(etot_dmet, self.cc.e_tot, rtol=0)
        etot_dmet = emb.get_dmet_energy(part_cumulant=False)
        self.assertAllclose(etot_dmet, self.cc.e_tot, rtol=0)
        # Not implemented:
        #etot_dmet = emb.get_dmet_energy(approx_cumulant=False)
        #self.assertAllclose(etot_dmet, self.cc.e_tot, rtol=0)

    def test_dmet_energy_dmet_bath(self):
        emb = self.emb(np.inf)
        self.assertAllclose(emb.get_dmet_energy(), -76.13138293069929, rtol=0)
        self.assertAllclose(emb.get_dmet_energy(part_cumulant=False), -76.14003024590309, rtol=0)


@pytest.mark.slow
class Test_CCSD(test_h2.Test_CCSD):

    @classmethod
    def setUpClass(cls):
        cls.mf = testsystems.water_631g.rhf()
        cls.cc = testsystems.water_631g.rccsd()

    @classmethod
    @cache
    def emb(cls, bno_threshold):
        solver_opts = dict(conv_tol= 1e-10, conv_tol_normt=1e-8, solve_lambda=True)
        emb = vayesta.ewf.EWF(cls.mf, bath_options=dict(threshold=bno_threshold),
                              solver_options=solver_opts)
        with emb.sao_fragmentation() as f:
            f.add_all_atomic_fragments()
        emb.kernel()
        return emb

    def test_dmet_energy_full_bath(self):
        """Tests DMET energy."""
        emb = self.emb(-1)
        etot_dmet = emb.get_dmet_energy()
        self.assertAllclose(etot_dmet, self.cc.e_tot, rtol=0)
        etot_dmet = emb.get_dmet_energy(part_cumulant=False)
        self.assertAllclose(etot_dmet, self.cc.e_tot, rtol=0)
        etot_dmet = emb.get_dmet_energy(approx_cumulant=False)
        self.assertAllclose(etot_dmet, self.cc.e_tot, rtol=0)

    def test_dmet_energy_dmet_bath(self):
        emb = self.emb(np.inf)
        self.assertAllclose(emb.get_dmet_energy(), -76.12825899582526, rtol=0)
        self.assertAllclose(emb.get_dmet_energy(part_cumulant=False), -76.16067576457522, rtol=0)

    def test_dm1_demo(self):
        emb = self.emb(-1)
        dm1_exact = self.cc.make_rdm1(ao_repr=True)
        dm1 = emb.make_rdm1_demo(ao_basis=True)
        self.assertAllclose(dm1, dm1_exact)

    def test_dm2_demo(self):
        emb = self.emb(-1)
        dm1_exact = self.cc.make_rdm1(ao_repr=True)
        dm1 = emb.make_rdm1_demo(ao_basis=True)
        self.assertAllclose(dm1, dm1_exact)


class Test_UMP2(test_h2.Test_UMP2):

    @classmethod
    def setUpClass(cls):
        cls.mf = testsystems.water_cation_631g.uhf()
        cls.cc = testsystems.water_cation_631g.ump2()

    @classmethod
    @cache
    def emb(cls, bno_threshold):
        emb = vayesta.ewf.EWF(cls.mf, solver='MP2', bath_options=dict(threshold=bno_threshold))
        with emb.sao_fragmentation() as f:
            f.add_all_atomic_fragments()
        emb.kernel()
        return emb


@pytest.mark.slow
class Test_UCCSD(Test_CCSD, test_h2.Test_UCCSD):

    @classmethod
    def setUpClass(cls):
        cls.mf = testsystems.water_cation_631g.uhf()
        cls.cc = testsystems.water_cation_631g.uccsd()

    def test_dmet_energy_dmet_bath(self):
        emb = self.emb(np.inf)
        self.assertAllclose(emb.get_dmet_energy(), -75.69175122675338, rtol=0)
        self.assertAllclose(emb.get_dmet_energy(part_cumulant=False), -75.69526013218531, rtol=0)


@pytest.mark.slow
class Test_RCCSD_vs_UCCSD(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.rhf = testsystems.water_631g.rhf()
        cls.uhf = testsystems.water_631g.uhf()
        cls.rcc = testsystems.water_631g.rccsd()
        cls.ucc = testsystems.water_631g.uccsd()

    @classmethod
    def tearDownClass(cls):
        del cls.rhf
        del cls.uhf
        del cls.rcc
        del cls.ucc
        cls.remb.cache_clear()
        cls.uemb.cache_clear()

    @classmethod
    @cache
    def remb(cls, bno_threshold):
        solver_opts = dict(conv_tol= 1e-10, conv_tol_normt=1e-8, solve_lambda=True)
        emb = vayesta.ewf.EWF(cls.rhf, bath_options=dict(threshold=bno_threshold),
                              solver_options=solver_opts)
        with emb.sao_fragmentation() as f:
            f.add_all_atomic_fragments()
        emb.kernel()
        return emb

    @classmethod
    @cache
    def uemb(cls, bno_threshold):
        solver_opts = dict(conv_tol= 1e-10, conv_tol_normt=1e-8, solve_lambda=True)
        emb = vayesta.ewf.EWF(cls.uhf, bath_options=dict(threshold=bno_threshold),
                              solver_options=solver_opts)
        with emb.sao_fragmentation() as f:
            f.add_all_atomic_fragments()
        emb.kernel()
        return emb

    def test_energy(self):
        """Tests energy."""
        remb = self.remb(-1)
        uemb = self.uemb(-1)

        e_ref = self.rcc.e_tot
        self.assertAllclose(remb.e_tot, e_ref)
        self.assertAllclose(uemb.e_tot, e_ref)

    def test_dm1_global_wf(self):
        rdm1 = self.remb(-1)._make_rdm1_ccsd_global_wf(ao_basis=True)
        udm1 = np.sum(self.uemb(-1)._make_rdm1_ccsd_global_wf(ao_basis=True), axis=0)
        self.assertAllclose(rdm1, udm1)

    def test_dm1_demo(self):
        rdm1 = self.remb(-1).make_rdm1(ao_basis=True)
        udm1 = np.sum(self.uemb(-1).make_rdm1(ao_basis=True), axis=0)
        self.assertAllclose(rdm1, udm1)

    def test_dm2_demo(self):
        rdm2 = self.remb(-1).make_rdm2(ao_basis=True)
        udm2aa, udm2ab, udm2bb = self.uemb(-1).make_rdm2(ao_basis=True)
        udm2 = (udm2aa + udm2ab + udm2ab.transpose(2,3,0,1) + udm2bb)
        self.assertAllclose(rdm2, udm2)

    def test_dmet_energy_full_bath(self):
        """Tests DMET energy."""
        remb = self.remb(-1)
        uemb = self.uemb(-1)
        e_ref = self.rcc.e_tot
        e_rdmet = remb.get_dmet_energy()
        e_udmet = uemb.get_dmet_energy()
        self.assertAllclose(e_rdmet, e_ref)
        self.assertAllclose(e_udmet, e_ref)
        e_rdmet = remb.get_dmet_energy(part_cumulant=False)
        e_udmet = uemb.get_dmet_energy(part_cumulant=False)
        self.assertAllclose(e_rdmet, e_ref)
        self.assertAllclose(e_udmet, e_ref)
        e_rdmet = remb.get_dmet_energy(approx_cumulant=False)
        e_udmet = uemb.get_dmet_energy(approx_cumulant=False)
        self.assertAllclose(e_rdmet, e_ref)
        self.assertAllclose(e_udmet, e_ref)

    def test_dmet_energy_dmet_bath(self):
        remb = self.remb(np.inf)
        uemb = self.uemb(np.inf)
        e_rdmet = remb.get_dmet_energy()
        e_udmet = uemb.get_dmet_energy()
        self.assertAllclose(e_rdmet, e_udmet)
        e_rdmet = remb.get_dmet_energy(part_cumulant=False)
        e_udmet = uemb.get_dmet_energy(part_cumulant=False)
        self.assertAllclose(e_rdmet, e_udmet)
        e_rdmet = remb.get_dmet_energy(approx_cumulant=False)
        e_udmet = uemb.get_dmet_energy(approx_cumulant=False)
        self.assertAllclose(e_rdmet, e_udmet)


if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
