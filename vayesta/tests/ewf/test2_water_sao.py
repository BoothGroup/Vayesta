import numpy as np
import unittest
import test_h2
import vayesta
from vayesta.core.util import cache
from vayesta.tests import testsystems
from vayesta.tests.common import TestCase

#class Test_MP2(test_h2.Test_MP2):
#
#    @classmethod
#    def setUpClass(cls):
#        cls.mf = testsystems.water_631g.rhf()
#        cls.cc = testsystems.water_631g.rmp2()

class Test_CCSD(test_h2.Test_CCSD):

    @classmethod
    def setUpClass(cls):
        cls.mf = testsystems.water_631g.rhf()
        cls.cc = testsystems.water_631g.rccsd()

    @classmethod
    @cache
    def emb(cls, bno_threshold):
        emb = vayesta.ewf.EWF(cls.mf, bno_threshold=bno_threshold, solve_lambda=True)
        with emb.sao_fragmentation() as f:
            f.add_all_atomic_fragments()
        emb.kernel()
        return emb

    def test_dmet_energy(self):
        """Tests DMET energy."""
        emb = self.emb(-1)
        etot_dmet = emb.get_dmet_energy()
        self.assertAllclose(etot_dmet, self.cc.e_tot, rtol=0)
        etot_dmet = emb.get_dmet_energy(version=2)
        self.assertAllclose(etot_dmet, self.cc.e_tot, rtol=0)
        etot_dmet = emb.get_dmet_energy(version=2, approx_cumulant=False)
        self.assertAllclose(etot_dmet, self.cc.e_tot, rtol=0)

#class Test_UMP2(test_h2.Test_UMP2):
#
#    @classmethod
#    def setUpClass(cls):
#        cls.mf = testsystems.water_cation_631g.uhf()
#        cls.cc = testsystems.water_cation_631g.ump2()

class Test_UCCSD(Test_CCSD, test_h2.Test_UCCSD):

    @classmethod
    def setUpClass(cls):
        cls.mf = testsystems.water_cation_631g.uhf()
        cls.cc = testsystems.water_cation_631g.uccsd()

class Test_RHF_vs_UHF(TestCase):

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

    @classmethod
    @cache
    def remb(cls, bno_threshold):
        solver_opts = dict(conv_tol= 1e-10, conv_tol_normt=1e-8)
        emb = vayesta.ewf.EWF(cls.rhf, bno_threshold=bno_threshold, solve_lambda=True,
                solver_options=solver_opts)
        with emb.sao_fragmentation() as f:
            f.add_all_atomic_fragments()
        emb.kernel()
        return emb

    @classmethod
    @cache
    def uemb(cls, bno_threshold):
        solver_opts = dict(conv_tol= 1e-10, conv_tol_normt=1e-8)
        emb = vayesta.ewf.EWF(cls.uhf, bno_threshold=bno_threshold, solve_lambda=True,
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

    def test_dmet_energy(self):
        """Tests DMET energy."""
        remb = self.remb(-1)
        uemb = self.uemb(-1)

        e_ref = self.rcc.e_tot
        e_rdmet = remb.get_dmet_energy()
        e_udmet = uemb.get_dmet_energy()
        self.assertAllclose(e_rdmet, e_ref)
        self.assertAllclose(e_udmet, e_ref)
        e_rdmet = remb.get_dmet_energy(version=2)
        e_udmet = uemb.get_dmet_energy(version=2)
        self.assertAllclose(e_rdmet, e_ref)
        self.assertAllclose(e_udmet, e_ref)
        e_rdmet = remb.get_dmet_energy(version=2, approx_cumulant=False)
        e_udmet = uemb.get_dmet_energy(version=2, approx_cumulant=False)
        self.assertAllclose(e_rdmet, e_ref)
        self.assertAllclose(e_udmet, e_ref)

    def test_dmet_energy_finite_bath(self):
        remb = self.remb(np.inf)
        uemb = self.uemb(np.inf)

        e_rdmet = remb.get_dmet_energy()
        e_udmet = uemb.get_dmet_energy()
        self.assertAllclose(e_rdmet, e_udmet)
        e_rdmet = remb.get_dmet_energy(version=2)
        e_udmet = uemb.get_dmet_energy(version=2)
        self.assertAllclose(e_rdmet, e_udmet)
        e_rdmet = remb.get_dmet_energy(version=2, approx_cumulant=False)
        e_udmet = uemb.get_dmet_energy(version=2, approx_cumulant=False)
        self.assertAllclose(e_rdmet, e_udmet)


if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
