import unittest
import numpy as np
import vayesta
import vayesta.ewf
from vayesta.core.util import cache
from vayesta.tests import testsystems
from vayesta.tests.common import TestCase


class TestTwoElectron(TestCase):
    system = testsystems.h2_ccpvdz_df
    e_ref = {"mrpa": -1.123779303361342, "crpa": -1.1237769151822752}

    @classmethod
    def setUpClass(cls):
        cls.mf = cls.system.uhf()

    @classmethod
    def tearDownClass(cls):
        del cls.mf
        cls.emb.cache_clear()

    @classmethod
    @cache
    def emb(cls, bno_threshold, solver, screening):
        emb = vayesta.ewf.EWF(
            cls.mf,
            bath_options=dict(threshold=bno_threshold),
            solver=solver,
            screening=screening,
            solver_options=dict(conv_tol=1e-12),
        )
        emb.kernel()
        return emb

    def test_ccsd_mrpa(self):
        emb = self.emb(np.inf, "CCSD", "mrpa")
        emb.kernel()
        self.assertAllclose(emb.e_tot, self.e_ref["mrpa"])

    def test_fci_mrpa(self):
        emb = self.emb(np.inf, "FCI", "mrpa")
        emb.kernel()
        self.assertAllclose(emb.e_tot, self.e_ref["mrpa"])

    def test_ccsd_crpa(self):
        emb = self.emb(np.inf, "CCSD", "crpa")
        emb.kernel()
        self.assertAllclose(emb.e_tot, self.e_ref["crpa"])

    def test_fci_crpa(self):
        emb = self.emb(np.inf, "FCI", "crpa")
        emb.kernel()
        self.assertAllclose(emb.e_tot, self.e_ref["crpa"])


class TestTwoHole(TestTwoElectron):
    system = testsystems.f2_sto6g_df
    e_ref = {"mrpa": -197.84155758368854, "crpa": -197.83928243962046}


class TestTwoHoleRHF(TestTwoElectron):
    @classmethod
    def setUpClass(cls):
        cls.mf = cls.system.rhf()


class TestMatchClusterFock(TestCase):
    """RHF and UHF references have to give the same screened cluster Hamiltonians, also for match_cluster_fock=True."""

    system = testsystems.water_631g_df

    @classmethod
    def setUpClass(cls):
        cls.mf = {"rhf": cls.system.rhf(), "uhf": cls.system.uhf()}

    @classmethod
    def tearDownClass(cls):
        del cls.mf
        cls.emb.cache_clear()

    @classmethod
    @cache
    def emb(cls, spin, match_cluster_fock):
        emb = vayesta.ewf.EWF(
            cls.mf[spin],
            solver="CCSD",
            screening="mrpa",
            match_cluster_fock=match_cluster_fock,
            bath_options=dict(bathtype="dmet"),
            solver_options=dict(conv_tol=1e-10, conv_tol_normt=1e-8),
        )
        with emb.sao_fragmentation() as f:
            f.add_all_atomic_fragments()
        emb.kernel()
        return emb

    def test_rhf_vs_uhf(self):
        for match_cluster_fock in (False, True):
            e_rhf = self.emb("rhf", match_cluster_fock).e_corr
            e_uhf = self.emb("uhf", match_cluster_fock).e_corr
            self.assertAllclose(e_rhf, e_uhf)

    def test_match_cluster_fock_not_noop(self):
        for spin in ("rhf", "uhf"):
            e_match = self.emb(spin, True).e_corr
            e_nomatch = self.emb(spin, False).e_corr
            self.assertGreater(abs(e_match - e_nomatch), 1e-7)


if __name__ == "__main__":
    print("Running %s" % __file__)
    unittest.main()
