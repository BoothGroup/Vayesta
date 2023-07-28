import unittest
import numpy as np
import vayesta
import vayesta.ewf
from vayesta.core.util import cache
from vayesta.tests import testsystems
from vayesta.tests.common import TestCase


class TestWaterRHF(TestCase):
    system = testsystems.water_631g_df
    e_ref = {"mrpa":-1.123779303361342, "crpa":-1.1237769151822752}

    @classmethod
    def setUpClass(cls):
        cls.mf = cls.system.rhf()

    @classmethod
    def tearDownClass(cls):
        del cls.mf
        cls.emb.cache_clear()

    @classmethod
    @cache
    def emb(cls, bno_threshold, solver, screening):
        emb = vayesta.ewf.EWF(cls.mf, bath_options=dict(bathtype="rpa", threshold=bno_threshold), solver=solver,
                screening=screening, solver_options=dict(conv_tol=1e-12))
        emb.kernel()
        return emb

    def test_ccsd_dmet(self):
        emb = self.emb(np.inf, 'CCSD', None)
        emb.kernel()
        self.assertAllclose(emb.e_tot, self.e_ref['mrpa'])

    def test_ccsd_001(self):
        emb = self.emb(np.inf, 'CCSD', None)
        emb.kernel()
        self.assertAllclose(emb.e_tot, self.e_ref['mrpa'])

