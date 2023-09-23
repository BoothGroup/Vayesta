import pytest

import vayesta
import vayesta.ewf
from tests import systems
from tests.common import TestCase
from vayesta.core.util import cache


class TestWaterRHF(TestCase):
    system = systems.water_631g_df

    @classmethod
    def setUpClass(cls):
        cls.mf = cls.system.rhf()

    @classmethod
    def tearDownClass(cls):
        del cls.mf
        cls.emb.cache_clear()

    @classmethod
    @cache
    def emb(cls, bno_threshold, solver):
        emb = vayesta.ewf.EWF(
            cls.mf,
            bath_options=dict(bathtype="rpa", threshold=bno_threshold),
            solver=solver,
            solver_options=dict(conv_tol=1e-12),
        )
        emb.kernel()
        return emb

    def test_ccsd_rpa_1(self):
        eta = 10 ** -(1.5)
        emb = self.emb(eta, "CCSD")
        emb.kernel()
        self.assertAllclose(emb.e_tot, -76.10582744548097)

    @pytest.mark.slow
    def test_ccsd_rpa_2(self):
        eta = 1e-2
        emb = self.emb(eta, "CCSD")
        emb.kernel()
        self.assertAllclose(emb.e_tot, -76.12444492796294)
