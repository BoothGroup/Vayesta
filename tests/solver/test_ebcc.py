import pytest

import vayesta
import vayesta.ewf

from tests.common import TestCase
from tests import systems

# Note that ebCC currently doesn't support density fitting, so we're just testing non-DF results here.


class TestEBCC(TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import ebcc
        except ImportError:
            pytest.skip("Requires ebcc")

    def _test(self, system, mf, ansatz):
        # Test a complete bath calculation with given ansatz reproduces full calculation.
        mymf = getattr(getattr(systems, system), mf)()

        emb = vayesta.ewf.EWF(
            mymf,
            solver=f"EB{ansatz}",
            bath_options=dict(bathtype="full"),
            solver_options=dict(solve_lambda=False, store_as_ccsd=False),
        )
        emb.kernel()
        import ebcc

        cc = ebcc.EBCC(mymf, ansatz=ansatz)
        cc.kernel()

        self.assertAlmostEqual(emb.e_corr, cc.e_corr)
        self.assertAlmostEqual(emb.e_tot, cc.e_tot)

    @pytest.mark.fast
    def test_rccsd_h2(self):
        return self._test("h2_ccpvdz", "rhf", "CCSD")

    @pytest.mark.fast
    def test_rccsd_water_sto3g(self):
        return self._test("water_sto3g", "rhf", "CCSD")

    @pytest.mark.fast
    def test_uccsd_water_sto3g(self):
        return self._test("water_sto3g", "uhf", "CCSD")

    def test_uccsd_water_cation_sto3g(self):
        return self._test("water_cation_sto3g", "uhf", "CCSD")

    @pytest.mark.fast
    def test_rccsdt_water_sto3g(self):
        return self._test("water_sto3g", "rhf", "CCSDT")

    @pytest.mark.slow
    def test_uccsdt_water_cation_sto3g(self):
        return self._test("water_cation_sto3g", "uhf", "CCSDT")


class TestEBCCActSpace(TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import ebcc
        except ImportError:
            pytest.skip("Requires ebcc")

    def _test(self, system, mf, actansatz, fullansatz, bathtype="dmet", setcas=False):
        # Test that active space calculation with complete active space reproduces equivalent calculation using higher-
        # level approach of active space. This defaults to a DMET bath space.
        mymf = getattr(getattr(systems, system), mf)()

        embfull = vayesta.ewf.EWF(
            mymf,
            solver=f"EB{fullansatz}",
            bath_options=dict(bathtype=bathtype),
            solver_options=dict(solve_lambda=False),
        )
        embfull.kernel()

        embact = vayesta.ewf.EWF(
            mymf, solver=f"EB{actansatz}", bath_options=dict(bathtype=bathtype), solver_options=dict(solve_lambda=False)
        )
        if setcas:
            # Set up fragmentation, then set CAS to complete cluster space in previous calculation.
            with embact.iao_fragmentation() as f:
                f.add_all_atomic_fragments()

            for ffull, fact in zip(embfull.loop(), embact.loop()):
                fact.set_cas(c_occ=ffull.cluster.c_active_occ, c_vir=ffull.cluster.c_active_vir)
        embact.kernel()

        self.assertAlmostEqual(embact.e_corr, embfull.e_corr)
        self.assertAlmostEqual(embact.e_tot, embfull.e_tot)

    @pytest.mark.fast
    def test_rccsdtprime_water_sto3g_dmet(self):
        return self._test("water_sto3g", "rhf", "CCSDt'", "CCSDT", bathtype="dmet", setcas=False)

    def test_uccsdtprime_water_sto3g_dmet(self):
        return self._test("water_sto3g", "uhf", "CCSDt'", "CCSDT", bathtype="dmet", setcas=False)

    @pytest.mark.slow
    def test_rccsdtprime_h4_sto3g_setcas_full(self):
        return self._test("h4_sto3g", "rhf", "CCSDt'", "CCSDT", bathtype="full", setcas=True)

    def test_uccsdtprime_h3_sto3g_setcas_full(self):
        return self._test("h3_sto3g", "uhf", "CCSDt'", "CCSDT", bathtype="full", setcas=True)
