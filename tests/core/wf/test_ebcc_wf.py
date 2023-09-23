import pytest

import vayesta
import vayesta.ewf

from tests.common import TestCase
from tests import systems


class TestEBCCWavefunctions(TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import ebcc
        except ImportError:
            pytest.skip("Requires ebcc")

    def _test(self, system, mf, ansatz):
        assert mf in ["rhf", "uhf"]
        # Test a complete bath calculation with given ansatz reproduces full calculation.
        mymf = getattr(getattr(systems, system), mf)()

        emborig = vayesta.ewf.EWF(
            mymf,
            solver=f"EB{ansatz}",
            bath_options=dict(bathtype="dmet"),
            solver_options=dict(solve_lambda=True, store_as_ccsd=True),
        )
        with emborig.iao_fragmentation() as f:
            f.add_atomic_fragment([0])
        emborig.kernel()

        f = emborig.fragments[0]

        wf1 = f._results.wf

        f.opts.solver_options["store_as_ccsd"] = False
        f._results = None
        f._hamil = None
        f.kernel()
        wf2 = f._results.wf

        if mf == "rhf":
            attributes = ["t1", "t2", "l1", "l2"]
        elif mf == "uhf":
            attributes = ["t1a", "t1b", "t2aa", "t2ab", "t2bb", "l1a", "l1b", "l2aa", "l2ab", "l2bb"]

        for attr in attributes:
            self.assertAllclose(getattr(wf1, attr), getattr(wf2, attr))

        if ansatz == "CCSD":
            # Should only be the same for CCSD.
            if mf == "rhf":
                self.assertAllclose(wf1.make_rdm1(), wf2.make_rdm1())
            elif mf == "uhf":
                dmorig = wf1.make_rdm1()
                dmnew = wf2.make_rdm1()
                self.assertAllclose(dmorig[0], dmnew[0])
                self.assertAllclose(dmorig[1], dmnew[1])

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
