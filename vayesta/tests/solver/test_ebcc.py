import pytest

import pyscf
import pyscf.cc

import vayesta
import vayesta.ewf

from vayesta.tests.common import TestCase
from vayesta.tests import testsystems

# Note that ebCC currently doesn't support density fitting, so we'll just test non-DF results.

@pytest.mark.fast
class TestCCSD(TestCase):
    try:
        import ebcc
    except ImportError:
        pytest.skip("Requires cvxpy")

    def _test(self, key):
        mf = getattr(getattr(testsystems, key[0]), key[1])()

        emb = vayesta.ewf.EWF(mf, solver=f'EB{key[2]}', bath_options=dict(bathtype='full'),
                              solver_options=dict(solve_lambda=False))
        emb.kernel()
        import ebcc

        cc = ebcc.EBCC(mf, ansatz=key[-1])
        cc.kernel()

        self.assertAlmostEqual(emb.e_corr, cc.e_corr)
        self.assertAlmostEqual(emb.e_tot, cc.e_tot)

    def test_rccsd_h2(self):
        return self._test(('h2_ccpvdz', 'rhf', 'CCSD'))

    def test_rccsd_water_sto3g(self):
        return self._test(('water_sto3g', 'rhf', 'CCSD'))

    def test_uccsd_water_sto3g(self):
        return self._test(('water_sto3g', 'uhf', 'CCSD'))

    def test_uccsd_water_cation_sto3g(self):
        return self._test(('water_cation_sto3g', 'uhf', 'CCSD'))

    def test_rccsdt_water_sto3g(self):
        return self._test(('water_sto3g', 'rhf', 'CCSDT'))

    def test_uccsdt_water_cation_sto3g(self):
        return self._test(('water_cation_sto3g', 'uhf', 'CCSDT'))

    def test_rccsdtprime_water_sto3g(self):
        return self._test(('water_sto3g', 'rhf', "CCSDt'", 'CCSDT'))

    def test_uccsdtprime_water_sto3g(self):
        return self._test(('water_sto3g', 'uhf', "CCSDt'", 'CCSDT'))
