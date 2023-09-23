import pytest
import unittest

import pyscf
import pyscf.fci

import vayesta
import vayesta.ewf
from tests import testsystems


@pytest.mark.fast
class TestSolvers(unittest.TestCase):
    def _test(self, key, ss=None, places=8):
        mf = getattr(getattr(testsystems, key[0]), key[1])()

        emb = vayesta.ewf.EWF(mf, solver="FCI", bath_options=dict(bathtype="full"), solver_options={"conv_tol": 1e-12})
        emb.kernel()

        fci = pyscf.fci.FCI(mf)
        fci.threads = 1
        fci.conv_tol = 1e-12
        if ss is not None:
            fci = pyscf.fci.addons.fix_spin_(fci, ss=ss)
        e_fci, ci = fci.kernel()
        fci.e_corr = fci.e_tot - mf.e_tot

        self.assertAlmostEqual(emb.e_corr, fci.e_corr, places=places)
        self.assertAlmostEqual(emb.e_tot, fci.e_tot, places=places)

    # TODO: Why inaccurate?

    def test_rfci_h2(self):
        return self._test(("h2_ccpvdz", "rhf"), ss=0, places=4)

    def test_rfci_h2_df(self):
        return self._test(("h2_ccpvdz_df", "rhf"), ss=0, places=4)

    def test_ufci_h3(self):
        return self._test(("h3_ccpvdz", "uhf"))

    def test_ufci_h3_df(self):
        return self._test(("h3_ccpvdz_df", "uhf"), places=4)


if __name__ == "__main__":
    print("Running %s" % __file__)
    unittest.main()
