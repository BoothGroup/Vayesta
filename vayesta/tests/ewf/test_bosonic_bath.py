import numpy as np
import pytest

from vayesta import ewf
from vayesta.tests.common import TestCase
from vayesta.tests import testsystems


class BosonicBathTests(TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import ebcc
        except ImportError:
            pytest.skip("Requires ebcc")

    def _get_emb(self, mf, solver, target_orbs, local_projection, threshold):
        emb = ewf.EWF(
            mf,
            solver=solver,
            bath_options=dict(bathtype="mp2", threshold=threshold, project_dmet_order=1, project_dmet_mode="full"),
            bosonic_bath_options=dict(
                bathtype="rpa", target_orbitals=target_orbs, local_projection=local_projection, threshold=threshold
            ),
        )
        emb.kernel()
        return emb.e_tot

    def test_water_ccpvdz_CCSDS11_full_fragment(self):
        mf = testsystems.water_ccpvdz_df.rhf()
        e = self._get_emb(mf, "CCSD-S-1-1", "full", "fragment", 1e-3)
        self.assertAlmostEqual(e, -76.16807860786547)

    def test_water_ccpvdz_CCSDSD11_dmet_fragment(self):
        mf = testsystems.water_ccpvdz_df.rhf()
        e = self._get_emb(mf, "CCSD-SD-1-1", "dmet", "fragment", 1e-4)
        self.assertAlmostEqual(e, -76.20829873999101)

    def test_water_ccpvdz_CCSDS11_full_noproj(self):
        mf = testsystems.water_ccpvdz_df.rhf()
        e = self._get_emb(mf, "CCSD-S-1-1", "full", None, 1e-5)
        self.assertAlmostEqual(e, -76.21937784813458)

    def test_water_ccpvdz_CCSDS11_dmet_noproj(self):
        mf = testsystems.water_ccpvdz_df.rhf()
        e = self._get_emb(mf, "CCSD-S-1-1", "dmet", None, 1e-6)
        self.assertAlmostEqual(e, -76.23158153460919)
