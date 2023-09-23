from tests.common import TestCase
from tests import testsystems
import vayesta.ewf
import pytest
import numpy as np


class QBA_RPA_Bath_Test(TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import ebcc
        except ImportError:
            pytest.skip("Requires ebcc")

    def _get_occupation(self, target_orbs, local_projection):
        rhf = testsystems.ethanol_631g_df.rhf()
        emb = vayesta.ewf.EWF(
            rhf,
            bosonic_bath_options=dict(
                bathtype="rpa", target_orbitals=target_orbs, local_projection=local_projection, threshold=1e-1
            ),
            bath_options=dict(bathtype="mp2", threshold=1e-4),
            solver="CCSD-S-1-1",
        )
        with emb.iao_fragmentation() as f:
            f.add_atomic_fragment([0])
        emb.kernel()
        f = emb.fragments[0]
        return f._boson_bath_factory.occup

    def test_target_full_project_fragment(self):
        occ = self._get_occupation("full", "fragment")
        values = np.array([np.amin(occ), np.amax(occ), np.mean(occ)])
        known_values = np.array([-1.23436133e-19, 3.65521662e-03, 5.97968938e-05])
        self.assertAllclose(values, known_values)

    def test_target_dmet_project_fragment(self):
        occ = self._get_occupation("dmet", "fragment")
        values = np.array([np.amin(occ), np.amax(occ), np.mean(occ)])
        known_values = np.array([1.60784581e-08, 2.42148871e-03, 1.37378118e-04])
        self.assertAllclose(values, known_values)

    def test_target_full_no_project(self):
        occ = self._get_occupation("full", None)
        values = np.array([np.amin(occ), np.amax(occ), np.mean(occ)])
        known_values = np.array([5.23868995e-17, 3.12442358e-02, 5.99949472e-04])
        self.assertAllclose(values, known_values)

    def test_target_dmet_no_project(self):
        occ = self._get_occupation("dmet", None)
        values = np.array([np.amin(occ), np.amax(occ), np.mean(occ)])
        known_values = np.array([1.62184716e-08, 9.82891292e-03, 5.32748449e-04])
        self.assertAllclose(values, known_values)
