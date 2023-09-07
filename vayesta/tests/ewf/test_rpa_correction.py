import unittest
import numpy as np
import vayesta
import vayesta.ewf
from vayesta.core.util import cache
from vayesta.tests import testsystems
from vayesta.tests.common import TestCase


class Test_RPA_Corrections_Ethanol_RHF(TestCase):
    system = testsystems.ethanol_631g_df

    @property
    def mf(self):
        return self.system.rhf()

    def get_nl_energies(self, correction, bathtype="dmet"):
        emb = vayesta.ewf.EWF(
            self.mf, bath_options={"bathtype": bathtype}, solver="CCSD", screening="mrpa", ext_rpa_correction=correction
        )
        with emb.iao_fragmentation() as f:
            f.add_all_atomic_fragments()
        emb.kernel()
        return emb.e_nonlocal, emb.e_rpa

    def test_rpa_correction(self):
        enl, erpa = self.get_nl_energies("erpa")
        self.assertAlmostEqual(erpa, -0.29138256715100397)
        self.assertAlmostEqual(enl, -0.3087486792144427)

    def test_cumulant_correction(self):
        enl, erpa = self.get_nl_energies("cumulant")
        self.assertAlmostEqual(erpa, -0.5145262339186916)
        self.assertAlmostEqual(enl, -0.31553721476368396)


class Test_RPA_Corrections_complete(Test_RPA_Corrections_Ethanol_RHF):
    """Tests with a complete bath in all clusters. This should give no nonlocal correction in any case."""

    system = testsystems.water_631g_df

    def test_rpa_correction(self):
        enl, erpa = self.get_nl_energies("erpa", "full")
        self.assertAlmostEqual(enl, 0.0)

    def test_cumulant_correction(self):
        enl, erpa = self.get_nl_energies("cumulant", "full")
        self.assertAlmostEqual(enl, 0.0)
