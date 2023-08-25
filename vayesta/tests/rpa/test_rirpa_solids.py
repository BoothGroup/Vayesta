import pytest

from vayesta import rpa
from vayesta.tests.common import TestCase
from vayesta.tests import testsystems


class DiamondRIRPATest(TestCase):
    PLACES = 8

    @classmethod
    def setUpClass(cls):
        cls.sys = testsystems.diamond_sto3g_s211
        cls.known_results = dict(e_tot=-149.51936410641733, e_corr=-0.19193623440986585)

    def _test_energy(self, myrpa):
        """Test the RPA energy."""
        self.assertAlmostEqual(myrpa.e_corr, self.known_results["e_corr"], self.PLACES)
        self.assertAlmostEqual(myrpa.e_tot, self.known_results["e_tot"], self.PLACES)

    @pytest.mark.slow
    def test_energy_rhf_opt(self):
        """Tests for diamond with optimised RHF dRPA code."""

        rirpa = rpa.rirpa.ssRIdRRPA(self.sys.rhf())
        rirpa.kernel_energy()
        self._test_energy(rirpa)

    @pytest.mark.fast
    def test_energy_rhf_generic(self):
        """Tests for diamond with generic RHF RIRPA code."""

        rirpa = rpa.rirpa.ssRIRRPA(self.sys.rhf())
        rirpa.kernel_energy()
        self._test_energy(rirpa)

    @pytest.mark.slow
    def test_energy_uhf(self):
        """Tests for diamond with generic UHF RIRPA code."""

        rirpa = rpa.rirpa.ssRIURPA(self.sys.uhf())
        rirpa.kernel_energy()
        self._test_energy(rirpa)

    @pytest.mark.fast
    def test_rhf_moments(self):
        gen_rirpa = rpa.rirpa.ssRIRRPA(self.sys.rhf())
        opt_rirpa = rpa.rirpa.ssRIdRRPA(self.sys.rhf())
        mom0_gen = gen_rirpa.kernel_moms(0)[0]
        mom0_opt = opt_rirpa.kernel_moms(0)[0]
        self.assertAllclose(mom0_gen, mom0_opt, self.PLACES)


@pytest.mark.slow
class GrapheneRIRPATest(DiamondRIRPATest):
    @classmethod
    def setUpClass(cls):
        cls.sys = testsystems.graphene_sto3g_s211
        cls.known_results = dict(e_tot=-150.15057360171875, e_corr=-0.17724246753903117)
