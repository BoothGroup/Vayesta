import unittest

from vayesta import rpa
from tests.common import TestCase
from tests import systems


class MoleculeRPATest(TestCase):
    PLACES = 8

    def _test_energy(self, emb, known_values):
        """Test the RPA energy."""

        self.assertAlmostEqual(emb.e_tot, known_values["e_tot"], self.PLACES)

    def _test_mom0(self, rpa_orig, rirpa_moms):
        """Test that the RPA and RIRPA zeroth moments agree."""
        self.assertAlmostEqual(abs(rirpa_moms - rpa_orig.gen_moms(3)).max(), 0.0, self.PLACES)

    def test_n2_ccpvdz_dRIRPA(self):
        """Tests for LiH cc-pvdz with dRIRPA."""

        orig_rpa = rpa.ssRPA(systems.n2_ccpvdz_df.rhf())
        orig_rpa.kernel()

        known_values = {"e_tot": -109.27376877774732}
        self._test_energy(orig_rpa, known_values)
        # Check dRPA specialised RHF code.
        rirpa = rpa.rirpa.ssRIdRRPA(systems.n2_ccpvdz_df.rhf())
        rirpa.kernel_energy()
        self._test_energy(rirpa, known_values)
        # Check spin-generic code.
        rirpa = rpa.rirpa.ssRIRRPA(systems.n2_ccpvdz_df.rhf())
        rirpa.kernel_energy()
        self._test_energy(rirpa, known_values)

        rirpa_moms, error_est = rirpa.kernel_moms(3, analytic_lower_bound=True)
        self._test_mom0(orig_rpa, rirpa_moms)

    def test_n2_ccpvdz_dRIRPA_error_estimates(self):
        rirpa = rpa.rirpa.ssRIRRPA(systems.n2_ccpvdz_df.rhf())
        # Use number of points where errors will be meaningful.
        # Note that with this few points the fact that the quadrature optimisation is not invariant to orbital rotations
        # can cause issues, so we can just use a specific grid spacing.
        rirpa_moms, error_est = rirpa.kernel_moms(
            0, analytic_lower_bound=True, npoints=4, ainit=1.5214230673470202, opt_quad=False
        )
        self.assertAlmostEqual(error_est[0], 0.05074756294730469)
        self.assertAlmostEqual(error_est[1], 0.00024838720802440015)
