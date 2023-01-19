import unittest
import numpy as np

from vayesta import rpa
from vayesta.tests.common import TestCase
from vayesta.tests import testsystems


class MoleculeRPATest(TestCase):
    PLACES = 8

    def _test_energy(self, emb, known_values):
        """Test the RPA energy.
        """

        self.assertAlmostEqual(emb.e_tot, known_values['e_tot'], self.PLACES)

    def _test_mom0(self, rpa_orig, rirpa_moms):
        """Test that the RPA and RIRPA zeroth moments agree.
        """
        self.assertAlmostEqual(abs(rirpa_moms - rpa_orig.gen_moms(3)).max(), 0.0, self.PLACES)

    def test_n2_ccpvdz_dRIRPA(self):
        """Tests for LiH cc-pvdz with dRIRPA.
        """

        orig_rpa = rpa.ssRPA(testsystems.n2_ccpvdz_df.rhf())
        orig_rpa.kernel()

        known_values = {"e_tot": -109.27376877774732}
        self._test_energy(orig_rpa, known_values)

        rirpa = rpa.ssRIRPA(testsystems.n2_ccpvdz_df.rhf())
        rirpa_moms, error_est = rirpa.kernel_moms(3, analytic_lower_bound=True)

        self._test_mom0(orig_rpa, rirpa_moms)

    def test_n2_ccpvdz_dRIRPA_error_estimates(self):
        rirpa = rpa.ssRIRPA(testsystems.n2_ccpvdz_df.rhf())
        # Use number of points where errors will be meaningful.
        rirpa_moms, error_est = rirpa.kernel_moms(0, analytic_lower_bound=True, npoints=4)
        self.assertAlmostEqual(error_est[0], 0.0600120085568549,)
        self.assertAlmostEqual(error_est[1], 0.0003040334773499559)

if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
