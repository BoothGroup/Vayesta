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

    def _test_mom0(self, rpa_orig, rirpa):
        """Test that the RPA and RIRPA zeroth moments agree.
        """

        rim0, error_est = rirpa.kernel_moms(0)

        self.assertAlmostEqual(abs(rim0[0] - rpa_orig.gen_moms(0)[0]).max(), 0.0, self.PLACES)

    def test_n2_ccpvdz_dRIRPA(self):
        """Tests for LiH cc-pvdz with dRIRPA.
        """

        orig_rpa = rpa.ssRPA(testsystems.n2_ccpvdz_df.rhf())
        orig_rpa.kernel()

        known_values = {"e_tot": -109.27376877774732}
        self._test_energy(orig_rpa, known_values)

        rirpa = rpa.ssRIRPA(testsystems.n2_ccpvdz_df.rhf())

        self._test_mom0(orig_rpa, rirpa)


if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
