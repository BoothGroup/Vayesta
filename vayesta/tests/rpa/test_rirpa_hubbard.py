import unittest

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

    def test_14_u4(self):
        """Tests for N=14 U=4 Hubbard model.
        """

        key = 'hubb_14_u4_df'
        known_values_drpa = {'e_tot': -7.776536889696544}

        emb = rpa.ssRPA(getattr(testsystems, key).rhf())
        emb.kernel()
        self._test_energy(emb, known_values_drpa)

        rirpa = rpa.ssRIRPA(getattr(testsystems, key).rhf())

        self._test_mom0(emb, rirpa)


if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
