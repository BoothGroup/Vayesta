import unittest

from vayesta import rpa
from vayesta.tests.cache import latts


class MoleculeRPATest(unittest.TestCase):
    PLACES = 8

    def _test_energy(self, emb, known_values):
        """Test the RPA energy.
        """

        self.assertAlmostEqual(emb.e_tot, known_values['e_tot'], self.PLACES)

    def _test_mom0(self, rpa_orig, rirpa):
        """Test that the RPA and RIRPA zeroth moments agree.
        """

        rim0, error_est = rirpa.kernel_moms()

        self.assertAlmostEqual(abs(rim0 - rpa_orig.gen_moms(0)[0]).max(), 0.0, self.PLACES)

    def test_16_u4(self):
        """Tests for N=16 U=4 Hubbard model.
        """

        key = 'hubb_16_u4_df'
        known_values_drpa = {'e_tot': -8.847604550030152}

        emb = rpa.ssRPA(latts[key]['rhf'])
        emb.kernel()
        self._test_energy(emb, known_values_drpa)

        rirpa = rpa.ssRIRPA(latts[key]['rhf'])

        self._test_mom0(emb, rirpa)


if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
