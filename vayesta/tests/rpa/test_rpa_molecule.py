import unittest
import numpy as np

from vayesta import rpa
from vayesta.tests.cache import moles


class MoleculeRPATest(unittest.TestCase):
    PLACES = 8

    def _test_energy(self, emb, known_values):
        """Test the RPA energy.
        """

        self.assertAlmostEqual(emb.e_tot, known_values['e_tot'], self.PLACES)

    def test_lih_ccpvdz_RPAX(self):
        """Tests for LiH cc-pvdz with RPAX.
        """

        emb = rpa.RPA(moles['lih_ccpvdz']['rhf'])
        emb.kernel('rpax')

        known_values = {"e_tot": -8.021765296851472}

        self._test_energy(emb, known_values)

    def test_lih_ccpvdz_dRPA(self):
        """Tests for LiH cc-pvdz with dPRA.
        """

        emb = rpa.RPA(moles['lih_ccpvdz']['rhf'])
        emb.kernel('drpa')

        known_values = {"e_tot": -8.015594007709575}

        self._test_energy(emb, known_values)

        emb = rpa.dRPA(moles['lih_ccpvdz']['rhf'])
        emb.kernel()

        self._test_energy(emb, known_values)


if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
