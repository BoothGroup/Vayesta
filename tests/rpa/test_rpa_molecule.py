import unittest

from vayesta import rpa
from tests.common import TestCase
from tests import systems


class MoleculeRPATest(TestCase):
    PLACES = 8

    def _test_energy(self, emb, known_values):
        """Test the RPA energy."""

        self.assertAlmostEqual(emb.e_tot, known_values["e_tot"], self.PLACES)

    def test_lih_ccpvdz_RPAX(self):
        """Tests for LiH cc-pvdz with RPAX."""

        emb = rpa.RPA(systems.lih_ccpvdz.rhf())
        emb.kernel("rpax")

        known_values = {"e_tot": -8.021765296851472}

        self._test_energy(emb, known_values)

    def test_lih_ccpvdz_dRPA(self):
        """Tests for LiH cc-pvdz with dPRA."""

        emb = rpa.RPA(systems.lih_ccpvdz.rhf())
        emb.kernel("drpa")

        known_values = {"e_tot": -8.015594007709575}

        self._test_energy(emb, known_values)

        emb = rpa.ssRPA(systems.lih_ccpvdz.rhf())
        emb.kernel()

        self._test_energy(emb, known_values)


if __name__ == "__main__":
    print("Running %s" % __file__)
    unittest.main()
