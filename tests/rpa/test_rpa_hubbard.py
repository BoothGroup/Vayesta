from tests import systems
from tests.common import TestCase
from vayesta import rpa


class MoleculeRPATest(TestCase):
    PLACES = 8

    def _test_energy(self, emb, known_values):
        """Test the RPA energy."""

        self.assertAlmostEqual(emb.e_tot, known_values["e_tot"], self.PLACES)

    def test_6_u0(self):
        """Tests for N=6 U=0 Hubbard model."""

        key = "hubb_6_u0"
        known_values_rpax = {"e_tot": -8.0}
        known_values_drpa = {"e_tot": -8.0}

        emb = rpa.RPA(getattr(systems, key).rhf())
        emb.kernel("rpax")
        self._test_energy(emb, known_values_rpax)

        emb = rpa.RPA(getattr(systems, key).rhf())
        emb.kernel("drpa")
        self._test_energy(emb, known_values_drpa)

        emb = rpa.ssRPA(getattr(systems, key).rhf())
        emb.kernel()
        self._test_energy(emb, known_values_drpa)

    def test_10_u2(self):
        """Tests for N=10 U=2 Hubbard model."""

        key = "hubb_10_u2"
        known_values_rpax = {"e_tot": -9.064312326273644}
        known_values_drpa = {"e_tot": -8.824440982421532}

        emb = rpa.RPA(getattr(systems, key).rhf())
        emb.kernel("rpax")
        self._test_energy(emb, known_values_rpax)

        emb = rpa.RPA(getattr(systems, key).rhf())
        emb.kernel("drpa")
        self._test_energy(emb, known_values_drpa)

        emb = rpa.ssRPA(getattr(systems, key).rhf())
        emb.kernel()
        self._test_energy(emb, known_values_drpa)

    def test_6x6_u0(self):
        """Tests for 6x6 U=0 Hubbard model."""

        key = "hubb_6x6_u0_1x1imp"
        known_values_rpax = {"e_tot": -56.0}
        known_values_drpa = {"e_tot": -56.0}

        emb = rpa.RPA(getattr(systems, key).rhf())
        emb.kernel("rpax")
        self._test_energy(emb, known_values_rpax)

        emb = rpa.RPA(getattr(systems, key).rhf())
        emb.kernel("drpa")
        self._test_energy(emb, known_values_drpa)

        emb = rpa.ssRPA(getattr(systems, key).rhf())
        emb.kernel()
        self._test_energy(emb, known_values_drpa)

    def test_6x6_u2(self):
        """Tests for 6x6 U=2 Hubbard model."""

        key = "hubb_6x6_u2_1x1imp"
        known_values_rpax = {"e_tot": -48.314526436495500}
        known_values_drpa = {"e_tot": -48.740268837302494}

        emb = rpa.RPA(getattr(systems, key).rhf())
        emb.kernel("rpax")
        self._test_energy(emb, known_values_rpax)

        emb = rpa.RPA(getattr(systems, key).rhf())
        emb.kernel("drpa")
        self._test_energy(emb, known_values_drpa)

        emb = rpa.ssRPA(getattr(systems, key).rhf())
        emb.kernel()
        self._test_energy(emb, known_values_drpa)
