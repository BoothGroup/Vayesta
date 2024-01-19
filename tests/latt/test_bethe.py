import pytest

from tests.common import TestCase
from vayesta.lattmod import bethe


@pytest.mark.fast
class BetheTests(TestCase):
    PLACES_ENERGY = 8
    PLACES_DOCC = 8

    def _test_bethe(self, t, u, known_values):
        """Test the Bethe lattice for a given T, U."""

        e = bethe.hubbard1d_bethe_energy(t, u)
        self.assertAlmostEqual(e, known_values["energy"], self.PLACES_ENERGY)

        d = bethe.hubbard1d_bethe_docc(t, u)
        self.assertAlmostEqual(d, known_values["docc"], self.PLACES_DOCC)

        d1 = bethe.hubbard1d_bethe_docc_numdiff(t, u, du=1e-12, order=1)
        self.assertAlmostEqual(d1, known_values["docc-numdiff1"], self.PLACES_DOCC)

        d2 = bethe.hubbard1d_bethe_docc_numdiff(t, u, du=1e-12, order=2)
        self.assertAlmostEqual(d2, known_values["docc-numdiff2"], self.PLACES_DOCC)

    def test_bethe_T1_U0(self):
        """Test the T=1, U=0 Bethe lattice."""

        known_values = {
            "energy": -1.2732565954632262,
            "docc": 0.2499999972419595,
            "docc-numdiff1": 0.25002222514558525,
            "docc-numdiff2": 0.2500916140846243,
        }

        self._test_bethe(1.0, 0.0, known_values)

    def test_bethe_T2_U4(self):
        """Test the T=2, U=4 Bethe lattice."""

        known_values = {
            "energy": -1.688748682251278,
            "docc": 0.17545254464835117,
            "docc-numdiff1": 0.17530421558831222,
            "docc-numdiff2": 0.17510992655900282,
        }

        self._test_bethe(2.0, 4.0, known_values)
