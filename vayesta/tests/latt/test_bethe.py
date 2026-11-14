import pytest
import unittest

from vayesta.lattmod import bethe
from vayesta.tests.common import TestCase


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

        # Tiny steps amplify energy roundoff; compare both stencils to the analytic
        # derivative at two resolved step sizes instead of storing cancellation errors.
        for order in (1, 2):
            for du in (1e-5, 1e-6):
                with self.subTest(order=order, du=du):
                    d_num = bethe.hubbard1d_bethe_docc_numdiff(t, u, du=du, order=order)
                    self.assertAlmostEqual(d_num, d, self.PLACES_DOCC)

    def test_bethe_T1_U0(self):
        """Test the T=1, U=0 Bethe lattice."""

        known_values = {
            "energy": -1.2732565954632262,
            "docc": 0.2499999972419595,
        }

        self._test_bethe(1.0, 0.0, known_values)

    def test_bethe_T2_U4(self):
        """Test the T=2, U=4 Bethe lattice."""

        known_values = {
            "energy": -1.688748682251278,
            "docc": 0.17545254464835117,
        }

        self._test_bethe(2.0, 4.0, known_values)


if __name__ == "__main__":
    print("Running %s" % __file__)
    unittest.main()
