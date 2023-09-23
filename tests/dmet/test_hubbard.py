import pytest
import unittest
from vayesta import dmet
from tests.common import TestCase
from tests import testsystems


class HubbardDMETTests(TestCase):
    PLACES_ENERGY = 6
    CONV_TOL = 1e-8

    @classmethod
    def setUpClass(cls):
        try:
            import cvxpy
        except ImportError:
            pytest.skip("Requires cvxpy")

    def _test_converged(self, emb, known_values=None):
        """Test that the DMET has converged."""

        self.assertTrue(emb.converged)

    def _test_energy(self, emb, known_values):
        """Test that the energy matches a known value."""
        self.assertAlmostEqual(emb.e_tot, known_values["e_tot"], self.PLACES_ENERGY)

    def test_6_u0_1imp(self):
        """Tests for N=6 U=0 Hubbard model with single site impurities."""
        emb = dmet.DMET(
            testsystems.hubb_6_u0.rhf(), solver="FCI", charge_consistent=False, conv_tol=self.CONV_TOL, maxiter=50
        )
        with emb.site_fragmentation() as f:
            frag = f.add_atomic_fragment(0)
            frag.add_tsymmetric_fragments(tvecs=[6, 1, 1])
        emb.kernel()

        known_values = {"e_tot": -8.0}

        self._test_converged(emb)
        self._test_energy(emb, known_values)

    def test_10_u2_2imp(self):
        """Tests for N=10 U=2 Hubbard model with double site impurities."""
        emb = dmet.DMET(
            testsystems.hubb_10_u2.rhf(), solver="FCI", charge_consistent=False, conv_tol=self.CONV_TOL, maxiter=50
        )
        with emb.site_fragmentation() as f:
            frag = f.add_atomic_fragment([0, 1])
            frag.add_tsymmetric_fragments(tvecs=[5, 1, 1])
        emb.kernel()

        known_values = {"e_tot": -8.741824246073978}

        self._test_converged(emb)
        self._test_energy(emb, known_values)

    def test_6x6_u0_1x1imp(self):
        """Tests for 6x6 U=0 Hubbard model with single site impurities."""
        emb = dmet.DMET(
            testsystems.hubb_6x6_u0_1x1imp.rhf(),
            solver="FCI",
            charge_consistent=False,
            conv_tol=self.CONV_TOL,
            maxiter=50,
        )
        with emb.site_fragmentation() as f:
            frag = f.add_atomic_fragment([0])
            frag.add_tsymmetric_fragments(tvecs=[6, 6, 1])
        emb.kernel()

        known_values = {"e_tot": -56.0}

        self._test_converged(emb)
        self._test_energy(emb, known_values)

    def test_6x6_u6_1x1imp(self):
        """Tests for 6x6 U=6 Hubbard model with single site impurities."""
        emb = dmet.DMET(
            testsystems.hubb_6x6_u6_1x1imp.rhf(),
            solver="FCI",
            charge_consistent=False,
            conv_tol=self.CONV_TOL,
            maxiter=50,
            solver_options={"conv_tol": 1e-12},
        )
        with emb.site_fragmentation() as f:
            frag = f.add_atomic_fragment([0])
            frag.add_tsymmetric_fragments(tvecs=[6, 6, 1])
        emb.kernel()

        known_values = {"e_tot": -41.040841420346695}

        self._test_converged(emb)
        self._test_energy(emb, known_values)

    def test_8x8_u2_2x2imp(self):
        """Tests for 8x8 U=2 Hubbard model with 2x2 impurities."""
        emb = dmet.DMET(
            testsystems.hubb_8x8_u2_2x2imp.rhf(),
            solver="FCI",
            charge_consistent=False,
            conv_tol=self.CONV_TOL,
            maxiter=100,
            solver_options={"conv_tol": 1e-12},
        )
        with emb.site_fragmentation() as f:
            frag = f.add_atomic_fragment([0, 1, 2, 3])
            frag.add_tsymmetric_fragments(tvecs=[4, 4, 1])
        emb.kernel()

        known_values = {"e_tot": -85.02643076273672}

        self._test_converged(emb)
        self._test_energy(emb, known_values)


if __name__ == "__main__":
    print("Running %s" % __file__)
    unittest.main()
