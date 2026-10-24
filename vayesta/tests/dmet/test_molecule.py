import sys
import pytest
import unittest
import numpy as np
from vayesta import dmet
from vayesta.tests.common import TestCase
from vayesta.tests import testsystems


FCI_SOLVER_OPTS = dict(davidson_only=False, init_guess="mf")


class MoleculeTest(TestCase):
    PLACES_ENERGY = 7
    CONV_TOL = 1e-9

    @classmethod
    def setUpClass(cls):
        try:
            import cvxpy
        except ImportError:
            pytest.skip("Requires cvxpy")
        cls.mf = testsystems.h6_sto6g.rhf()

    @classmethod
    def tearDownClass(cls):
        del cls.mf

    def _test_converged(self, emb, known_values=None):
        """Test that the DMET has converged."""

        self.assertTrue(emb.converged)

    def _test_energy(self, emb, known_values):
        """Test that the energy matches a known value."""
        self.assertAlmostEqual(emb.e_tot, known_values["e_tot"], self.PLACES_ENERGY)

    def test_cc(self):
        """Test H6 STO-6G with FCI solver, IAO fragmentation and charge consistency."""
        emb = dmet.DMET(
            testsystems.h6_sto6g.rhf(),
            solver="FCI",
            charge_consistent=True,
            bath_options=dict(bathtype="dmet"),
            conv_tol=self.CONV_TOL,
            solver_options=FCI_SOLVER_OPTS,
        )
        with emb.iao_fragmentation() as f:
            f.add_atomic_fragment([0, 1])
            f.add_atomic_fragment([2, 3])
            f.add_atomic_fragment([4, 5])
        emb.kernel()

        known_values = {"e_tot": -3.259355757394294}

        self._test_converged(emb)
        self._test_energy(emb, known_values)

    def test_nocc(self):
        """Test H6 STO-6G with FCI solver, IAO fragmentation and no charge consistency."""
        emb = dmet.DMET(
            testsystems.h6_sto6g.rhf(),
            solver="FCI",
            charge_consistent=False,
            bath_options=dict(bathtype="dmet"),
            conv_tol=self.CONV_TOL,
            solver_options=FCI_SOLVER_OPTS,
        )
        with emb.iao_fragmentation() as f:
            f.add_atomic_fragment([0, 1])
            f.add_atomic_fragment([2, 3])
            f.add_atomic_fragment([4, 5])
        emb.kernel()

        known_values = {"e_tot": -3.2593557575050305}

        self._test_converged(emb)
        self._test_energy(emb, known_values)

    def test_nocc_ccsd(self):
        """Test H6 STO-6G with FCI solver, IAO fragmentation and no charge consistency."""
        emb = dmet.DMET(
            testsystems.h6_sto6g.rhf(),
            solver="CCSD",
            charge_consistent=False,
            bath_options=dict(bathtype="dmet"),
            conv_tol=self.CONV_TOL,
        )
        with emb.iao_fragmentation() as f:
            f.add_atomic_fragment([0, 1])
            f.add_atomic_fragment([2, 3])
            f.add_atomic_fragment([4, 5])
        emb.kernel()

        known_values = {"e_tot": -3.2593387676678667}

        self._test_converged(emb)
        self._test_energy(emb, known_values)

    def test_renorm_interaction(self):
        emb = dmet.DMET(
            testsystems.h6_sto6g_df.uhf(),
            solver="FCI",
            charge_consistent=False,
            bath_options=dict(bathtype="dmet"),
            conv_tol=self.CONV_TOL,
            oneshot=True,
            screening="mrpa",
            solver_options=FCI_SOLVER_OPTS,
        )
        with emb.iao_fragmentation() as f:
            f.add_atomic_fragment([0, 1])
            f.add_atomic_fragment([2, 3])
            f.add_atomic_fragment([4, 5])
        emb.kernel()

        known_values = {"e_tot": -3.2536357694705185}

        self._test_energy(emb, known_values)

    def test_chempot_bracket_after_solver_failure(self):
        """The chemical potential bracket search has to continue in the same direction after a solver failure."""

        def get_emb():
            emb = dmet.DMET(
                testsystems.water_sto3g.rhf(),
                solver="CCSD",
                charge_consistent=False,
                bath_options=dict(bathtype="dmet"),
                oneshot=True,
                # The chemical potential is only converged to within max_elec_err * nelectron:
                max_elec_err=1e-7,
            )
            with emb.iao_fragmentation() as f:
                f.add_all_atomic_fragments()
            return emb

        emb_ref = get_emb()
        emb_ref.kernel()

        emb = get_emb()
        calc_electron_number_defect = emb.calc_electron_number_defect
        failures = []

        def calc_electron_number_defect_with_failure(chempot, *args, **kwargs):
            # Simulate a solver failure at the first trial chemical potential of the bracket search:
            if chempot != 0 and not failures:
                failures.append(chempot)
                raise np.linalg.LinAlgError("Simulated DIIS failure")
            return calc_electron_number_defect(chempot, *args, **kwargs)

        emb.calc_electron_number_defect = calc_electron_number_defect_with_failure
        emb.kernel()
        # Make sure that the bracket search was exercised:
        self.assertEqual(len(failures), 1)
        self.assertAlmostEqual(emb.e_tot, emb_ref.e_tot, self.PLACES_ENERGY)

    def test_full_bath(self):
        """Test H6 STO-6G with FCI solver, IAO fragmentation and complete bath."""
        emb = dmet.DMET(
            testsystems.h6_sto6g.rhf(),
            solver="FCI",
            charge_consistent=False,
            bath_options=dict(bathtype="full"),
            conv_tol=self.CONV_TOL,
            solver_options=FCI_SOLVER_OPTS,
        )
        with emb.iao_fragmentation() as f:
            f.add_atomic_fragment([0, 1])
            f.add_atomic_fragment([2, 3])
            f.add_atomic_fragment([4, 5])
        emb.kernel()

        known_values = {"e_tot": -3.2587710893946102}

        self._test_converged(emb)
        self._test_energy(emb, known_values)


if __name__ == "__main__":
    print("Running %s" % __file__)
    unittest.main()
