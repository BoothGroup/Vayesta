import unittest
import numpy as np
from vayesta import dmet
from vayesta.tests.cache import moles
from vayesta.tests.common import TestCase
from vayesta.tests import testsystems


class MoleculeDMETTest(TestCase):
    PLACES_ENERGY = 7
    CONV_TOL = 1e-9

    @classmethod
    def setUpClass(cls):
        cls.mf = testsystems.h6_sto6g.rhf()

    @classmethod
    def tearDownClass(cls):
        del cls.mf

    def _test_converged(self, emb, known_values=None):
        """Test that the DMET has converged.
        """

        self.assertTrue(emb.converged)

    def _test_energy(self, emb, known_values):
        """Test that the energy matches a known value.
        """
        self.assertAlmostEqual(emb.e_tot, known_values['e_tot'], self.PLACES_ENERGY)

    def test_cc(self):
        """Test H6 STO-6G with FCI solver, IAO fragmentation and charge consistency.
        """
        emb = dmet.DMET(moles['h6_sto6g']['rhf'], solver='FCI', charge_consistent=True,
                bath_options=dict(bathtype='dmet'), conv_tol=self.CONV_TOL)
        with emb.iao_fragmentation() as f:
            f.add_atomic_fragment([0, 1])
            f.add_atomic_fragment([2, 3])
            f.add_atomic_fragment([4, 5])
        emb.kernel()

        known_values = {'e_tot': -3.259355757394294}

        self._test_converged(emb)
        self._test_energy(emb, known_values)

    def test_nocc(self):
        """Test H6 STO-6G with FCI solver, IAO fragmentation and no charge consistency.
        """
        emb = dmet.DMET(moles['h6_sto6g']['rhf'], solver='FCI', charge_consistent=False,
                bath_options=dict(bathtype='dmet'), conv_tol=self.CONV_TOL)
        with emb.iao_fragmentation() as f:
            f.add_atomic_fragment([0, 1])
            f.add_atomic_fragment([2, 3])
            f.add_atomic_fragment([4, 5])
        emb.kernel()

        known_values = {'e_tot': -3.2593557575050305}

        self._test_converged(emb)
        self._test_energy(emb, known_values)

    def test_nocc_ccsd(self):
        """Test H6 STO-6G with FCI solver, IAO fragmentation and no charge consistency.
        """
        emb = dmet.DMET(moles['h6_sto6g']['rhf'], solver='CCSD', charge_consistent=False,
                bath_options=dict(bathtype='dmet'), conv_tol=self.CONV_TOL)
        with emb.iao_fragmentation() as f:
            f.add_atomic_fragment([0, 1])
            f.add_atomic_fragment([2, 3])
            f.add_atomic_fragment([4, 5])
        emb.kernel()

        known_values = {'e_tot': -3.2593387676678667}

        self._test_converged(emb)
        self._test_energy(emb, known_values)


    def test_full_bath(self):
        """Test H6 STO-6G with FCI solver, IAO fragmentation and complete bath.
        """
        emb = dmet.DMET(moles['h6_sto6g']['rhf'], solver='FCI', charge_consistent=False,
                bath_options=dict(bathtype='full'), conv_tol=self.CONV_TOL)
        with emb.iao_fragmentation() as f:
            f.add_atomic_fragment([0, 1])
            f.add_atomic_fragment([2, 3])
            f.add_atomic_fragment([4, 5])
        emb.kernel()

        known_values = {'e_tot': -3.2587710893946102}

        self._test_converged(emb)
        self._test_energy(emb, known_values)


if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
