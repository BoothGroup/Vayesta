import unittest
import numpy as np

from vayesta import dmet
from vayesta.tests.cache import moles


class MoleculeDMETTest(unittest.TestCase):
    PLACES_ENERGY = 7
    CONV_TOL = 1e-9

    def _test_converged(self, emb, known_values=None):
        """Test that the DMET has converged.
        """

        self.assertTrue(emb.converged)

    def _test_energy(self, emb, known_values):
        """Test that the energy matches a known value.
        """

        self.assertAlmostEqual(emb.e_tot, known_values['e_tot'], self.PLACES_ENERGY)

    def test_h6_sto6g_FCI_IAO_cc(self):
        """Test H6 STO-6G with FCI solver, IAO fragmentation and charge consistency.
        """

        emb = dmet.DMET(
                moles['h6_sto6g']['rhf'],
                solver='FCI',
                charge_consistent=True,
                bath_type=None,
                conv_tol=self.CONV_TOL,
        )
        emb.iao_fragmentation()
        emb.add_atomic_fragment([0, 1])
        emb.add_atomic_fragment([2, 3])
        emb.add_atomic_fragment([4, 5])
        emb.kernel()

        known_values = {'e_tot': -3.2596560444286524}

        self._test_converged(emb)
        self._test_energy(emb, known_values)

    def test_h6_sto6g_FCI_IAO_nocc(self):
        """Test H6 STO-6G with FCI solver, IAO fragmentation and no charge consistency.
        """

        emb = dmet.DMET(
                moles['h6_sto6g']['rhf'],
                solver='FCI',
                charge_consistent=False,
                bath_type=None,
                conv_tol=self.CONV_TOL,
        )
        emb.iao_fragmentation()
        emb.add_atomic_fragment([0, 1])
        emb.add_atomic_fragment([2, 3])
        emb.add_atomic_fragment([4, 5])
        emb.kernel()

        known_values = {'e_tot': -3.2596414844443995}

        self._test_converged(emb)
        self._test_energy(emb, known_values)

    def test_h6_sto6g_FCI_IAO_all(self):
        """Test H6 STO-6G with FCI solver, IAO fragmentation and complete bath.
        """

        emb = dmet.DMET(
                moles['h6_sto6g']['rhf'],
                solver='FCI',
                charge_consistent=False,
                bath_type='all',
                conv_tol=self.CONV_TOL,
        )
        emb.iao_fragmentation()
        emb.add_atomic_fragment([0, 1])
        emb.add_atomic_fragment([2, 3])
        emb.add_atomic_fragment([4, 5])
        emb.kernel()

        known_values = {'e_tot': -3.2585986118561703}

        self._test_converged(emb)
        self._test_energy(emb, known_values)

    def test_h6_sto6g_FCI_IAO_BNO(self):
        """Test H6 STO-6G with FCI solver, IAO fragmentation and complete BNO bath.
        """

        emb = dmet.DMET(
                moles['h6_sto6g']['rhf'],
                solver='FCI',
                charge_consistent=False,
                bath_type='MP2-BNO',
                bno_threshold=np.inf,
                conv_tol=self.CONV_TOL,
        )
        emb.iao_fragmentation()
        emb.add_atomic_fragment([0, 1])
        emb.add_atomic_fragment([2, 3])
        emb.add_atomic_fragment([4, 5])
        emb.kernel()

        known_values = {'e_tot': -3.2596414844443995}

        self._test_converged(emb)
        self._test_energy(emb, known_values)


if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
