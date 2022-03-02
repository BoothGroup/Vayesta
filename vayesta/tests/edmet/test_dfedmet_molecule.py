import unittest
import numpy as np

import pyscf.gto
import pyscf.scf
import pyscf.tools.ring

from vayesta import edmet
from vayesta.tests.cache import moles


class MolecularDFEDMETTest(unittest.TestCase):
    ENERGY_PLACES = 8
    CONV_TOL = 1e-9

    def _test_energy(self, emb, known_values):
        """Tests that the energy matfhes a known values.
        """

        self.assertAlmostEqual(emb.e_tot, known_values['e_tot'], self.ENERGY_PLACES)

    def test_h6_sto6g_EBFCI_IAO_1occ(self):
        emb = edmet.EDMET(
                moles['h6_sto6g_df']['rhf'],
                solver='EBFCI',
                max_boson_occ=1,
                conv_tol=self.CONV_TOL,
        )
        emb.iao_fragmentation()
        emb.add_atomic_fragment([0, 1])
        emb.add_atomic_fragment([2, 3])
        emb.add_atomic_fragment([4, 5])
        emb.kernel()

        known_values = {'e_tot': -3.2542107531828446}

        self._test_energy(emb, known_values)

    def test_h6_sto6g_EBFCI_IAO_2occ(self):
        emb = edmet.EDMET(
                moles['h6_sto6g_df']['rhf'],
                solver='EBFCI',
                max_boson_occ=2,
                conv_tol=self.CONV_TOL,
        )
        emb.iao_fragmentation()
        emb.add_atomic_fragment([0, 1])
        emb.add_atomic_fragment([2, 3])
        emb.add_atomic_fragment([4, 5])
        emb.kernel()

        known_values = {'e_tot': -3.2543264783930717}

        self._test_energy(emb, known_values)


if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
