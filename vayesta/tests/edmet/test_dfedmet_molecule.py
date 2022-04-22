import unittest
import numpy as np

import pyscf.gto
import pyscf.scf
import pyscf.tools.ring

from vayesta import edmet
from vayesta.tests.cache import moles
import vayesta

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
                solver_options={"max_boson_occ":1},
                conv_tol=self.CONV_TOL,
                bosonic_interaction="direct",
                oneshot=True,
                make_dd_moments=False,
        )
        emb.iao_fragmentation()
        emb.add_atomic_fragment([0, 1])
        emb.add_atomic_fragment([2, 3])
        emb.add_atomic_fragment([4, 5])
        emb.kernel()

        known_values = {'e_tot': -3.2644471120076393}

        self._test_energy(emb, known_values)

    def test_h6_sto6g_EBFCI_IAO_2occ(self):
        emb = edmet.EDMET(
                moles['h6_sto6g_df']['rhf'],
                solver='EBFCI',
                solver_options={"max_boson_occ":2},
                conv_tol=self.CONV_TOL,
                bosonic_interaction="direct",
                oneshot=True,
                make_dd_moments=False,
        )
        emb.iao_fragmentation()
        emb.add_atomic_fragment([0, 1])
        emb.add_atomic_fragment([2, 3])
        emb.add_atomic_fragment([4, 5])
        emb.kernel()

        uemb = edmet.EDMET(
                moles['h6_sto6g_df']['uhf'],
                solver='EBFCI',
                solver_options={"max_boson_occ":2},
                conv_tol=self.CONV_TOL,
                bosonic_interaction="direct",
                oneshot=True,
                make_dd_moments=False,
        )
        uemb.iao_fragmentation()
        uemb.add_atomic_fragment([0, 1])
        uemb.add_atomic_fragment([2, 3])
        uemb.add_atomic_fragment([4, 5])
        uemb.kernel()

        known_values = {'e_tot': -3.2645626611174006}

        self._test_energy(emb, known_values)
        self._test_energy(uemb, known_values)

    @unittest.skipIf(vayesta.ebcc is None, "EBCC installation not found.")
    def test_h2o_ccpvdz_EBCCSD_IAO(self):
        emb = edmet.EDMET(
                moles['h2o_ccpvdz_df']['rhf'],
                solver='EBCCSD',
                conv_tol=self.CONV_TOL,
                oneshot=True,
                make_dd_moments=False,
                bosonic_interaction="direct",
        )
        emb.iao_fragmentation()
        emb.add_all_atomic_fragments()
        emb.kernel()

        known_values = {'e_tot': -76.26516984456478}

        self._test_energy(emb, known_values)

        uemb = edmet.EDMET(
                moles['h2o_ccpvdz_df']['uhf'],
                solver='EBCCSD',
                conv_tol=self.CONV_TOL,
                oneshot=True,
                make_dd_moments=False,
                bosonic_interaction="direct",
        )
        uemb.iao_fragmentation()
        uemb.add_all_atomic_fragments()
        uemb.kernel()

        self._test_energy(uemb, known_values)

    @unittest.skipIf(vayesta.ebcc is None, "EBCC installation not found.")
    def test_h6_sto3g_EBCCSD_IAO_2occ(self):
        emb = edmet.EDMET(
                moles['h6_sto6g_df']['rhf'],
                solver='EBCCSD',
                conv_tol=self.CONV_TOL,
                maxiter=30,
                bosonic_interaction="qba_bos_ex",
                occ_proj_kernel=False,
        )
        emb.iao_fragmentation()
        emb.add_all_atomic_fragments()
        emb.kernel()

        known_values = {'e_tot': -3.282140724303993}

        self._test_energy(emb, known_values)

        uemb = edmet.EDMET(
                moles['h6_sto6g_df']['uhf'],
                solver='EBCCSD',
                conv_tol=self.CONV_TOL,
                maxiter=30,
                bosonic_interaction="qba_bos_ex",
                occ_proj_kernel=False,
        )
        uemb.iao_fragmentation()
        uemb.add_all_atomic_fragments()
        uemb.kernel()

        self._test_energy(uemb, known_values)

if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
