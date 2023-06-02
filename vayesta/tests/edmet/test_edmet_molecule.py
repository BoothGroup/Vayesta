import unittest
import numpy as np

import pyscf.gto
import pyscf.scf
import pyscf.tools.ring

import vayesta
from vayesta import edmet
from vayesta.tests.common import TestCase
from vayesta.tests import testsystems


class MolecularEDMETTest(TestCase):
    ENERGY_PLACES = 7
    CONV_TOL = 1e-9

    def _test_energy(self, emb, known_values):
        """Tests that the energy matfhes a known values.
        """
        self.assertAlmostEqual(emb.e_tot, known_values['e_tot'], self.ENERGY_PLACES)

    def test_h6_sto6g_FCI_IAO_1occ(self):
        emb = edmet.EDMET(
                testsystems.h6_sto6g.rhf(),
                solver='FCI',
                solver_options={"max_boson_occ":1},
                conv_tol=self.CONV_TOL,
                bosonic_interaction="direct",
                oneshot=True,
                make_dd_moments=False,
        )
        with emb.iao_fragmentation() as f:
            f.add_atomic_fragment([0, 1])
            f.add_atomic_fragment([2, 3])
            f.add_atomic_fragment([4, 5])
        emb.kernel()

        uemb = edmet.EDMET(
                testsystems.h6_sto6g.uhf(),
                solver='FCI',
                solver_options={"max_boson_occ":1},
                conv_tol=self.CONV_TOL,
                bosonic_interaction="direct",
                oneshot=True,
                make_dd_moments=False,
        )
        with uemb.iao_fragmentation() as f:
            f.add_atomic_fragment([0, 1])
            f.add_atomic_fragment([2, 3])
            f.add_atomic_fragment([4, 5])
        uemb.kernel()

        known_values = {'e_tot': -3.2687823852000726}

        self._test_energy(emb, known_values)
        self._test_energy(uemb, known_values)

    def test_h6_sto6g_FCI_IAO_2occ(self):
        emb = edmet.EDMET(
                testsystems.h6_sto6g.rhf(),
                solver='FCI',
                solver_options={"max_boson_occ":2},
                conv_tol=self.CONV_TOL,
        )
        with emb.iao_fragmentation() as f:
            f.add_atomic_fragment([0, 1])
            f.add_atomic_fragment([2, 3])
            f.add_atomic_fragment([4, 5])
        emb.kernel()

        uemb = edmet.EDMET(
                testsystems.h6_sto6g.uhf(),
                solver='FCI',
                solver_options={"max_boson_occ":2},
                conv_tol=self.CONV_TOL,
                bosonic_interaction="direct",
                oneshot=True,
                make_dd_moments=False,
        )
        with uemb.iao_fragmentation() as f:
            f.add_atomic_fragment([0, 1])
            f.add_atomic_fragment([2, 3])
            f.add_atomic_fragment([4, 5])
        uemb.kernel()


        known_values = {'e_tot': -3.268894063433855}

        self._test_energy(emb, known_values)
        self._test_energy(uemb, known_values)

    def test_h6_sto6g_FCI_IAO_2occ(self):
        emb = edmet.EDMET(
                testsystems.h6_sto6g.rhf(),
                solver='FCI',
                solver_options={"max_boson_occ":2},
                conv_tol=self.CONV_TOL,
                bosonic_interaction="direct",
                oneshot=True,
                make_dd_moments=False,
        )
        with emb.iao_fragmentation() as f:
            f.add_atomic_fragment([0, 1])
            f.add_atomic_fragment([2, 3])
            f.add_atomic_fragment([4, 5])
        emb.kernel()

        known_values = {'e_tot': -3.268894063433855}

        self._test_energy(emb, known_values)

        uemb = edmet.EDMET(
                testsystems.h6_sto6g.uhf(),
                solver='FCI',
                solver_options={"max_boson_occ":2},
                conv_tol=self.CONV_TOL,
                bosonic_interaction="direct",
                oneshot=True,
                make_dd_moments=False,
        )
        with uemb.iao_fragmentation() as f:
            f.add_atomic_fragment([0, 1])
            f.add_atomic_fragment([2, 3])
            f.add_atomic_fragment([4, 5])
        uemb.kernel()

        self._test_energy(uemb, known_values)

    @unittest.skipIf(vayesta.ebcc is None, "EBCC installation not found.")
    def test_h2o_ccpvdz_EBCCSD_IAO_2occ(self):
        emb = edmet.EDMET(
                testsystems.water_ccpvdz.rhf(),
                solver='CCSD-S-1-1',
                conv_tol=self.CONV_TOL,
                bosonic_interaction="direct",
                oneshot=True,
                make_dd_moments=False,
        )
        with emb.iao_fragmentation(minao='minao') as f:
            f.add_all_atomic_fragments()
        emb.kernel()

        known_values = {'e_tot': -76.26535574691708}

        self._test_energy(emb, known_values)

        uemb = edmet.EDMET(
                testsystems.water_ccpvdz.uhf(),
                solver='CCSD-S-1-1',
                conv_tol=self.CONV_TOL,
                bosonic_interaction="direct",
                oneshot=True,
                make_dd_moments=False,
        )
        with uemb.iao_fragmentation(minao='minao') as f:
            f.add_all_atomic_fragments()
        uemb.kernel()

        self._test_energy(uemb, known_values)


if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
