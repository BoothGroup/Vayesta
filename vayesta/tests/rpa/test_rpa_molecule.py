import copy
import unittest

import pyscf.ao2mo

from vayesta import rpa
from vayesta.tests.common import TestCase
from vayesta.tests import testsystems


class MoleculeRPATest(TestCase):
    PLACES = 8

    def _test_energy(self, emb, known_values):
        """Test the RPA energy."""

        self.assertAlmostEqual(emb.e_tot, known_values["e_tot"], self.PLACES)

    def test_lih_ccpvdz_RPAX(self):
        """Tests for LiH cc-pvdz with RPAX."""

        emb = rpa.RPA(testsystems.lih_ccpvdz.rhf())
        emb.kernel("rpax")

        known_values = {"e_tot": -8.021765296851472}

        self._test_energy(emb, known_values)

    def test_lih_ccpvdz_dRPA(self):
        """Tests for LiH cc-pvdz with dPRA."""

        emb = rpa.RPA(testsystems.lih_ccpvdz.rhf())
        emb.kernel("drpa")

        known_values = {"e_tot": -8.015594007709575}

        self._test_energy(emb, known_values)

        emb = rpa.ssRPA(testsystems.lih_ccpvdz.rhf())
        emb.kernel()

        self._test_energy(emb, known_values)

    def test_water_cation_ssurpa_spin_dependent_eris(self):
        """Tests ssURPA with spin-dependent ERIs, given as a tuple (aa, ab, bb)."""

        uhf = testsystems.water_cation_sto3g.uhf()
        emb = rpa.ssURPA(uhf)
        emb.kernel()

        mf = copy.copy(uhf)
        eri = pyscf.ao2mo.restore(1, uhf._eri, uhf.mol.nao)
        mf._eri = (eri, eri, eri)
        emb_tuple = rpa.ssURPA(mf)
        emb_tuple.kernel()

        self.assertAlmostEqual(emb_tuple.e_corr, emb.e_corr, self.PLACES)


if __name__ == "__main__":
    print("Running %s" % __file__)
    unittest.main()
