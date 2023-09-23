import unittest
import numpy as np
import pyscf
import pyscf.ao2mo
from vayesta.core.ao2mo import helper
from vayesta.core.ao2mo import pyscf_eris
from tests.common import TestCase
from tests import testsystems


class TestRSpin(TestCase):
    system = testsystems.water_631g

    @classmethod
    def setUpClass(cls):
        cls.mf = cls.system.rhf()
        nmo = cls.mf.mo_coeff.shape[-1]
        cls.nocc = np.count_nonzero(cls.mf.mo_occ > 0)
        cls.fock = np.diag(cls.mf.mo_energy)
        cls.eris = pyscf.ao2mo.kernel(cls.mf.mol, cls.mf.mo_coeff, compact=False).reshape(4 * [nmo])

    @classmethod
    def tearDownClass(cls):
        del cls.mf, cls.fock, cls.eris

    def test_make_ccsd_eris(self):
        """Convert to _ChemistERIs and back"""
        eris_pyscf = pyscf_eris.make_ccsd_eris(self.fock, self.eris, self.nocc)
        eris = helper.get_full_array_rhf(eris_pyscf)
        self.assertAllclose(eris, self.eris, atol=1e-13, rtol=0)


class TestUSpin(TestRSpin):
    system = testsystems.water_cation_631g

    @classmethod
    def setUpClass(cls):
        cls.mf = cls.system.uhf()
        cls.nocc = (np.count_nonzero(cls.mf.mo_occ[0] > 0), np.count_nonzero(cls.mf.mo_occ[1] > 0))
        cls.fock = (np.diag(cls.mf.mo_energy[0]), np.diag(cls.mf.mo_energy[1]))
        nmoa = cls.mf.mo_coeff[0].shape[-1]
        nmob = cls.mf.mo_coeff[1].shape[-1]
        eris_aa = pyscf.ao2mo.kernel(cls.mf.mol, cls.mf.mo_coeff[0], compact=False).reshape(4 * [nmoa])
        eris_bb = pyscf.ao2mo.kernel(cls.mf.mol, cls.mf.mo_coeff[1], compact=False).reshape(4 * [nmob])
        eris_ab = pyscf.ao2mo.kernel(
            cls.mf.mol, 2 * [cls.mf.mo_coeff[0]] + 2 * [cls.mf.mo_coeff[1]], compact=False
        ).reshape(2 * [nmoa] + 2 * [nmob])
        cls.eris = (eris_aa, eris_ab, eris_bb)

    def test_make_ccsd_eris(self):
        """Convert to _ChemistERIs and back"""
        eris_pyscf = pyscf_eris.make_uccsd_eris(self.fock, self.eris, self.nocc)
        eris = helper.get_full_array_uhf(eris_pyscf)
        self.assertAllclose(eris, self.eris, atol=1e-13, rtol=0)


if __name__ == "__main__":
    print("Running %s" % __file__)
    unittest.main()
