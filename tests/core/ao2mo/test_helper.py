import unittest
import numpy as np
import pyscf
import pyscf.ao2mo
from vayesta.core.ao2mo import helper
from vayesta.core.vpyscf import uccsd_rdm
from tests.common import TestCase
from tests import systems


class Test_RHF(TestCase):
    system = systems.water_631g

    @classmethod
    def setUpClass(cls):
        cls.mf = cls.system.rhf()
        cls.cc = cls.system.rccsd()
        cls.eris = cls.cc.ao2mo()
        nmo = cls.mf.mo_coeff.shape[-1]
        cls.g_ref = pyscf.ao2mo.kernel(cls.mf.mol, cls.mf.mo_coeff, compact=False).reshape(4 * [nmo])

    @classmethod
    def tearDownClass(cls):
        del cls.mf
        del cls.cc
        del cls.eris
        del cls.g_ref

    def test_get_full_array_ccsd(self):
        g_test = helper.get_full_array_rhf(self.eris)
        self.assertAllclose(self.g_ref, g_test)

    def test_dm2_eris_ccsd(self):
        dm2 = self.cc.make_rdm2()
        e_ref = np.einsum("ijkl,ijkl", dm2, self.g_ref)
        e_test = helper.contract_dm2_eris_rhf(dm2, self.eris)
        self.assertAllclose(e_ref, e_test)


class Test_UHF(TestCase):
    system = systems.water_cation_631g

    @classmethod
    def setUpClass(cls):
        cls.mf = cls.system.uhf()
        cls.cc = cls.system.uccsd()
        cls.eris = cls.cc.ao2mo()
        g_ref_aa = pyscf.ao2mo.kernel(cls.mf.mol, cls.mf.mo_coeff[0], compact=False)
        g_ref_bb = pyscf.ao2mo.kernel(cls.mf.mol, cls.mf.mo_coeff[1], compact=False)
        g_ref_ab = pyscf.ao2mo.kernel(cls.mf.mol, 2 * [cls.mf.mo_coeff[0]] + 2 * [cls.mf.mo_coeff[1]], compact=False)
        nmoa = cls.mf.mo_coeff[0].shape[-1]
        nmob = cls.mf.mo_coeff[1].shape[-1]
        cls.g_ref_aa = g_ref_aa.reshape(4 * [nmoa])
        cls.g_ref_ab = g_ref_ab.reshape(2 * [nmoa] + 2 * [nmob])
        cls.g_ref_bb = g_ref_bb.reshape(4 * [nmob])

        # cls.dm2 = cls.cc.make_rdm2()
        cls.dm2 = uccsd_rdm.make_rdm2(cls.cc, cls.cc.t1, cls.cc.t2, cls.cc.l1, cls.cc.l2, with_dm1=False)
        cls.e_ref = (
            np.einsum("ijkl,ijkl", cls.dm2[0], cls.g_ref_aa)
            + np.einsum("ijkl,ijkl", cls.dm2[1], cls.g_ref_ab) * 2
            + np.einsum("ijkl,ijkl", cls.dm2[2], cls.g_ref_bb)
        )

    @classmethod
    def tearDownClass(cls):
        del cls.mf
        del cls.cc
        del cls.eris
        del cls.g_ref_aa
        del cls.g_ref_ab
        del cls.g_ref_bb
        del cls.dm2
        del cls.e_ref

    def test_get_full_array_ccsd(self):
        g_test_aa, g_test_ab, g_test_bb = helper.get_full_array_uhf(self.eris)
        self.assertAllclose(self.g_ref_aa, g_test_aa)
        self.assertAllclose(self.g_ref_ab, g_test_ab)
        self.assertAllclose(self.g_ref_bb, g_test_bb)

    def test_dm2_eris_ccsd(self):
        e_test = helper.contract_dm2_eris_uhf(self.dm2, self.eris)
        self.assertAllclose(e_test, self.e_ref)

    def test_dm2intermeds_eris_ccsd(self):
        cc = self.cc
        d2 = uccsd_rdm._gamma2_intermediates(cc, cc.t1, cc.t2, cc.l1, cc.l2)
        e_test = helper.contract_dm2intermeds_eris_uhf(d2, self.eris)
        self.assertAllclose(e_test, self.e_ref)


if __name__ == "__main__":
    print("Running %s" % __file__)
    unittest.main()
