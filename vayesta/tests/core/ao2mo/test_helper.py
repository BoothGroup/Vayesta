import unittest
import numpy as np
import pyscf
import pyscf.ao2mo
import vayesta
from vayesta.core.ao2mo import helper
from vayesta.tests.common import TestCase
from vayesta.tests import testsystems


class Test_RHF(TestCase):

    system = testsystems.water_631g

    @classmethod
    def setUpClass(cls):
        cls.mf = cls.system.rhf()
        cls.cc = cls.system.rccsd()
        cls.eris = cls.cc.ao2mo()
        nmo = cls.mf.mo_coeff.shape[-1]
        cls.g_ref = pyscf.ao2mo.kernel(cls.mf.mol, cls.mf.mo_coeff, compact=False).reshape(4*[nmo])

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
        e_ref = np.einsum('ijkl,ijkl', dm2, self.g_ref)
        e_test = helper.contract_dm2_eris_rhf(dm2, self.eris)
        self.assertAllclose(e_ref, e_test)


class Test_UHF(TestCase):

    system = testsystems.water_cation_631g

    @classmethod
    def setUpClass(cls):
        cls.mf = cls.system.uhf()
        cls.cc = cls.system.uccsd()
        cls.eris = cls.cc.ao2mo()
        g_ref_aa = pyscf.ao2mo.kernel(cls.mf.mol, cls.mf.mo_coeff[0], compact=False)
        g_ref_bb = pyscf.ao2mo.kernel(cls.mf.mol, cls.mf.mo_coeff[1], compact=False)
        g_ref_ab = pyscf.ao2mo.kernel(cls.mf.mol, 2*[cls.mf.mo_coeff[0]] + 2*[cls.mf.mo_coeff[1]], compact=False)
        nmoa = cls.mf.mo_coeff[0].shape[-1]
        nmob = cls.mf.mo_coeff[1].shape[-1]
        cls.g_ref_aa = g_ref_aa.reshape(4*[nmoa])
        cls.g_ref_ab = g_ref_ab.reshape(2*[nmoa] + 2*[nmob])
        cls.g_ref_bb = g_ref_bb.reshape(4*[nmob])

    @classmethod
    def tearDownClass(cls):
        del cls.mf
        del cls.cc
        del cls.eris
        del cls.g_ref_aa
        del cls.g_ref_ab
        del cls.g_ref_bb

    def test_get_full_array_ccsd(self):
        g_test_aa, g_test_ab, g_test_bb = helper.get_full_array_uhf(self.eris)
        self.assertAllclose(self.g_ref_aa, g_test_aa)
        self.assertAllclose(self.g_ref_ab, g_test_ab)
        self.assertAllclose(self.g_ref_bb, g_test_bb)

    def test_dm2_eris_ccsd(self):
        dm2 = self.cc.make_rdm2()
        e_ref = (np.einsum('ijkl,ijkl', dm2[0], self.g_ref_aa)
               + np.einsum('ijkl,ijkl', dm2[1], self.g_ref_ab)*2
               + np.einsum('ijkl,ijkl', dm2[2], self.g_ref_bb))
        e_test = helper.contract_dm2_eris_uhf(dm2, self.eris)
        self.assertAllclose(e_ref, e_test)


if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
