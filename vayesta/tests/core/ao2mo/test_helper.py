import unittest
import numpy as np
import pyscf
import pyscf.ao2mo
import vayesta
from vayesta.core.ao2mo import helper
from vayesta.tests.common import TestCase
from vayesta.tests import testsystems


class Test_RHF(TestCase):

    def test_get_full_array_ccsd(self):
        system = testsystems.water_631g
        mf = system.rhf()
        cc = system.rccsd()
        eris = cc.ao2mo()
        g_ref = pyscf.ao2mo.kernel(mf.mol, mf.mo_coeff, compact=False)
        nmo = mf.mo_coeff.shape[-1]
        g_ref = g_ref.reshape(4*[nmo])
        g_test = helper.get_full_array_rhf(eris)
        self.assertAllclose(g_ref, g_test)

class Test_UHF(TestCase):

    def test_get_full_array_ccsd(self):
        system = testsystems.water_cation_631g
        mf = system.uhf()
        cc = system.uccsd()
        eris = cc.ao2mo()
        g_ref_aa = pyscf.ao2mo.kernel(mf.mol, mf.mo_coeff[0], compact=False)
        g_ref_bb = pyscf.ao2mo.kernel(mf.mol, mf.mo_coeff[1], compact=False)
        g_ref_ab = pyscf.ao2mo.kernel(mf.mol, 2*[mf.mo_coeff[0]] + 2*[mf.mo_coeff[1]], compact=False)
        nmoa = mf.mo_coeff[0].shape[-1]
        nmob = mf.mo_coeff[1].shape[-1]
        g_ref_aa = g_ref_aa.reshape(4*[nmoa])
        g_ref_ab = g_ref_ab.reshape(2*[nmoa] + 2*[nmob])
        g_ref_bb = g_ref_bb.reshape(4*[nmob])
        g_test_aa, g_test_ab, g_test_bb = helper.get_full_array_uhf(eris)
        self.assertAllclose(g_ref_aa, g_test_aa)
        self.assertAllclose(g_ref_ab, g_test_ab)
        self.assertAllclose(g_ref_bb, g_test_bb)

if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
