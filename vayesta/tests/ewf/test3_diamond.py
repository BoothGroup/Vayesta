
import unittest

import numpy as np

import vayesta
import vayesta.ewf
from vayesta.core.util import cache
from vayesta.tests import testsystems
from vayesta.tests.common import TestCase

class Test(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.mf = testsystems.diamond_sto3g_k333.rhf()

    @classmethod
    def tearDownClass(cls):
        del cls.mf
        cls.emb.cache_clear()

    @classmethod
    @cache
    def emb(cls, bno_threshold):
        emb = vayesta.ewf.EWF(cls.mf, bno_threshold=bno_threshold, solve_lambda=True)
        emb.kernel()
        return emb

    def test_dm1_symmetry(self):
        emb = self.emb(1e-4)
        dm1_nosym = emb._make_rdm1_ccsd_global_wf(use_sym=False)
        dm1_sym = emb._make_rdm1_ccsd_global_wf(use_sym=True)
        self.assertAllclose(dm1_sym, dm1_nosym)

if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
