import unittest
import numpy as np
import vayesta
import vayesta.ewf
from vayesta.core.util import cache
from vayesta.tests import testsystems
from vayesta.tests.common import TestCase


class TestTwoElectron(TestCase):

    system = testsystems.h2_ccpvdz_df
    e_ref = -1.123779303361342

    @classmethod
    def setUpClass(cls):
        cls.mf = cls.system.uhf()

    @classmethod
    def tearDownClass(cls):
        del cls.mf
        cls.emb.cache_clear()

    @classmethod
    @cache
    def emb(cls, bno_threshold, solver):
        emb = vayesta.ewf.EWF(cls.mf, bath_options=dict(threshold=bno_threshold), solver=solver,
                screening='mrpa', solver_options=dict(conv_tol=1e-10))
        emb.kernel()
        return emb

    def test_ccsd(self):
        emb = self.emb(np.inf, 'CCSD')
        emb.kernel()
        print(emb.e_tot)
        self.assertAllclose(emb.e_tot, self.e_ref)

    def test_fci(self):
        emb = self.emb(np.inf, 'FCI')
        emb.kernel()
        self.assertAllclose(emb.e_tot, self.e_ref)

class TestTwoHole(TestTwoElectron):

    system = testsystems.f2_sto6g_df
    e_ref = -197.84155758368854

if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
