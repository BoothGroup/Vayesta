import unittest
import numpy as np
import vayesta
import vayesta.ewf
from vayesta.core.util import cache
from vayesta.tests import testsystems
from vayesta.tests.common import TestCase


class TestTwoElectron(TestCase):

    system = testsystems.h2_ccpvdz_df
    e_ref = {"mrpa":-1.123779303361342, "crpa":-1.1237769151822752}

    @classmethod
    def setUpClass(cls):
        cls.mf = cls.system.uhf()

    @classmethod
    def tearDownClass(cls):
        del cls.mf
        cls.emb.cache_clear()

    @classmethod
    @cache
    def emb(cls, bno_threshold, solver, screening):
        emb = vayesta.ewf.EWF(cls.mf, bath_options=dict(threshold=bno_threshold), solver=solver,
                screening=screening, solver_options=dict(conv_tol=1e-12))
        emb.kernel()
        return emb

    def test_ccsd_mrpa(self):
        emb = self.emb(np.inf, 'CCSD', 'mrpa')
        emb.kernel()
        self.assertAllclose(emb.e_tot, self.e_ref['mrpa'])

    def test_fci_mrpa(self):
        emb = self.emb(np.inf, 'FCI', 'mrpa')
        emb.kernel()
        self.assertAllclose(emb.e_tot, self.e_ref['mrpa'])

    def test_ccsd_crpa(self):
        emb = self.emb(np.inf, 'CCSD', 'crpa')
        emb.kernel()
        self.assertAllclose(emb.e_tot, self.e_ref['crpa'])

    def test_fci_crpa(self):
        emb = self.emb(np.inf, 'FCI', 'crpa')
        emb.kernel()
        self.assertAllclose(emb.e_tot, self.e_ref['crpa'])

class TestTwoHole(TestTwoElectron):

    system = testsystems.f2_sto6g_df
    e_ref = {"mrpa":-197.84155758368854, "crpa":-197.83928243962046}

if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
