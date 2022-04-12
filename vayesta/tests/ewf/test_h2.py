import unittest

import numpy as np

import vayesta
import vayesta.ewf
from vayesta.core.util import *
from vayesta.tests import testsystems

class TestH2(unittest.TestCase):

    rhf = testsystems.h2_dz.rhf()
    rccsd = testsystems.h2_dz.rccsd()

    @cache
    def emb(self, bno_threshold):
        self.emb = vayesta.ewf.EWF(self.rhf, bno_threshold=bno_threshold, solve_lambda=True)
        self.emb.iao_fragmentation()
        self.emb.add_all_atomic_fragments()
        self.emb.kernel()
        return self.emb

    def test_energy(self):
        emb = self.emb(-1)
        self.assertTrue(abs(emb.e_tot - self.rccsd.e_tot) < 1e-8)
        self.assertTrue(abs(emb.e_corr - self.rccsd.e_corr) < 1e-8)

    def test_t_amplitudes(self):
        emb = self.emb(-1)
        t1 = emb.get_global_t1()
        t2 = emb.get_global_t2()
        self.assertIsNone(np.testing.assert_allclose(t1, self.rccsd.t1, atol=1e-8))
        self.assertIsNone(np.testing.assert_allclose(t2, self.rccsd.t2, atol=1e-8))

    def test_l_amplitudes(self):
        emb = self.emb(-1)
        l1 = emb.get_global_l1()
        l2 = emb.get_global_l2()
        self.assertIsNone(np.testing.assert_allclose(l1, self.rccsd.l1, atol=1e-8))
        self.assertIsNone(np.testing.assert_allclose(l2, self.rccsd.l2, atol=1e-8))

    def test_cluster_dm1(self):
        emb = self.emb(-1)
        t = np.dot(self.rhf.get_ovlp(), self.rhf.mo_coeff)
        dm1_exact = self.rccsd.make_rdm1()
        for x in emb.fragments:
            dm1 = x.results.wf.make_rdm1(ao_basis=True)
            dm1 = np.linalg.multi_dot((t.T, dm1, t))
            self.assertIsNone(np.testing.assert_allclose(dm1, dm1_exact, atol=1e-8))

    def test_cluster_dm2(self):
        emb = self.emb(-1)
        t = np.dot(self.rhf.get_ovlp(), self.rhf.mo_coeff)
        dm2_exact = self.rccsd.make_rdm2()
        for x in emb.fragments:
            dm2 = x.results.wf.make_rdm2(ao_basis=True)
            dm2 = einsum('ijkl,iI,jJ,kK,lL->IJKL', dm2, t, t, t, t)
            self.assertIsNone(np.testing.assert_allclose(dm2, dm2_exact, atol=1e-8))

    def test_global_dm1(self):
        emb = self.emb(-1)
        dm1 = emb.make_rdm1_ccsd()
        dm1_exact = self.rccsd.make_rdm1()
        self.assertIsNone(np.testing.assert_allclose(dm1, dm1_exact, atol=1e-8))


if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
