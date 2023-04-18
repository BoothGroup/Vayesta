import pytest
import unittest
import numpy as np
import vayesta
import vayesta.ewf
from vayesta.core.util import cache
from vayesta.tests import testsystems
from vayesta.tests.common import TestCase


class TestCCSD(TestCase):

    solver = 'CCSD'
    targets = [9.0, 0.7, 0.3]

    @classmethod
    def setUpClass(cls):
        cls.mf = testsystems.water_631g.rhf()

    @classmethod
    @cache
    def emb(cls, bno_threshold):
        emb = vayesta.ewf.EWF(cls.mf, solver=cls.solver, bno_threshold=bno_threshold)
        with emb.sao_fragmentation() as f:
            f.add_atomic_fragment(0, nelectron_target=cls.targets[0], nelectron_target_atol=1e-7, nelectron_target_rtol=0)
            f.add_atomic_fragment(1, nelectron_target=cls.targets[1], nelectron_target_atol=1e-7, nelectron_target_rtol=0)
            f.add_atomic_fragment(2, nelectron_target=cls.targets[2], nelectron_target_atol=1e-7, nelectron_target_rtol=0)
        emb.kernel()
        return emb

    @classmethod
    def tearDownClass(cls):
        del cls.mf
        cls.emb.cache_clear()

    def _test_nelectron_target(self, bno_threshold):
        emb = self.emb(bno_threshold)
        for idx, label in enumerate(['O', 'H', 'H']):
            dm1 = emb.fragments[idx].results.wf.make_rdm1(ao_basis=True)
            pop = emb.pop_analysis(dm1)
            aos = [(('%d %s' % (idx, label)) in ao) for ao in emb.mol.ao_labels()]
            if emb.spinsym == 'restricted':
                ne = sum(pop[aos])
            elif emb.spinsym == 'unrestricted':
                ne = sum(pop[0][aos]) + sum(pop[1][aos])
            self.assertAllclose(ne, self.targets[idx])

    def test_nelectron_target_dmet_bath(self):
        return self._test_nelectron_target(np.inf)

    #def test_nelectron_target_full_bath(self):
    #    return self._test_nelectron_target(-np.inf)


class TestUCCSD(TestCCSD):

    targets = [8.0, 0.7, 0.3]

    @classmethod
    def setUpClass(cls):
        cls.mf = testsystems.water_cation_631g.uhf()


class TestFCI(TestCCSD):

    solver = 'FCI'

    @classmethod
    def setUpClass(cls):
        cls.mf = testsystems.water_sto3g.rhf()

class TestUFCI(TestUCCSD):

    solver = 'FCI'

    @classmethod
    def setUpClass(cls):
        cls.mf = testsystems.water_cation_sto3g.rhf()

if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
