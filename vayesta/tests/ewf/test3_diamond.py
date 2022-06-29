
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
        emb = vayesta.ewf.EWF(cls.mf, bath_options=dict(threshold=bno_threshold),
                solver_options=dict(solve_lambda=True))
        emb.kernel()
        return emb

    @classmethod
    @cache
    def emb_rotsym(cls, bno_threshold):
        emb = vayesta.ewf.EWF(cls.mf, bath_options=dict(threshold=bno_threshold),
                solver_options=dict(solve_lambda=True))
        emb.symmetry.add_rotation(2, axis=[1, 0, -1], center=[1/8, 1/8, 1/8], unit='latvec')
        with emb.fragmentation() as frag:
            frag.add_atomic_fragment(0)
        emb.kernel()
        return emb

    def test_dm1_symmetry(self):
        emb = self.emb(1e-4)
        dm1_nosym = emb._make_rdm1_ccsd_global_wf(use_sym=False)
        dm1_sym = emb._make_rdm1_ccsd_global_wf(use_sym=True)
        self.assertAllclose(dm1_sym, dm1_nosym)

    def test_dm1_rotsymmetry(self):
        emb_rotsym = self.emb_rotsym(1e-4)
        emb = self.emb(1e-4)
        dm1_nosym = emb._make_rdm1_ccsd_global_wf()
        dm1_sym = emb_rotsym._make_rdm1_ccsd_global_wf()
        self.assertAllclose(dm1_sym, dm1_nosym)

    def test_corrfunc_dndn_symmetry(self):
        emb = self.emb(1e-4)
        corr_nosym = emb.get_corrfunc('dN,dN', use_symmetry=False)
        corr_sym = emb.get_corrfunc('dN,dN', use_symmetry=True)
        self.assertAllclose(corr_sym, corr_nosym)

    def test_corrfunc_szsz_symmetry(self):
        emb = self.emb(1e-4)
        corr_nosym = emb.get_corrfunc('Sz,Sz', use_symmetry=False)
        corr_sym = emb.get_corrfunc('Sz,Sz', use_symmetry=True)
        self.assertAllclose(corr_sym, corr_nosym)

if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
