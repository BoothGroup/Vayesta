import pytest
import unittest
import numpy as np

import pyscf
import pyscf.mp

import vayesta
import vayesta.ewf

from vayesta.tests.common import TestCase
from vayesta.tests import testsystems


@pytest.mark.fast
class TestSolvers(unittest.TestCase):

    def _test(self, key):
        mf = getattr(getattr(testsystems, key[0]), key[1])()

        emb = vayesta.ewf.EWF(mf, solver='MP2', bath_type='full')
        emb.kernel()

        mp2 = pyscf.mp.MP2(mf)
        mp2.kernel()

        self.assertAlmostEqual(emb.e_corr, mp2.e_corr)
        self.assertAlmostEqual(emb.e_tot, mp2.e_tot)

    def test_rmp2_h2o(self):
        return self._test(('water_ccpvdz', 'rhf'))

    def test_rmp2_h2o_df(self):
        return self._test(('water_ccpvdz_df', 'rhf'))

    def test_ump2_h2o(self):
        return self._test(('water_cation_631g', 'uhf'))

    def test_ump2_h2o_df(self):
        return self._test(('water_cation_631g_df', 'uhf'))


if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
