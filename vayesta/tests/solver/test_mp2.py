import unittest
import numpy as np

import pyscf
import pyscf.mp

import vayesta
import vayesta.ewf

from vayesta.tests import cache


class TestSolvers(unittest.TestCase):

    def _test(self, key):
        mf = cache.moles[key[0]][key[1]]

        emb = vayesta.ewf.EWF(mf, solver='MP2', bath_type='full')
        emb.kernel()

        mp2 = pyscf.mp.MP2(mf)
        mp2.kernel()

        self.assertAlmostEqual(emb.e_corr, mp2.e_corr)
        self.assertAlmostEqual(emb.e_tot, mp2.e_tot)

    def test_rmp2_h2o(self):
        return self._test(('h2o_ccpvdz', 'rhf'))

    def test_rmp2_h2o_df(self):
        return self._test(('h2o_ccpvdz_df', 'rhf'))

    def test_ump2_h2o(self):
        return self._test(('no2_ccpvdz', 'uhf'))

    def test_ump2_h2o_df(self):
        return self._test(('no2_ccpvdz_df', 'uhf'))


if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
