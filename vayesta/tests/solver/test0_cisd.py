import unittest
import numpy as np

import pyscf
import pyscf.ci

import vayesta
import vayesta.ewf

from vayesta.tests.common import TestCase
from vayesta.tests import testsystems


class TestSolvers(TestCase):

    def _test(self, key):
        mf = getattr(getattr(testsystems, key[0]), key[1])()

        solver_opts = dict(conv_tol=1e-10)
        emb = vayesta.ewf.EWF(mf, solver='CISD', bath_type='full', solver_options=solver_opts)
        emb.kernel()

        ci = pyscf.ci.CISD(mf)
        ci.conv_tol = 1e-10
        ci.kernel()

        self.assertAlmostEqual(emb.e_corr, ci.e_corr)
        self.assertAlmostEqual(emb.e_tot, ci.e_tot)

    def test_rcisd_h2(self):
        return self._test(('h2_ccpvdz', 'rhf'))

    def test_ucisd_h3(self):
        return self._test(('h3_ccpvdz', 'uhf'))


if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
