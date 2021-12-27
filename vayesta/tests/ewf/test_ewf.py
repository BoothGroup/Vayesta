import unittest
import numpy as np

import pyscf
import pyscf.cc

from vayesta import ewf
from vayesta.tests.cache import moles
from vayesta.tests.common import TestCase

class InterclusterMP2_Test(TestCase):

    def test_rhf(self):
        mol = moles['h2o_ccpvdz_df']['mol']
        mf = moles['h2o_ccpvdz_df']['rhf']
        emb = ewf.EWF(mf)

        # Test finite bath
        emb.kernel(bno_threshold=np.inf)
        e4 = emb.get_intercluster_mp2_energy(1e-4)
        e6 = emb.get_intercluster_mp2_energy(1e-6)
        e8 = emb.get_intercluster_mp2_energy(1e-8)
        e4_expected = -0.1415720217653774
        e6_expected = -0.1463770484732253
        e8_expected = -0.1464956750957701
        self.assertAllclose(e4, e4_expected)
        self.assertAllclose(e6, e6_expected)
        self.assertAllclose(e8, e8_expected)

        # Test full bath
        emb.kernel(bno_threshold=-np.inf)
        e4 = emb.get_intercluster_mp2_energy(1e-4)
        e6 = emb.get_intercluster_mp2_energy(1e-6)
        e8 = emb.get_intercluster_mp2_energy(1e-8)
        self.assertAllclose(e4, 0)
        self.assertAllclose(e6, 0)
        self.assertAllclose(e8, 0)

    def test_uhf(self):
        mol = moles['h2o_ccpvdz_df']['mol']
        mf = moles['h2o_ccpvdz_df']['uhf']
        emb = ewf.EWF(mf)

        # Test finite bath
        emb.kernel(bno_threshold=np.inf)
        e4 = emb.get_intercluster_mp2_energy(1e-4)
        e6 = emb.get_intercluster_mp2_energy(1e-6)
        e8 = emb.get_intercluster_mp2_energy(1e-8)
        e4_expected = -0.1415720217653774
        e6_expected = -0.1463770484732253
        e8_expected = -0.1464956750957701
        self.assertAllclose(e4, e4_expected)
        self.assertAllclose(e6, e6_expected)
        self.assertAllclose(e8, e8_expected)

        # Test full bath
        emb.kernel(bno_threshold=-np.inf)
        e4 = emb.get_intercluster_mp2_energy(1e-4)
        e6 = emb.get_intercluster_mp2_energy(1e-6)
        e8 = emb.get_intercluster_mp2_energy(1e-8)
        self.assertAllclose(e4, 0)
        self.assertAllclose(e6, 0)
        self.assertAllclose(e8, 0)


if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
