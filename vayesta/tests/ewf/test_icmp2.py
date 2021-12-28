import unittest
import numpy as np

import pyscf
import pyscf.cc

from vayesta import ewf
from vayesta.tests.cache import moles, cells
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

    def test_rhf_solid(self):
        mf = cells['h2_cp_k211']['rhf']

        # Add each fragment:
        emb = ewf.EWF(mf)
        emb.iao_fragmentation()
        for atom in range(4):
            emb.add_atomic_fragment(atom, add_symmetric=False)
        emb.kernel(bno_threshold=1e-4)
        e6_expected = -0.0021036512144716295
        e8_expected = -0.0021000931253910525
        e6 = emb.get_intercluster_mp2_energy(1e-6)
        e8 = emb.get_intercluster_mp2_energy(1e-8)
        self.assertAllclose(e6, e6_expected)
        self.assertAllclose(e8, e8_expected)

        # Add unit cell fragments only:
        emb = ewf.EWF(mf)
        emb.iao_fragmentation()
        emb.add_atomic_fragment(0)
        emb.add_atomic_fragment(1)
        emb.kernel(bno_threshold=1e-4)
        e6 = emb.get_intercluster_mp2_energy(1e-6)
        e8 = emb.get_intercluster_mp2_energy(1e-8)
        self.assertAllclose(e6, e6_expected)
        self.assertAllclose(e8, e8_expected)

        # Add symmetry unique fragment only:
        emb = ewf.EWF(mf)
        emb.iao_fragmentation()
        emb.add_atomic_fragment(0, sym_factor=2)
        emb.kernel(bno_threshold=1e-4)
        e6 = emb.get_intercluster_mp2_energy(1e-6)
        e8 = emb.get_intercluster_mp2_energy(1e-8)
        self.assertAllclose(e6, e6_expected)
        self.assertAllclose(e8, e8_expected)

        # Compare to UHF
        uhf = cells['h2_cp_k211']['uhf']
        emb = ewf.EWF(uhf)
        emb.iao_fragmentation()
        emb.add_atomic_fragment(0)
        emb.add_atomic_fragment(1)
        emb.kernel(bno_threshold=1e-4)
        e6 = emb.get_intercluster_mp2_energy(1e-6)
        e8 = emb.get_intercluster_mp2_energy(1e-8)
        self.assertAllclose(e6, e6_expected)
        self.assertAllclose(e8, e8_expected)

        # Test supercell
        mf = cells['h2_cp_g211']['rhf']
        emb = ewf.EWF(mf)
        emb.iao_fragmentation()
        for atom in range(4):
            emb.add_atomic_fragment(atom, add_symmetric=False)
        emb.kernel(bno_threshold=1e-4)
        e6 = emb.get_intercluster_mp2_energy(1e-6)
        e8 = emb.get_intercluster_mp2_energy(1e-8)
        self.assertAllclose(e6/2, e6_expected)
        self.assertAllclose(e8/2, e8_expected)

    def test_uhf_solid(self):
        mf = cells['h3_cp_k211']['uhf']

        # Add each fragment:
        emb = ewf.EWF(mf)
        emb.iao_fragmentation()
        for atom in range(6):
            emb.add_atomic_fragment(atom, add_symmetric=False)
        emb.kernel(bno_threshold=1e-4)
        e6_expected = -0.016221464358847706 / 2
        e8_expected = -0.016297794448082218 / 2
        e6 = emb.get_intercluster_mp2_energy(1e-6)
        e8 = emb.get_intercluster_mp2_energy(1e-8)
        self.assertAllclose(e6, e6_expected)
        self.assertAllclose(e8, e8_expected)

        # Add unit cell fragments only:
        emb = ewf.EWF(mf)
        emb.iao_fragmentation()
        emb.add_atomic_fragment(0)
        emb.add_atomic_fragment(1)
        emb.add_atomic_fragment(2)
        emb.kernel(bno_threshold=1e-4)
        e6 = emb.get_intercluster_mp2_energy(1e-6)
        e8 = emb.get_intercluster_mp2_energy(1e-8)
        self.assertAllclose(e6, e6_expected)
        self.assertAllclose(e8, e8_expected)


if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
