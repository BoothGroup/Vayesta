import unittest
import numpy as np

import vayesta
from vayesta import ewf

from vayesta.core.util import cache
from vayesta.tests.common import TestCase
from vayesta.tests import testsystems


class ICMP2_Test(TestCase):

    @classmethod
    def tearDownClass(cls):
        del cls.mf
        cls.emb.cache_clear()

    @classmethod
    @cache()
    def emb(cls, bno_threshold):
        emb = vayesta.ewf.EWF(cls.mf, bno_threshold=bno_threshold, solver='MP2')
        emb.kernel()
        return emb

    @classmethod
    @cache()
    def emb_nosym(cls, bno_threshold):
        emb = vayesta.ewf.EWF(cls.mf, bno_threshold=bno_threshold, solver='MP2')
        with emb.iao_fragmentation(add_symmetric=False) as f:
            for natom in range(emb.mol.natm):
                f.add_atomic_fragment(natom)
        emb.kernel()
        return emb

class ICMP2_RHF_Test(ICMP2_Test):

    @classmethod
    def setUpClass(cls):
        cls.mf = testsystems.water_631g_df.rhf()

    def test_dmet_bath(self):
        emb = self.emb(np.inf)
        e_ic = emb.get_intercluster_mp2_energy(bno_threshold_vir=1e-3)
        self.assertAllclose(e_ic, -0.05927010355296033)
        e_ic = emb.get_intercluster_mp2_energy(bno_threshold_vir=1e-5)
        self.assertAllclose(e_ic, -0.0721498800072952)

    def test_full_bath(self):
        emb = self.emb(-np.inf)
        e_ic = emb.get_intercluster_mp2_energy(bno_threshold_vir=1e-4)
        self.assertAllclose(e_ic, 0)

class ICMP2_UHF_Test(ICMP2_Test):

    @classmethod
    def setUpClass(cls):
        cls.mf = testsystems.water_cation_631g_df.uhf()

    def test_dmet_bath(self):
        emb = self.emb(np.inf)
        e_ic = emb.get_intercluster_mp2_energy(1e-3)
        self.assertAllclose(e_ic, -0.03883469123728171)
        e_ic = emb.get_intercluster_mp2_energy(1e-5)
        self.assertAllclose(e_ic, -0.048787144599376324)

    def test_full_bath(self):
        emb = self.emb(-np.inf)
        e_ic = emb.get_intercluster_mp2_energy(1e-4)
        self.assertAllclose(e_ic, 0)

class ICMP2_RHF_PBC_Test(ICMP2_Test):

    @classmethod
    def setUpClass(cls):
        cls.mf = testsystems.h2_sto3g_k311.rhf()

    def test_dmet_bath(self):
        emb = self.emb(np.inf)
        e_ic = emb.get_intercluster_mp2_energy(1e-1)
        self.assertAllclose(e_ic, -0.0002008447130808661)
        e_ic = emb.get_intercluster_mp2_energy(1e-8)
        self.assertAllclose(e_ic, -0.00020293482504415167)

    def test_dmet_bath_nosym(self):
        emb = self.emb_nosym(np.inf)
        e_ic = emb.get_intercluster_mp2_energy(1e-1)
        self.assertAllclose(e_ic, -0.0002008447130808661)
        e_ic = emb.get_intercluster_mp2_energy(1e-8)
        self.assertAllclose(e_ic, -0.00020293482504415167)

class ICMP2_UHF_PBC_Test(ICMP2_Test):

    @classmethod
    def setUpClass(cls):
        cls.mf = testsystems.h3_sto3g_k311.uhf()

    def test_dmet_bath(self):
        emb = self.emb(np.inf)
        e_ic = emb.get_intercluster_mp2_energy(1e-1)
        self.assertAllclose(e_ic, -0.0013459605792552982)
        e_ic = emb.get_intercluster_mp2_energy(1e-8)
        self.assertAllclose(e_ic, -0.0010737526376497356)

    def test_dmet_bath_nosym(self):
        emb = self.emb_nosym(np.inf)
        e_ic = emb.get_intercluster_mp2_energy(1e-1)
        self.assertAllclose(e_ic, -0.0013459605792552982)
        e_ic = emb.get_intercluster_mp2_energy(1e-8)
        self.assertAllclose(e_ic, -0.0010737526376497356)


if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
