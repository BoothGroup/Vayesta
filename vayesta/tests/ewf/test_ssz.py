import unittest

import numpy as np

import vayesta
import vayesta.ewf
from vayesta.core.util import cache
from vayesta.tests import testsystems
from vayesta.tests.common import TestCase


class Test_Water(TestCase):

    @classmethod
    def setUpClass(cls):
        #sys = testsystems.water_631g
        sys = testsystems.h2_dz
        cls.rhf = sys.rhf()
        cls.uhf = sys.uhf()
        cls.rcc = sys.rccsd()
        cls.ucc = sys.uccsd()

    @classmethod
    def tearDownClass(cls):
        del cls.rcc
        del cls.ucc
        cls.remb.cache_clear()
        cls.uemb.cache_clear()

    @classmethod
    @cache
    def remb(cls, bno_threshold):
        remb = vayesta.ewf.EWF(cls.rhf, bno_threshold=bno_threshold, solve_lambda=True)
        remb.kernel()
        return remb

    @classmethod
    @cache
    def uemb(cls, bno_threshold):
        uemb = vayesta.ewf.EWF(cls.uhf, bno_threshold=bno_threshold, solve_lambda=True)
        uemb.kernel()
        return uemb

    def test_full_bath(self):
        eta = -1
        remb = self.remb(eta)
        uemb = self.uemb(eta)

        ssz = rssz = remb.get_atomic_ssz()
        ussz = uemb.get_atomic_ssz()
        self.assertAllclose(rssz, ssz, atol=1e-6)
        self.assertAllclose(ussz, ssz, atol=1e-6)

        # Full 2DMs
        rdm2 = remb.make_rdm2()
        udm2 = uemb.make_rdm2()
        rssz = remb.get_atomic_ssz(dm2=rdm2)
        ussz = uemb.get_atomic_ssz(dm2=udm2)
        self.assertAllclose(rssz, ssz, atol=1e-6)
        self.assertAllclose(ussz, ssz, atol=1e-6)

        # Cluster DMs
        rssz = remb.get_atomic_ssz(use_cluster_dms=True)
        ussz = uemb.get_atomic_ssz(use_cluster_dms=True)
        self.assertAllclose(rssz, ssz, atol=1e-6)
        self.assertAllclose(ussz, ssz, atol=1e-6)

    def test_finite_bath(self):
        eta = +1
        remb = self.remb(eta)
        uemb = self.uemb(eta)

        ssz = rssz = remb.get_atomic_ssz()
        ussz = uemb.get_atomic_ssz()
        self.assertAllclose(rssz, ssz, atol=1e-6)
        self.assertAllclose(ussz, ssz, atol=1e-6)

        # Full 2DMs
        rdm2 = remb.make_rdm2()
        udm2 = uemb.make_rdm2()
        rssz = remb.get_atomic_ssz(dm2=rdm2)
        ussz = uemb.get_atomic_ssz(dm2=udm2)
        self.assertAllclose(rssz, ssz, atol=1e-6)
        self.assertAllclose(ussz, ssz, atol=1e-6)

class Test_H2Anion(TestCase):

    @classmethod
    def setUpClass(cls):
        sys = testsystems.h2anion_dz
        cls.uhf = sys.uhf()
        cls.ucc = sys.uccsd()

    @classmethod
    def tearDownClass(cls):
        del cls.ucc
        cls.uemb.cache_clear()

    @classmethod
    @cache
    def uemb(cls, bno_threshold):
        uemb = vayesta.ewf.EWF(cls.uhf, bno_threshold=bno_threshold, solve_lambda=True)
        uemb.kernel()
        return uemb

    def test_full_bath(self):
        eta = -1
        uemb = self.uemb(eta)

        ssz = ussz = uemb.get_atomic_ssz()
        self.assertAllclose(ussz, ssz, atol=1e-6)

        # Full 2DMs
        udm2 = uemb.make_rdm2()
        ussz = uemb.get_atomic_ssz(dm2=udm2, dm2_with_dm1=True)
        self.assertAllclose(ussz, ssz, atol=1e-6)

        # Cluster DMs
        ussz = uemb.get_atomic_ssz(use_cluster_dms=True)
        self.assertAllclose(ussz, ssz, atol=1e-6)

if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
