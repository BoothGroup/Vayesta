import unittest
import numpy as np
import vayesta
import vayesta.ewf
from vayesta.core.util import cache
from vayesta.tests import testsystems
from vayesta.tests.common import TestCase


class Test_Restricted(TestCase):

    system = testsystems.h2_dz

    @classmethod
    def setUpClass(cls):
        cls.rhf = cls.system.rhf()
        cls.uhf = cls.system.uhf()
        cls.rcc = cls.system.rccsd()
        cls.ucc = cls.system.uccsd()

    @classmethod
    def tearDownClass(cls):
        del cls.rcc
        del cls.ucc
        cls.remb.cache_clear()
        cls.uemb.cache_clear()

    @classmethod
    @cache
    def remb(cls, bno_threshold, run_kernel=True):
        remb = vayesta.ewf.EWF(cls.rhf, bno_threshold=bno_threshold, solve_lambda=True)
        if run_kernel:
            remb.kernel()
        return remb

    @classmethod
    @cache
    def uemb(cls, bno_threshold, run_kernel=True):
        uemb = vayesta.ewf.EWF(cls.uhf, bno_threshold=bno_threshold, solve_lambda=True)
        if run_kernel:
            uemb.kernel()
        return uemb

    def test_mf(self):
        remb = self.remb(np.inf, run_kernel=False)
        uemb = self.uemb(np.inf, run_kernel=False)
        ssz_r = remb.get_corrfunc_mf('Sz,Sz')
        ssz_u = uemb.get_corrfunc_mf('Sz,Sz')
        self.assertAllclose(ssz_r, ssz_u, atol=1e-6)
        # OLD
        ssz_old_u = uemb.get_atomic_ssz_mf()
        self.assertAllclose(ssz_old_u, ssz_u, atol=1e-6)

    def test_full_bath(self):
        eta = -1
        remb = self.remb(eta)
        uemb = self.uemb(eta)

        ssz = rssz = remb.get_atomic_ssz()
        ussz = uemb.get_atomic_ssz()
        self.assertAllclose(rssz, ssz, atol=1e-6)
        self.assertAllclose(ussz, ssz, atol=1e-6)

        # NEW
        corr_r = remb.get_corrfunc('Sz,Sz')
        corr_u = uemb.get_corrfunc('Sz,Sz')
        self.assertAllclose(corr_r, ssz, atol=1e-6)
        self.assertAllclose(corr_u, ssz, atol=1e-6)

        # Full 2DMs
        rdm2 = remb.make_rdm2()
        udm2 = uemb.make_rdm2()
        rssz = remb.get_atomic_ssz(dm2=rdm2)
        ussz = uemb.get_atomic_ssz(dm2=udm2)
        self.assertAllclose(rssz, ssz, atol=1e-6)
        self.assertAllclose(ussz, ssz, atol=1e-6)

        # NEW
        corr_r = remb.get_corrfunc('Sz,Sz', dm2=rdm2)
        corr_u = uemb.get_corrfunc('Sz,Sz', dm2=udm2)
        self.assertAllclose(corr_r, ssz, atol=1e-6)
        self.assertAllclose(corr_u, ssz, atol=1e-6)

    def test_finite_bath(self):
        eta = +1
        remb = self.remb(eta)
        uemb = self.uemb(eta)

        ssz = rssz = remb.get_atomic_ssz()
        ussz = uemb.get_atomic_ssz()
        self.assertAllclose(rssz, ssz, atol=1e-6)
        self.assertAllclose(ussz, ssz, atol=1e-6)

        # NEW
        corr_r = remb.get_corrfunc('Sz,Sz')
        corr_u = uemb.get_corrfunc('Sz,Sz')
        self.assertAllclose(corr_r, ssz, atol=1e-6)
        self.assertAllclose(corr_u, ssz, atol=1e-6)

        # Full 2DMs
        rdm2 = remb.make_rdm2()
        udm2 = uemb.make_rdm2()
        rssz = remb.get_atomic_ssz(dm2=rdm2)
        ussz = uemb.get_atomic_ssz(dm2=udm2)
        self.assertAllclose(rssz, ssz, atol=1e-6)
        self.assertAllclose(ussz, ssz, atol=1e-6)

        # NEW
        corr_r = remb.get_corrfunc('Sz,Sz', dm2=rdm2)
        corr_u = uemb.get_corrfunc('Sz,Sz', dm2=udm2)
        self.assertAllclose(corr_r, ssz, atol=1e-6)
        self.assertAllclose(corr_u, ssz, atol=1e-6)


class Test_Unrestricted(TestCase):

    system = testsystems.water_cation_631g

    @classmethod
    def setUpClass(cls):
        cls.uhf = cls.system.uhf()
        cls.ucc = cls.system.uccsd()

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

        ssz = uemb.get_atomic_ssz()

        # Full 2DMs
        udm2 = uemb.make_rdm2()
        ssz_2 = uemb.get_atomic_ssz(dm2=udm2, dm2_with_dm1=True)
        self.assertAllclose(ssz_2, ssz, atol=1e-6)

        # NEW
        ssz_3 = uemb.get_corrfunc('Sz,Sz', dm2=udm2, dm2_with_dm1=True)
        self.assertAllclose(ssz_3, ssz, atol=1e-6)


if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
