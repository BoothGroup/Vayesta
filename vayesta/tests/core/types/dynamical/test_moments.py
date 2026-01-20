import unittest
import numpy as np

import vayesta
import vayesta.egf
from vayesta.core.util import cache, einsum
from vayesta.core.types.dynamical import GF_MomentRep, SE_MomentRep
from vayesta.tests.common import TestCase
from vayesta.tests import testsystems

try:
    import dyson
except ImportError:
    pytest.skip("Requires dyson")

class Test_FCI_Dynamical_Types(TestCase):

    solver = "FCI"
    hermitian = True
    nmom = 10
    norm_max = 4

    @classmethod
    def setUpClass(cls):
        cls.mf = testsystems.water_sto3g.rhf()
        cls.fci = testsystems.water_sto3g.rfci()

    @classmethod
    @cache
    def full_moms(cls):
        exprh = dyson.FCI.hole.from_mf(cls.mf)
        gfh_moms = exprh.build_gf_moments(cls.nmom)

        exprp = dyson.FCI.particle.from_mf(cls.mf)
        gfp_moms = exprp.build_gf_moments(cls.nmom)
        if cls.hermitian:
            gfh_moms = 0.5 * (gfh_moms + gfh_moms.conj().transpose(0,2,1))
            gfp_moms = 0.5 * (gfp_moms + gfp_moms.conj().transpose(0,2,1))

        return gfh_moms, gfp_moms
    

    @classmethod
    def assertMomsClose(cls, moms1, moms2, rtol=1e-8):
        cls.assertAllclose(cls, moms1[:,:cls.norm_max], moms2[:,:cls.norm_max], rtol=rtol)
        
    @classmethod
    def tearDownClass(cls):
        del cls.mf
        del cls.fci


    def test_gf_moment_to_spectral(self):

        gfh_moms, gfp_moms = self.full_moms()
        gf_moms = GF_MomentRep([gfh_moms, gfp_moms])
        se_moms = gf_moms.to_se_moments()

        self.assertMomsClose(gf_moms.moments, se_moms.to_gf_moments().moments, rtol=1e-10)
        assert gf_moms.hermitian == self.hermitian

        spectral = gf_moms.to_spectral()
        gf_moms_spec = spectral.to_gf_moments()
        se_moms_spec = spectral.to_se_moments()

        self.assertTrue((gf_moms.hermitian and gf_moms_spec.hermitian) == self.hermitian)
        self.assertTrue(gf_moms.moments.shape == gf_moms_spec.moments.shape)
        self.assertMomsClose(se_moms.moments, se_moms_spec.moments, rtol=1e-8)
        
        self.assertTrue((se_moms.hermitian and se_moms_spec.hermitian) == self.hermitian)
        self.assertTrue(se_moms.moments.shape == se_moms_spec.moments.shape)
        self.assertMomsClose(gf_moms.moments, gf_moms_spec.moments, rtol=1e-8)

    def test_se_moment_to_spectral(self):

        gfh_moms, gfp_moms = self.full_moms()
        gf_moms = GF_MomentRep([gfh_moms, gfp_moms], hermitian=self.hermitian)
        se_moms = gf_moms.to_se_moments()

        self.assertMomsClose(gf_moms.moments, se_moms.to_gf_moments().moments, rtol=1e-10)

        spectral = se_moms.to_spectral()
        gf_moms_spec = spectral.to_gf_moments()
        se_moms_spec = spectral.to_se_moments()

        self.assertTrue((gf_moms.hermitian and gf_moms_spec.hermitian) == self.hermitian)
        self.assertTrue(gf_moms.moments.shape == gf_moms_spec.moments.shape)
        self.assertMomsClose(se_moms.moments, se_moms_spec.moments, rtol=1e-8)
        

        self.assertTrue((se_moms.hermitian and se_moms_spec.hermitian) == self.hermitian)
        self.assertTrue(se_moms.moments.shape == se_moms_spec.moments.shape)
        self.assertMomsClose(gf_moms.moments, gf_moms_spec.moments, rtol=1e-8)


    def test_moment_to_lehmann(self):
        pass

    def test_lehmann_to_moment(self):
        pass


class Test_CCSD_Dynamical_Types(Test_FCI_Dynamical_Types):
    
    solver = "CCSD"
    hermitian = False
    nmom = 10
    norm_max = 2

    @classmethod
    def setUpClass(cls):
        cls.mf = testsystems.water_ccpvdz.rhf()
        cls.fci = testsystems.water_ccpvdz.rccsd()


    @classmethod
    @cache
    def full_moms(cls):
        exprh = dyson.CCSD.hole.from_mf(cls.mf)
        gfh_moms = exprh.build_gf_moments(cls.nmom)

        exprp = dyson.CCSD.particle.from_mf(cls.mf)
        gfp_moms = exprp.build_gf_moments(cls.nmom)
        if cls.hermitian:
            gfh_moms = 0.5 * (gfh_moms + gfh_moms.conj().transpose(0,2,1))
            gfp_moms = 0.5 * (gfp_moms + gfp_moms.conj().transpose(0,2,1))

        return gfh_moms, gfp_moms


if __name__ == "__main__":
    print("Running %s" % __file__)
    unittest.main()
