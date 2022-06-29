import pytest
import unittest
import numpy as np

from pyscf import mp, cc
from pyscf.pbc import tools
from pyscf.cc import dfccsd

from vayesta.core.ao2mo import kao2gmo
from vayesta.tests.common import TestCase
from vayesta.tests import testsystems


@pytest.mark.slow
class KAO2GMOTests(TestCase):
    PLACES_ERIS = 8

    @classmethod
    def setUpClass(cls):
        cls.kmf = testsystems.h2_sto3g_k311.rhf()
        cls.gmf = testsystems.h2_sto3g_s311.rhf()

    @classmethod
    def tearDownClass(cls):
        del cls.kmf, cls.gmf

    def test_gdf_to_eris(self):
        """Test agreement between unfolded and gamma-point integrals.
        """

        mf = self.kmf
        gmf = self.gmf
        mo_coeff_occ = gmf.mo_coeff[:, gmf.mo_occ > 0]
        mo_coeff_vir = gmf.mo_coeff[:, gmf.mo_occ == 0]

        eri0 = kao2gmo.gdf_to_eris(mf.with_df, gmf.mo_coeff, nocc=mo_coeff_occ.shape[1], only_ovov=False, j3c_threshold=1e-12)
        eri1 = kao2gmo.gdf_to_eris(mf.with_df, gmf.mo_coeff, nocc=mo_coeff_occ.shape[1], only_ovov=False, real_j3c=False)
        for key in eri0.keys():
            coeffs = tuple([mo_coeff_occ, mo_coeff_vir][k == 'v'] for k in key)
            eri2 = gmf.with_df.ao2mo(coeffs, compact=False).reshape([c.shape[1] for c in coeffs])
            self.assertAlmostEqual(np.max(np.abs(eri0[key]-eri2)), 0.0, self.PLACES_ERIS)
            self.assertAlmostEqual(np.max(np.abs(eri1[key]-eri2)), 0.0, self.PLACES_ERIS)

    def test_mp2_eris(self):
        """Test agreement between unfolded and gamma-point integrals
        for the MP2 pyscf method.
        """

        mf = self.kmf
        gmf = self.gmf
        cm = mp.mp2.MP2(gmf)
        mo_coeff_occ = gmf.mo_coeff[:, gmf.mo_occ > 0]
        mo_coeff_vir = gmf.mo_coeff[:, gmf.mo_occ == 0]

        eri0 = kao2gmo.gdf_to_pyscf_eris(gmf, mf.with_df, cm, fock=gmf.get_fock(), mo_energy=mf.mo_energy, e_hf=gmf.e_tot)
        for key in ['ovov']:
            coeffs = tuple([mo_coeff_occ, mo_coeff_vir][k == 'v'] for k in key)
            eri1 = gmf.with_df.ao2mo(coeffs, compact=False).reshape([c.shape[1] for c in coeffs])
            self.assertAlmostEqual(np.max(np.abs(getattr(eri0, key)-eri1)), 0.0, self.PLACES_ERIS)

    def test_rccsd_eris(self):
        """Test agreement between unfolded and gamma-point integrals
        for the RCCSD pyscf method.
        """

        mf = self.kmf
        gmf = self.gmf
        cm = cc.rccsd.RCCSD(gmf)
        mo_coeff_occ = gmf.mo_coeff[:, gmf.mo_occ > 0]
        mo_coeff_vir = gmf.mo_coeff[:, gmf.mo_occ == 0]

        eri0 = kao2gmo.gdf_to_pyscf_eris(gmf, mf.with_df, cm, fock=gmf.get_fock(), mo_energy=gmf.mo_energy, e_hf=gmf.e_tot)
        for key in ['ovov', 'oovv', 'ovvo', 'ovoo', 'oooo', 'ovvv', 'vvvv']:
            coeffs = tuple([mo_coeff_occ, mo_coeff_vir][k == 'v'] for k in key)
            eri1 = gmf.with_df.ao2mo(coeffs, compact=False).reshape([c.shape[1] for c in coeffs])
            self.assertAlmostEqual(np.max(np.abs(getattr(eri0, key)-eri1)), 0.0, self.PLACES_ERIS)

    def test_dfccsd_eris(self):
        """Test agreement between unfolded and gamma-point integrals
        for the DFCCSD pyscf method.
        """

        mf = self.kmf
        gmf = self.gmf
        cm = dfccsd.RCCSD(gmf)
        mo_coeff_occ = gmf.mo_coeff[:, gmf.mo_occ > 0]
        mo_coeff_vir = gmf.mo_coeff[:, gmf.mo_occ == 0]

        eri0 = kao2gmo.gdf_to_pyscf_eris(gmf, mf.with_df, cm, fock=gmf.get_fock(), mo_energy=gmf.mo_energy, e_hf=gmf.e_tot)
        for key in ['ovov', 'oovv', 'ovvo', 'ovoo', 'oooo']:
            coeffs = tuple([mo_coeff_occ, mo_coeff_vir][k == 'v'] for k in key)
            eri1 = gmf.with_df.ao2mo(coeffs, compact=False).reshape([c.shape[1] for c in coeffs])
            self.assertAlmostEqual(np.max(np.abs(getattr(eri0, key)-eri1)), 0.0, self.PLACES_ERIS)

        # ovvv is compressed with sym=True:
        coeffs = [mo_coeff_occ, mo_coeff_vir, mo_coeff_vir, mo_coeff_vir]
        eri1 = gmf.with_df.ao2mo(coeffs, compact=True)
        eri1 = eri1.reshape([coeffs[0].shape[1], coeffs[1].shape[1], -1])
        self.assertAlmostEqual(np.max(np.abs(eri0.ovvv-eri1)), 0.0, self.PLACES_ERIS)

        # (vv|L)(L|vv) -> (vv|vv) contraction is performed late:
        coeffs = [mo_coeff_vir, mo_coeff_vir, mo_coeff_vir, mo_coeff_vir]
        eri1 = gmf.with_df.ao2mo(coeffs, compact=True)
        eri0_vvvv = np.dot(eri0.vvL, eri0.vvL.T)
        self.assertAlmostEqual(np.max(np.abs(eri0_vvvv-eri1)), 0.0, self.PLACES_ERIS)

    def test_ccsd_eris(self):
        """Test agreement between unfolded and gamma-point integrals
        for the CCSD pyscf method.
        """

        mf = self.kmf
        gmf = self.gmf
        cm = cc.ccsd.CCSD(gmf)
        mo_coeff_occ = gmf.mo_coeff[:, gmf.mo_occ > 0]
        mo_coeff_vir = gmf.mo_coeff[:, gmf.mo_occ == 0]

        eri0 = kao2gmo.gdf_to_pyscf_eris(gmf, mf.with_df, cm, fock=gmf.get_fock(), mo_energy=gmf.mo_energy, e_hf=gmf.e_tot)
        for key in ['ovov', 'oovv', 'ovvo', 'ovoo', 'oooo']:
            coeffs = tuple([mo_coeff_occ, mo_coeff_vir][k == 'v'] for k in key)
            eri1 = gmf.with_df.ao2mo(coeffs, compact=False).reshape([c.shape[1] for c in coeffs])
            self.assertAlmostEqual(np.max(np.abs(getattr(eri0, key)-eri1)), 0.0, self.PLACES_ERIS)

        # ovvv and vvvv are compressed with sym=True:
        coeffs = [mo_coeff_occ, mo_coeff_vir, mo_coeff_vir, mo_coeff_vir]
        eri1 = gmf.with_df.ao2mo(coeffs, compact=True)
        eri1 = eri1.reshape([coeffs[0].shape[1], coeffs[1].shape[1], -1])
        self.assertAlmostEqual(np.max(np.abs(eri0.ovvv-eri1)), 0.0, self.PLACES_ERIS)
        coeffs = [mo_coeff_vir, mo_coeff_vir, mo_coeff_vir, mo_coeff_vir]
        eri1 = gmf.with_df.ao2mo(coeffs, compact=True)
        self.assertAlmostEqual(np.max(np.abs(eri0.vvvv-eri1)), 0.0, self.PLACES_ERIS)


@pytest.mark.slow
class KAO2GMO2dTests(KAO2GMOTests):
    @classmethod
    def setUpClass(cls):
        cls.kmf = testsystems.h2_sto3g_k31.rhf()
        cls.gmf = testsystems.h2_sto3g_s31.rhf()

    #FIXME
    test_gdf_to_eris = None
    test_dfccsd_eris = None


if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
