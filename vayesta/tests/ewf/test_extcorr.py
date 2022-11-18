import pytest
import unittest
import numpy as np

import pyscf
import pyscf.cc

import vayesta
import vayesta.ewf

from vayesta.tests.common import TestCase
from vayesta.tests import testsystems


@pytest.mark.fast
class TestSolvers(TestCase):

    def _test(self, key, proj=0, mode='external-fciv', store_wf_ccsdtq=True):
        mf = getattr(getattr(testsystems, key[0]), key[1])()

        emb = vayesta.ewf.EWF(mf)
        with emb.iao_fragmentation() as f:
            if store_wf_ccsdtq:
                fci_frag = f.add_atomic_fragment([0, 1], solver='FCI', bath_options=dict(bathtype='full'), store_wf_type='CCSDTQ')
            else:
                fci_frag = f.add_atomic_fragment([0, 1], solver='FCI', bath_options=dict(bathtype='full'))
            ccsd = f.add_atomic_fragment([0], solver='CCSD', bath_options=dict(bathtype='full'), active=False)
            ccsd2 = f.add_atomic_fragment([1], solver='CCSD', bath_options=dict(bathtype='full'), active=False)
        emb.kernel()
        fci_frag_ecorr = emb.e_corr
        fci_frag_etot = emb.e_tot

        fci_frag.active = False
        ccsd.add_external_corrections([fci_frag], correction_type=mode, projectors=proj, test_extcorr=True)
        ccsd.active=True
        ccsd2.add_external_corrections([fci_frag], correction_type=mode, projectors=proj, test_extcorr=True)
        ccsd2.active=True
        emb.kernel()

        fci = pyscf.fci.FCI(mf)
        fci.threads = 1
        fci.conv_tol = 1e-12
        fci.davidson_only = True
        fci.kernel()

        self.assertAlmostEqual(fci_frag_ecorr, fci.e_tot - mf.e_tot)
        self.assertAlmostEqual(fci_frag_etot, fci.e_tot)
        self.assertAlmostEqual(emb.e_corr, fci_frag_ecorr)
        self.assertAlmostEqual(emb.e_tot, fci_frag_etot)

    # Test all combinations of options
    def test_r_exact_ec_lih_proj0_fciv_store(self):
        return self._test(('lih_ccpvdz', 'rhf'), proj=0, mode='external-fciv', store_wf_ccsdtq=True)
    def test_r_exact_ec_lih_proj1_fciv_store(self):
        return self._test(('lih_ccpvdz', 'rhf'), proj=1, mode='external-fciv', store_wf_ccsdtq=True)
    def test_r_exact_ec_lih_proj2_fciv_store(self):
        return self._test(('lih_ccpvdz', 'rhf'), proj=2, mode='external-fciv', store_wf_ccsdtq=True)
    def test_r_exact_ec_lih_proj0_fciv_nostore(self):
        return self._test(('lih_ccpvdz', 'rhf'), proj=0, mode='external-fciv', store_wf_ccsdtq=False)
    def test_r_exact_ec_lih_proj1_fciv_nostore(self):
        return self._test(('lih_ccpvdz', 'rhf'), proj=1, mode='external-fciv', store_wf_ccsdtq=False)
    def test_r_exact_ec_lih_proj2_fciv_nostore(self):
        return self._test(('lih_ccpvdz', 'rhf'), proj=2, mode='external-fciv', store_wf_ccsdtq=False)
    def test_r_exact_ec_lih_proj0_ccsdv_store(self):
        return self._test(('lih_ccpvdz', 'rhf'), proj=0, mode='external-ccsdv', store_wf_ccsdtq=True)
    def test_r_exact_ec_lih_proj1_ccsdv_store(self):
        return self._test(('lih_ccpvdz', 'rhf'), proj=1, mode='external-ccsdv', store_wf_ccsdtq=True)
    def test_r_exact_ec_lih_proj2_ccsdv_store(self):
        return self._test(('lih_ccpvdz', 'rhf'), proj=2, mode='external-ccsdv', store_wf_ccsdtq=True)
    def test_r_exact_ec_lih_proj0_ccsdv_nostore(self):
        return self._test(('lih_ccpvdz', 'rhf'), proj=0, mode='external-ccsdv', store_wf_ccsdtq=False)
    def test_r_exact_ec_lih_proj1_ccsdv_nostore(self):
        return self._test(('lih_ccpvdz', 'rhf'), proj=1, mode='external-ccsdv', store_wf_ccsdtq=False)
    def test_r_exact_ec_lih_proj2_ccsdv_nostore(self):
        return self._test(('lih_ccpvdz', 'rhf'), proj=2, mode='external-ccsdv', store_wf_ccsdtq=False)

    def _test_10_u4_2imp(self, mode, proj):
        """Tests for N=10 U=4 Hubbard model with double site CCSD impurities
        and complete FCI fragment
        """

        mf = testsystems.hubb_10_u4.rhf()

        emb = vayesta.ewf.EWF(mf)
        with emb.site_fragmentation() as f:
            fci_frag = f.add_atomic_fragment(list(range(10)), solver='FCI', store_wf_type='CCSDTQ')
            ccsd_frag = f.add_atomic_fragment([0, 1], solver='CCSD', bath_options=dict(bathtype='full'), active=False)
        ccsd_frag.add_tsymmetric_fragments(tvecs=[5, 1, 1])
        emb.kernel()
        fci_frag_ecorr = emb.e_corr
        fci_frag_etot = emb.e_tot

        fci_frag.active = False
        ccsd_frag.active = True
        ccsd_frag.add_external_corrections([fci_frag], correction_type=mode, projectors=proj, test_extcorr=True)
        emb.kernel()

        fci = pyscf.fci.FCI(mf)
        fci.threads = 1
        fci.conv_tol = 1e-12
        fci.davidson_only = True
        fci.kernel()

        self.assertAlmostEqual(fci_frag_ecorr, fci.e_tot - mf.e_tot)
        self.assertAlmostEqual(fci_frag_etot, fci.e_tot)
        self.assertAlmostEqual(emb.e_corr, fci_frag_ecorr)
        self.assertAlmostEqual(emb.e_tot, fci_frag_etot)
    
    def test_hub_ec_2imp_proj0_ccsdv(self):
        return self._test_10_u4_2imp(mode='external-ccsdv', proj=0)
    def test_hub_ec_2imp_proj1_ccsdv(self):
        return self._test_10_u4_2imp(mode='external-ccsdv', proj=1)
    def test_hub_ec_2imp_proj2_ccsdv(self):
        return self._test_10_u4_2imp(mode='external-ccsdv', proj=2)
    def test_hub_ec_2imp_proj0_fciv(self):
        return self._test_10_u4_2imp(mode='external-fciv', proj=0)
    def test_hub_ec_2imp_proj1_fciv(self):
        return self._test_10_u4_2imp(mode='external-fciv', proj=1)
    def test_hub_ec_2imp_proj2_fciv(self):
        return self._test_10_u4_2imp(mode='external-fciv', proj=2)

    def _test_10_u4_2impfci(self, mode):
        """Tests for N=10 U=4 Hubbard model with double site CCSD impurities
        and 2-site FCI fragment (but complete bath). With no projectors on
        external correction, should still be exact.
        """

        mf = testsystems.hubb_10_u2.rhf()

        emb = vayesta.ewf.EWF(mf)
        with emb.site_fragmentation() as f:
            fci_frag = f.add_atomic_fragment([0, 1], solver='FCI', bath_options=dict(bathtype='full'), store_wf_type='CCSDTQ')
            ccsd_frag = f.add_atomic_fragment([0, 1], solver='CCSD', bath_options=dict(bathtype='full'), active=False)
        ccsd_frag.add_tsymmetric_fragments(tvecs=[5, 1, 1])
        emb.kernel()

        fci_frag.active = False
        ccsd_frag.active = True
        # Given the complete bath of the FCI fragment, should still be exact
        ccsd_frag.add_external_corrections([fci_frag], correction_type=mode, projectors=0, test_extcorr=True)
        emb.kernel()

        fci = pyscf.fci.FCI(mf)
        fci.threads = 1
        fci.conv_tol = 1e-12
        fci.davidson_only = True
        fci.kernel()

        self.assertAlmostEqual(emb.e_corr, fci.e_tot - mf.e_tot)
        self.assertAlmostEqual(emb.e_tot, fci.e_tot)

    def test_hub_ec_2impfci_proj0_fciv(self):
        return self._test_10_u4_2impfci(mode='external-fciv')
    def test_hub_ec_2impfci_proj0_ccsdv(self):
        return self._test_10_u4_2impfci(mode='external-ccsdv')

    def _test_water_ec_regression(self, mode=None, projectors=None):
    
        mf = testsystems.water_ccpvdz_df.rhf()
        
        emb = vayesta.ewf.EWF(mf)
        fci_frags = []
        with emb.iao_fragmentation() as f:
            fci_frags = f.add_all_atomic_fragments(solver='FCI', bath_options=dict(bathtype='dmet'), store_wf_type='CCSDTQ')
            ccsd_frags = f.add_all_atomic_fragments(solver='CCSD', bath_options=dict(bathtype='full'), active=False)
        emb.kernel()

        for fci_frag in fci_frags:
            fci_frag.active = False

        for ccsd_frag in ccsd_frags:
            ccsd_frag.active = True
            ccsd_frag.add_external_corrections(fci_frags, correction_type=mode, projectors=projectors, test_extcorr=True)

        emb.kernel()
        return emb.e_tot

    def test_water_ec_regression_proj0_fciv(self):
        e_tot = self._test_water_ec_regression(mode='external-fciv', projectors=0)
        self.assertAlmostEqual(e_tot, -76.2402530047042)
    def test_water_ec_regression_proj0_ccsdv(self):
        e_tot = self._test_water_ec_regression(mode='external-ccsdv', projectors=0)
        self.assertAlmostEqual(e_tot, -76.24018922136538)
    def test_water_ec_regression_proj1_fciv(self):
        e_tot = self._test_water_ec_regression(mode='external-fciv', projectors=1)
        self.assertAlmostEqual(e_tot, -76.24022612405739)
    def test_water_ec_regression_proj1_ccsdv(self):
        e_tot = self._test_water_ec_regression(mode='external-ccsdv', projectors=1)
        self.assertAlmostEqual(e_tot, -76.24017967220551)
    def test_water_ec_regression_proj2_fciv(self):
        e_tot = self._test_water_ec_regression(mode='external-fciv', projectors=2)
        self.assertAlmostEqual(e_tot, -76.24020498388937)
    def test_water_ec_regression_proj2_ccsdv(self):
        e_tot = self._test_water_ec_regression(mode='external-ccsdv', projectors=2)
        self.assertAlmostEqual(e_tot, -76.24017093290053)
        
if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
