import pytest
import unittest
import numpy as np

import pyscf
import pyscf.cc
from pyscf.scf import convert_to_uhf

import vayesta
import vayesta.ewf

from vayesta.tests.common import TestCase
from vayesta.tests import testsystems


#@pytest.mark.fast
class TestFullEC_tailor(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fci = {}
        cls.mf = {}

        cls.mf['lih_631g'] = testsystems.lih_631g.rhf()
        cls.fci['lih_631g'] = pyscf.fci.FCI(cls.mf['lih_631g'])
        cls.fci['lih_631g'].threads = 1
        cls.fci['lih_631g'].conv_tol = 1.0e-12
        cls.fci['lih_631g'].davidson_only = True
        cls.fci['lih_631g'].kernel()

        cls.mf['lih_ccpvdz'] = testsystems.lih_ccpvdz.rhf()
        cls.fci['lih_ccpvdz'] = pyscf.fci.FCI(cls.mf['lih_ccpvdz'])
        cls.fci['lih_ccpvdz'].threads = 1
        cls.fci['lih_ccpvdz'].conv_tol = 1.0e-12
        cls.fci['lih_ccpvdz'].davidson_only = True
        cls.fci['lih_ccpvdz'].kernel()

    @classmethod
    def tearDownClass(cls):
        cls.fci.clear()
        cls.mf.clear()
        del cls.fci
        del cls.mf


    def _test(cls, key, proj=0, bathtype='full', fcifragtype = 'atomic', mode='external-fciv', store_wf_ccsdtq=True):
        mf = getattr(getattr(testsystems, key[0]), key[1])()
        if fcifragtype == 'fullsystem':
            emb = vayesta.ewf.EWF(mf, solver_options={'conv_tol':1.0e-12} )
            fci_frags = []
            with emb.iao_fragmentation() as f:
                if store_wf_ccsdtq:
                    fci_frags.append(f.add_full_system(solver='FCI', bath_options=dict(bathtype=bathtype),
                                                       store_wf_type='CCSDTQ', auxiliary=True))
                else:
                    fci_frags.append(f.add_full_system(solver='FCI', bath_options=dict(bathtype=bathtype),
                                                       auxiliary=True))
                ccsd_frag = f.add_full_system(solver='CCSD', bath_options=dict(bathtype='full'))
            ccsd_frag.add_external_corrections(fci_frags, correction_type=mode, projectors=proj,test_extcorr=True)
            emb.kernel()


        if fcifragtype == 'atomic':
            emb = vayesta.ewf.EWF(mf, solver_options={'conv_tol':1.0e-12})
            with emb.iao_fragmentation() as f:
                if store_wf_ccsdtq:
                    fci_frag = f.add_all_atomic_fragments(solver='FCI', bath_options=dict(bathtype=bathtype),
                                                          store_wf_type='CCSDTQ', auxiliary=True)
                else:
                    fci_frag = f.add_all_atomic_fragments(solver='FCI', bath_options=dict(bathtype=bathtype),
                                                          auxiliary=True)
                ccsd_frag = f.add_full_system(solver='CCSD', bath_options=dict(bathtype='full'))
            ccsd_frag.add_external_corrections(fci_frag, correction_type=mode, projectors=proj, test_extcorr=True)
            
            emb.kernel()

        return emb.e_tot, cls.fci[key[0]].e_tot


# FCI fragment bath type=full

    @pytest.mark.fast
    def test_r_regression_ec_lih_631g_atomicfrags_proj0_ccsdv_store(self):
        e_tot, fci_e_tot = self._test(('lih_631g', 'rhf'), proj=0, bathtype='full', fcifragtype = 'atomic',
                                      mode='external-ccsdv', store_wf_ccsdtq=True)
        self.assertAlmostEqual(e_tot, -7.988952842115175)

    @pytest.mark.fast
    def test_r_exact_ec_lih_631g_atomicfrags_proj1_ccsdv_store(self):
        e_tot, fci_e_tot = self._test(('lih_631g', 'rhf'), proj=1, bathtype='full', fcifragtype = 'atomic',
                                      mode='external-ccsdv', store_wf_ccsdtq=True)
        self.assertAlmostEqual(e_tot, fci_e_tot)

    @pytest.mark.fast
    def test_r_regression_ec_lih_631g_atomicfrags_proj2_ccsdv_store(self):
        e_tot, fci_e_tot = self._test(('lih_631g', 'rhf'), proj=2, bathtype='full', fcifragtype = 'atomic',
                                      mode='external-ccsdv', store_wf_ccsdtq=True)
        self.assertAlmostEqual(e_tot,  -7.98893732925952)

    @pytest.mark.slow
    def test_r_regression_ec_lih_631g_atomicfrags_proj0_fciv_store(self):
        e_tot, fci_e_tot = self._test(('lih_631g', 'rhf'), proj=0, bathtype='full', fcifragtype = 'atomic',
                                      mode='external-fciv', store_wf_ccsdtq=True)
        self.assertAlmostEqual(e_tot, -7.988952842133734)

    @pytest.mark.slow
    def test_r_exact_ec_lih_631g_atomicfrags_proj1_fciv_store(self):
        e_tot, fci_e_tot = self._test(('lih_631g', 'rhf'), proj=1, bathtype='full', fcifragtype = 'atomic',
                                      mode='external-fciv', store_wf_ccsdtq=True)
        self.assertAlmostEqual(e_tot, fci_e_tot)

    @pytest.mark.slow
    def test_r_regression_ec_lih_631g_atomicfrags_proj2_fciv_store(self):
        e_tot, fci_e_tot = self._test(('lih_631g', 'rhf'), proj=2, bathtype='full', fcifragtype = 'atomic',
                                      mode='external-fciv', store_wf_ccsdtq=True)
        self.assertAlmostEqual(e_tot,  -7.98893732924287)


    @pytest.mark.slow
    def test_r_exact_ec_lih_631g_fullsystem_proj0_ccsdv_store(self):
        e_tot, fci_e_tot = self._test(('lih_631g', 'rhf'), proj=0, bathtype='full', fcifragtype = 'fullsystem',
                                      mode='external-ccsdv', store_wf_ccsdtq=True)
        self.assertAlmostEqual(e_tot, fci_e_tot)

    @pytest.mark.slow
    def test_r_exact_ec_lih_631g_fullsystem_proj1_ccsdv_store(self):
        e_tot, fci_e_tot = self._test(('lih_631g', 'rhf'), proj=1, bathtype='full', fcifragtype = 'fullsystem',
                                      mode='external-ccsdv', store_wf_ccsdtq=True)
        self.assertAlmostEqual(e_tot, fci_e_tot)

    @pytest.mark.slow
    def test_r_exact_ec_lih_631g_fullsystem_proj2_ccsdv_store(self):
        e_tot, fci_e_tot = self._test(('lih_631g', 'rhf'), proj=2, bathtype='full', fcifragtype = 'fullsystem',
                                      mode='external-ccsdv', store_wf_ccsdtq=True)
        self.assertAlmostEqual(e_tot, fci_e_tot)

    @pytest.mark.slow
    def test_r_exact_ec_lih_631g_fullsystem_proj0_fciv_store(self):
        e_tot, fci_e_tot = self._test(('lih_631g', 'rhf'), proj=0, bathtype='full', fcifragtype = 'fullsystem',
                                      mode='external-fciv', store_wf_ccsdtq=True)
        self.assertAlmostEqual(e_tot, fci_e_tot)

    @pytest.mark.slow
    def test_r_exact_ec_lih_631g_fullsystem_proj1_fciv_store(self):
        e_tot, fci_e_tot = self._test(('lih_631g', 'rhf'), proj=1, bathtype='full', fcifragtype = 'fullsystem',
                                      mode='external-fciv', store_wf_ccsdtq=True)
        self.assertAlmostEqual(e_tot, fci_e_tot)

    @pytest.mark.slow
    def test_r_exact_ec_lih_631g_fullsystem_proj2_fciv_store(self):
        e_tot, fci_e_tot = self._test(('lih_631g', 'rhf'), proj=2, bathtype='full', fcifragtype = 'fullsystem',
                                      mode='external-fciv', store_wf_ccsdtq=True)
        self.assertAlmostEqual(e_tot, fci_e_tot)

# uhf tests


    @pytest.mark.slow
    def test_u_exact_ec_lih_631g_atomicfrags_proj0_ccsdv_store(self):
        e_tot, fci_e_tot = self._test(('lih_631g', 'uhf'), proj=0, bathtype='full', fcifragtype = 'atomic',
                                      mode='external-ccsdv', store_wf_ccsdtq=True)
        self.assertAlmostEqual(e_tot, fci_e_tot)

    @pytest.mark.slow
    def test_u_exact_ec_lih_631g_atomicfrags_proj1_ccsdv_store(self):
        e_tot, fci_e_tot = self._test(('lih_631g', 'uhf'), proj=1, bathtype='full', fcifragtype = 'atomic',
                                      mode='external-ccsdv', store_wf_ccsdtq=True)
        self.assertAlmostEqual(e_tot, fci_e_tot)

    @pytest.mark.slow
    def test_u_exact_ec_lih_631g_atomicfrags_proj2_ccsdv_store(self):
        e_tot, fci_e_tot = self._test(('lih_631g', 'uhf'), proj=2, bathtype='full', fcifragtype = 'atomic',
                                      mode='external-ccsdv', store_wf_ccsdtq=True)
        self.assertAlmostEqual(e_tot, fci_e_tot)

    @pytest.mark.slow
    def test_u_regression_ec_lih_631g_atomicfrags_proj0_fciv_store(self):
        e_tot, fci_e_tot = self._test(('lih_631g', 'uhf'), proj=0, bathtype='full', fcifragtype = 'atomic',
                                      mode='external-fciv', store_wf_ccsdtq=True)
        self.assertAlmostEqual(e_tot, -7.988952849629403)

# A test with nostore
    @pytest.mark.slow
    def test_u_regression_ec_lih_631g_atomicfrags_proj0_fciv_nostore(self):
        e_tot, fci_e_tot = self._test(('lih_631g', 'uhf'), proj=0, bathtype='full', fcifragtype = 'atomic',
                                      mode='external-fciv', store_wf_ccsdtq=False)
        self.assertAlmostEqual(e_tot, -7.988952849629403)

    @pytest.mark.slow
    def test_u_exact_ec_lih_631g_atomicfrags_proj1_fciv_store(self):
        e_tot, fci_e_tot = self._test(('lih_631g', 'uhf'), proj=1, bathtype='full', fcifragtype = 'atomic',
                                      mode='external-fciv', store_wf_ccsdtq=True)
        self.assertAlmostEqual(e_tot, fci_e_tot)

    @pytest.mark.slow
    def test_u_regression_ec_lih_631g_atomicfrags_proj2_fciv_store(self):
        e_tot, fci_e_tot = self._test(('lih_631g', 'uhf'), proj=2, bathtype='full', fcifragtype = 'atomic',
                                      mode='external-fciv', store_wf_ccsdtq=True)
        self.assertAlmostEqual(e_tot, -7.988937336775954)


    @pytest.mark.fast
    def test_u_exact_ec_lih_631g_fullsystem_proj0_ccsdv_store(self):
        e_tot, fci_e_tot = self._test(('lih_631g', 'uhf'), proj=0, bathtype='full', fcifragtype = 'fullsystem',
                                      mode='external-ccsdv', store_wf_ccsdtq=True)
        self.assertAlmostEqual(e_tot, fci_e_tot)

    @pytest.mark.slow
    def test_u_exact_ec_lih_631g_fullsystem_proj1_ccsdv_store(self):
        e_tot, fci_e_tot = self._test(('lih_631g', 'uhf'), proj=1, bathtype='full', fcifragtype = 'fullsystem',
                                      mode='external-ccsdv', store_wf_ccsdtq=True)
        self.assertAlmostEqual(e_tot, fci_e_tot)

    @pytest.mark.fast
    def test_u_exact_ec_lih_631g_fullsystem_proj2_ccsdv_store(self):
        e_tot, fci_e_tot = self._test(('lih_631g', 'uhf'), proj=2, bathtype='full', fcifragtype = 'fullsystem',
                                      mode='external-ccsdv', store_wf_ccsdtq=True)
        self.assertAlmostEqual(e_tot, fci_e_tot)

    @pytest.mark.slow
    def test_u_exact_ec_lih_631g_fullsystem_proj0_fciv_store(self):
        e_tot, fci_e_tot = self._test(('lih_631g', 'uhf'), proj=0, bathtype='full', fcifragtype = 'fullsystem',
                                      mode='external-fciv', store_wf_ccsdtq=True)
        self.assertAlmostEqual(e_tot, fci_e_tot)

# test tailoring
    @pytest.mark.fast
    def test_u_exact_tailor_lih_631g_fullsystem_proj0_fciv_store(self):
        e_tot, fci_e_tot = self._test(('lih_631g', 'uhf'), proj=0, bathtype='full', fcifragtype = 'fullsystem',
                                      mode='tailor', store_wf_ccsdtq=True)
        self.assertAlmostEqual(e_tot, fci_e_tot)

    @pytest.mark.fast
    def test_u_exact_tailor_lih_631g_fullsystem_proj0_fciv_nostore(self):
        e_tot, fci_e_tot = self._test(('lih_631g', 'uhf'), proj=0, bathtype='full', fcifragtype = 'fullsystem',
                                      mode='tailor', store_wf_ccsdtq=False)
        self.assertAlmostEqual(e_tot, fci_e_tot)

    @pytest.mark.fast
    def test_u_exact_ec_lih_631g_fullsystem_proj1_fciv_store(self):
        e_tot, fci_e_tot = self._test(('lih_631g', 'uhf'), proj=1, bathtype='full', fcifragtype = 'fullsystem',
                                      mode='external-fciv', store_wf_ccsdtq=True)
        self.assertAlmostEqual(e_tot, fci_e_tot)

    @pytest.mark.slow
    def test_u_exact_ec_lih_631g_fullsystem_proj2_fciv_store(self):
        e_tot, fci_e_tot = self._test(('lih_631g', 'uhf'), proj=2, bathtype='full', fcifragtype = 'fullsystem',
                                      mode='external-fciv', store_wf_ccsdtq=True)
        self.assertAlmostEqual(e_tot, fci_e_tot)

    #  FCI fragment bathtype = dmet    
    
    @pytest.mark.fast
    def test_r_regression_ec_lih_631g_atomicfrags_bathype_dmet_proj0_ccsdv_store(self):
        e_tot, fci_e_tot = self._test(('lih_631g', 'rhf'), proj=0, bathtype='dmet', fcifragtype = 'atomic',
                                      mode='external-ccsdv', store_wf_ccsdtq=True)
        self.assertAlmostEqual(e_tot, -7.988931393747139)

    @pytest.mark.fast
    def test_r_regression_ec_lih_631g_atomicfrags_bathype_dmet_proj1_ccsdv_store(self):
        e_tot, fci_e_tot = self._test(('lih_631g', 'rhf'), proj=1, bathtype='dmet', fcifragtype = 'atomic',
                                      mode='external-ccsdv', store_wf_ccsdtq=True)
        self.assertAlmostEqual(e_tot, -7.988931393747468)

    @pytest.mark.fast
    def test_r_regression_ec_lih_631g_atomicfrags_bathype_dmet_proj2_ccsdv_store(self):
        e_tot, fci_e_tot = self._test(('lih_631g', 'rhf'), proj=2, bathtype='dmet', fcifragtype = 'atomic',
                                      mode='external-ccsdv', store_wf_ccsdtq=True)
        self.assertAlmostEqual(e_tot,  -7.988931393747468)

    @pytest.mark.slow
    def test_r_regression_ec_lih_631g_atomicfrags_bathype_dmet_proj0_fciv_store(self):
        e_tot, fci_e_tot = self._test(('lih_631g', 'rhf'), proj=0, bathtype='dmet', fcifragtype = 'atomic',
                                      mode='external-fciv', store_wf_ccsdtq=True)
        self.assertAlmostEqual(e_tot, -7.9889313937490956)

    @pytest.mark.slow
    def test_r_regression_ec_lih_631g_atomicfrags_bathype_dmet_proj1_fciv_store(self):
        e_tot, fci_e_tot = self._test(('lih_631g', 'rhf'), proj=1, bathtype='dmet', fcifragtype = 'atomic',
                                      mode='external-fciv', store_wf_ccsdtq=True)
        self.assertAlmostEqual(e_tot, -7.988931393746693)

    @pytest.mark.slow
    def test_r_regression_ec_lih_631g_atomicfrags_bathype_dmet_proj2_fciv_store(self):
        e_tot, fci_e_tot = self._test(('lih_631g', 'rhf'), proj=2, bathtype='dmet', fcifragtype = 'atomic',
                                      mode='external-fciv', store_wf_ccsdtq=True)
        self.assertAlmostEqual(e_tot, -7.98893139374700)


    @pytest.mark.slow
    def test_r_regression_ec_lih_631g_fullsystem_bathype_dmet_proj0_ccsdv_store(self):
        e_tot, fci_e_tot = self._test(('lih_631g', 'rhf'), proj=0, bathtype='dmet', fcifragtype = 'fullsystem',
                                      mode='external-ccsdv', store_wf_ccsdtq=True)
        self.assertAlmostEqual(e_tot, -7.988931393745471)

    @pytest.mark.slow
    def test_r_regression_ec_lih_631g_fullsystem_bathype_dmet_proj1_ccsdv_store(self):
        e_tot, fci_e_tot = self._test(('lih_631g', 'rhf'), proj=1, bathtype='dmet', fcifragtype = 'fullsystem',
                                      mode='external-ccsdv', store_wf_ccsdtq=True)
        self.assertAlmostEqual(e_tot, -7.988931393747714)

    @pytest.mark.slow
    def test_r_regression_ec_lih_631g_fullsystem_bathype_dmet_proj2_ccsdv_store(self):
        e_tot, fci_e_tot = self._test(('lih_631g', 'rhf'), proj=2, bathtype='dmet', fcifragtype = 'fullsystem',
                                      mode='external-ccsdv', store_wf_ccsdtq=True)
        self.assertAlmostEqual(e_tot, -7.98893139374664)

    @pytest.mark.slow
    def test_r_regression_ec_lih_631g_fullsystem_bathype_dmet_proj0_fciv_store(self):
        e_tot, fci_e_tot = self._test(('lih_631g', 'rhf'), proj=0, bathtype='dmet', fcifragtype = 'fullsystem',
                                      mode='external-fciv', store_wf_ccsdtq=True)
        self.assertAlmostEqual(e_tot, -7.988931393748568)

    @pytest.mark.slow
    def test_r_regression_ec_lih_631g_fullsystem_bathype_dmet_proj1_fciv_store(self):
        e_tot, fci_e_tot = self._test(('lih_631g', 'rhf'), proj=1, bathtype='dmet', fcifragtype = 'fullsystem',
                                      mode='external-fciv', store_wf_ccsdtq=True)
        self.assertAlmostEqual(e_tot, -7.988931393746042)

    @pytest.mark.slow
    def test_r_regression_ec_lih_631g_fullsystem_bathype_dmet_proj2_fciv_store(self):
        e_tot, fci_e_tot = self._test(('lih_631g', 'rhf'), proj=2, bathtype='dmet', fcifragtype = 'fullsystem',
                                      mode='external-fciv', store_wf_ccsdtq=True)
        self.assertAlmostEqual(e_tot, -7.988931393748121)

    # uhf tests


    @pytest.mark.slow
    def test_u_ec_lih_631g_atomicfrags_bathype_dmet_proj0_ccsdv_store(self):
        e_tot, fci_e_tot = self._test(('lih_631g', 'uhf'), proj=0, bathtype='dmet', fcifragtype = 'atomic',
                                      mode='external-ccsdv', store_wf_ccsdtq=True)
        self.assertAlmostEqual(e_tot, fci_e_tot)

    @pytest.mark.slow
    def test_u_exact_ec_lih_631g_atomicfrags_bathype_dmet_proj1_ccsdv_store(self):
        e_tot, fci_e_tot = self._test(('lih_631g', 'uhf'), proj=1, bathtype='dmet', fcifragtype = 'atomic',
                                      mode='external-ccsdv', store_wf_ccsdtq=True)
        self.assertAlmostEqual(e_tot, fci_e_tot)

    @pytest.mark.slow
    def test_u_ec_lih_631g_atomicfrags_bathype_dmet_proj2_ccsdv_store(self):
        e_tot, fci_e_tot = self._test(('lih_631g', 'uhf'), proj=2, bathtype='dmet', fcifragtype = 'atomic',
                                      mode='external-ccsdv', store_wf_ccsdtq=True)
        self.assertAlmostEqual(e_tot, fci_e_tot)

    @pytest.mark.slow
    def test_u_regression_ec_lih_631g_atomicfrags_bathype_dmet_proj0_fciv_store(self):
        e_tot, fci_e_tot = self._test(('lih_631g', 'uhf'), proj=0, bathtype='dmet', fcifragtype = 'atomic',
                                      mode='external-fciv', store_wf_ccsdtq=True)
        self.assertAlmostEqual(e_tot, -7.988931393739508)

    # test with nostore
    @pytest.mark.slow
    def test_u_regression_ec_lih_631g_atomicfrags_bathype_dmet_proj0_fciv_nostore(self):
        e_tot, fci_e_tot = self._test(('lih_631g', 'uhf'), proj=0, bathtype='dmet', fcifragtype = 'atomic',
                                      mode='external-fciv', store_wf_ccsdtq=False)
        self.assertAlmostEqual(e_tot, -7.988931393739513)

    @pytest.mark.slow
    def test_u_regression_ec_lih_631g_atomicfrags_bathype_dmet_proj1_fciv_store(self):
        e_tot, fci_e_tot = self._test(('lih_631g', 'uhf'), proj=1, bathtype='dmet', fcifragtype = 'atomic',
                                      mode='external-fciv', store_wf_ccsdtq=True)
        self.assertAlmostEqual(e_tot, -7.9889313937395015 )

    @pytest.mark.slow
    def test_u_regression_ec_lih_631g_atomicfrags_bathype_dmet_proj2_fciv_store(self):
        e_tot, fci_e_tot = self._test(('lih_631g', 'uhf'), proj=2, bathtype='dmet', fcifragtype = 'atomic',
                                      mode='external-fciv', store_wf_ccsdtq=True)
        self.assertAlmostEqual(e_tot, -7.988931393739511)


    @pytest.mark.fast
    def test_u_exact_ec_lih_631g_fullsystem_bathype_dmet_proj0_ccsdv_store(self):
        e_tot, fci_e_tot = self._test(('lih_631g', 'uhf'), proj=0, bathtype='dmet', fcifragtype = 'fullsystem',
                                      mode='external-ccsdv', store_wf_ccsdtq=True)
        self.assertAlmostEqual(e_tot, fci_e_tot)

    @pytest.mark.slow
    def test_u_exact_ec_lih_631g_fullsystem_bathype_dmet_proj1_ccsdv_store(self):
        e_tot, fci_e_tot = self._test(('lih_631g', 'uhf'), proj=1, bathtype='dmet', fcifragtype = 'fullsystem',
                                      mode='external-ccsdv', store_wf_ccsdtq=True)
        self.assertAlmostEqual(e_tot, fci_e_tot)

    @pytest.mark.fast
    def test_u_exact_ec_lih_631g_fullsystem_bathype_dmet_proj2_ccsdv_store(self):
        e_tot, fci_e_tot = self._test(('lih_631g', 'uhf'), proj=2, bathtype='dmet', fcifragtype = 'fullsystem',
                                      mode='external-ccsdv', store_wf_ccsdtq=True)
        self.assertAlmostEqual(e_tot, fci_e_tot)

    @pytest.mark.slow
    def test_u_regression_ec_lih_631g_fullsystem_bathype_dmet_proj0_fciv_store(self):
        e_tot, fci_e_tot = self._test(('lih_631g', 'uhf'), proj=0, bathtype='dmet', fcifragtype = 'fullsystem',
                                      mode='external-fciv', store_wf_ccsdtq=True)
        self.assertAlmostEqual(e_tot, -7.988931393739515 )

    @pytest.mark.fast
    def test_u_regression_ec_lih_631g_fullsystem_bathype_dmet_proj1_fciv_store(self):
        e_tot, fci_e_tot = self._test(('lih_631g', 'uhf'), proj=1, bathtype='dmet', fcifragtype = 'fullsystem',
                                      mode='external-fciv', store_wf_ccsdtq=True)
        self.assertAlmostEqual(e_tot, -7.9889313937395015 )

    @pytest.mark.slow
    def test_u_regression_ec_lih_631g_fullsystem_bathype_dmet_proj2_fciv_store(self):
        e_tot, fci_e_tot = self._test(('lih_631g', 'uhf'), proj=2, bathtype='dmet', fcifragtype = 'fullsystem',
                                      mode='external-fciv', store_wf_ccsdtq=True)
        self.assertAlmostEqual(e_tot, -7.988931393739506 )


@pytest.mark.fast
class TestHubCompleteEC(TestCase):
    def _test_10_u4_2imp(self, mode, proj):
        """Tests for N=10 U=4 Hubbard model with double site CCSD impurities
        and complete FCI fragment
        """

        mf = testsystems.hubb_10_u4.rhf()

        emb = vayesta.ewf.EWF(mf)
        emb.opts.solver_options['conv_tol'] = 1.0e-12

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

@pytest.mark.fast
class TestHubBathEC(TestCase):
    def _test_10_u2_2impfci(self, mode):
        """Tests for N=10 U=2 Hubbard model with double site CCSD impurities
        and 2-site FCI fragment (but complete bath). With no projectors on
        external correction, should still be exact.
        """

        mf = testsystems.hubb_10_u2.rhf()

        emb = vayesta.ewf.EWF(mf)
        emb.opts.solver_options['conv_tol'] = 1.0e-12

        with emb.site_fragmentation() as f:
            fci_frag = f.add_atomic_fragment([0, 1], solver='FCI', bath_options=dict(bathtype='full'),
                                             store_wf_type='CCSDTQ')
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
        return self._test_10_u2_2impfci(mode='external-fciv')
    def test_hub_ec_2impfci_proj0_ccsdv(self):
        return self._test_10_u2_2impfci(mode='external-ccsdv')

@pytest.mark.fast
class TestECSym(TestCase):
    def _test_10_u2_eccc_sym(self, mode, proj=0):
        """Test symmetry in external correction via comparison to system without use of symmetry
        """

        mf = testsystems.hubb_10_u2.rhf()

        emb = vayesta.ewf.EWF(mf)
        emb.opts.solver_options['conv_tol'] = 1.0e-12

        fci_frags = []
        with emb.site_fragmentation() as f:
            # Set up a two-site FCI fragment on all symmetrically equivalent fragments
            fci_frags.append(f.add_atomic_fragment([0, 1], solver='FCI', bath_options=dict(bathtype='dmet'),
                                                   store_wf_type='CCSDTQ'))
            fci_frags.append(f.add_atomic_fragment([2, 3], solver='FCI', bath_options=dict(bathtype='dmet'),
                                                   store_wf_type='CCSDTQ'))
            fci_frags.append(f.add_atomic_fragment([4, 5], solver='FCI', bath_options=dict(bathtype='dmet'),
                                                   store_wf_type='CCSDTQ'))
            fci_frags.append(f.add_atomic_fragment([6, 7], solver='FCI', bath_options=dict(bathtype='dmet'),
                                                   store_wf_type='CCSDTQ'))
            fci_frags.append(f.add_atomic_fragment([8, 9], solver='FCI', bath_options=dict(bathtype='dmet'),
                                                   store_wf_type='CCSDTQ'))
            ccsd_frag = f.add_full_system(solver='CCSD', bath_options=dict(bathtype='full'), active=False)
        emb.kernel()

        for fci_frag in fci_frags:
            fci_frag.active = False
        ccsd_frag.active = True
        # Note that proj=0 will 'over correct' here and double-count bath-bath tailoring contributions.
        # However, it should still be equivalent to the non symmetry version
        ccsd_frag.add_external_corrections(fci_frags, correction_type=mode, projectors=proj, test_extcorr=True)
        emb.kernel()
        e_tot_nosym = emb.e_tot

        # use symmetry of FCI clusters.
        emb = vayesta.ewf.EWF(mf)
        emb.opts.solver_options['conv_tol'] = 1.0e-12

        fci_frags = []
        with emb.site_fragmentation() as f:
            fci_frags.append(f.add_atomic_fragment([0, 1], solver='FCI', bath_options=dict(bathtype='dmet'),
                                                   store_wf_type='CCSDTQ'))
            ccsd_frag = f.add_full_system(solver='CCSD', bath_options=dict(bathtype='full'), active=False)

        # Add the symmetry-derived FCI fragments
        for fci_frag_sym in fci_frags[0].add_tsymmetric_fragments(tvecs=[5, 1, 1]):
            fci_frags.append(fci_frag_sym)
        emb.kernel()

        for fci_frag in fci_frags:
            fci_frag.active = False
        ccsd_frag.active = True
        ccsd_frag.add_external_corrections(fci_frags, correction_type=mode, projectors=proj, test_extcorr=True)
        emb.kernel()

        self.assertAlmostEqual(emb.e_tot, e_tot_nosym)

    def test_hub_ec_sym_proj0_fciv(self):
        return self._test_10_u2_eccc_sym(mode='external-fciv', proj=0)
    def test_hub_ec_sym_proj1_fciv(self):
        return self._test_10_u2_eccc_sym(mode='external-fciv', proj=1)
    def test_hub_ec_sym_proj2_fciv(self):
        return self._test_10_u2_eccc_sym(mode='external-fciv', proj=2)
    def test_hub_ec_sym_proj0_ccsdv(self):
        return self._test_10_u2_eccc_sym(mode='external-ccsdv', proj=0)
    def test_hub_ec_sym_proj1_ccsdv(self):
        return self._test_10_u2_eccc_sym(mode='external-ccsdv', proj=1)
    def test_hub_ec_sym_proj2_ccsdv(self):
        return self._test_10_u2_eccc_sym(mode='external-ccsdv', proj=2)
    def test_hub_ec_sym_proj1_dtailor(self):
        return self._test_10_u2_eccc_sym(mode='delta-tailor', proj=1)
    def test_hub_ec_sym_proj1_tailor(self):
        return self._test_10_u2_eccc_sym(mode='tailor', proj=1)

@pytest.mark.fast
class TestRegEC_CCSD(TestCase):
    def _test_water_ec_regression(self, mode=None, projectors=None):
    
        mf = testsystems.water_ccpvdz_df.rhf()
        
        emb = vayesta.ewf.EWF(mf)
        emb.opts.solver_options['conv_tol'] = 1.0e-12

        fci_frags = []
        with emb.iao_fragmentation() as f:
            fci_frags = f.add_all_atomic_fragments(solver='FCI', bath_options=dict(bathtype='dmet'),
                                                   store_wf_type='CCSDTQ')
            ccsd_frags = f.add_all_atomic_fragments(solver='CCSD', bath_options=dict(bathtype='full'), active=False)
        emb.kernel()

        for fci_frag in fci_frags:
            fci_frag.active = False

        for ccsd_frag in ccsd_frags:
            ccsd_frag.active = True
            ccsd_frag.add_external_corrections(fci_frags, correction_type=mode, projectors=projectors,
                                               test_extcorr=True)

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
