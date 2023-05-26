import pytest
import unittest

import numpy as np

import pyscf

from vayesta.core.ao2mo import postscf_kao2gmo
from vayesta.core.ao2mo import postscf_kao2gmo_uhf
from vayesta.tests.common import TestCase
from vayesta.tests import testsystems


@pytest.mark.slow
class PostSCF_KAO2GMO_Tests(TestCase):

    def test_rhf(self):
        cell = testsystems.he_k321.mol
        khf = testsystems.he_k321.rhf()
        kpts = testsystems.he_k321.kpts
        gdf = khf.with_df
        scell = testsystems.he_s321.mol
        shf = testsystems.he_s321.rhf()
        self.assertAlmostEqual(shf.e_tot / len(kpts), khf.e_tot)

        nao = scell.nao
        nocc = np.count_nonzero(shf.mo_occ > 0)
        np.random.seed(2291)
        mo2 = np.random.rand(nao, nao)
        nactive_occ = 2
        nactive_vir = 3
        active = list(range(nocc-nactive_occ, nocc+nactive_vir))
        frozen = list(range(0, nocc-nactive_occ)) + list(range(nocc+nactive_vir, nao))

        fock = shf.get_fock()
        mo_energy = np.einsum('ai,ab,bi->i', shf.mo_coeff[:,active], fock, shf.mo_coeff[:,active])
        mo_energy2 = np.einsum('ai,ab,bi->i', mo2[:,active], fock, mo2[:,active])
        e_hf = shf.e_tot

        for postscfcls in [pyscf.cc.ccsd.CCSD, pyscf.cc.rccsd.RCCSD, pyscf.cc.dfccsd.RCCSD]:

            postscf = postscfcls(shf, frozen=frozen)
            eris_expected_1 = postscf.ao2mo()
            eris_expected_2 = postscf.ao2mo(mo_coeff=mo2)

            eris_1 = postscf_kao2gmo(postscf, gdf, fock=fock, mo_energy=mo_energy, e_hf=e_hf)
            eris_2 = postscf_kao2gmo(postscf, gdf, fock=fock, mo_energy=mo_energy2, e_hf=e_hf, mo_coeff=mo2)

            for eris, eris_expected in [(eris_1, eris_expected_1), (eris_2, eris_expected_2)]:
                for attr in ['oooo', 'ovoo', 'oovv', 'ovvo', 'ovov', 'ovvv', 'vvvv', 'vvL', 'fock', 'mo_energy', 'e_hf']:
                    expected = getattr(eris_expected, attr, None)
                    if expected is None:
                        continue
                    val = getattr(eris, attr)

                    if attr == 'vvL':
                        expected = np.einsum('iQ,jQ->ij', expected, expected)
                        val = np.einsum('iQ,jQ->ij', val, val)

                    self.assertEqual(val.shape, expected.shape)
                    self.assertIsNone(np.testing.assert_almost_equal(val, expected))

    def test_rhf_2d(self):
        cell = testsystems.he_k32.mol
        khf = testsystems.he_k32.rhf()
        kpts = testsystems.he_k32.kpts
        gdf = khf.with_df
        scell = testsystems.he_s32.mol
        shf = testsystems.he_s32.rhf()
        self.assertAlmostEqual(shf.e_tot / len(kpts), khf.e_tot)

        nao = scell.nao
        nocc = np.count_nonzero(shf.mo_occ > 0)
        np.random.seed(5018)
        mo2 = np.random.rand(nao, nao)
        nactive_occ = 2
        nactive_vir = 3
        active = list(range(nocc-nactive_occ, nocc+nactive_vir))
        frozen = list(range(0, nocc-nactive_occ)) + list(range(nocc+nactive_vir, nao))

        fock = shf.get_fock()
        mo_energy = np.einsum('ai,ab,bi->i', shf.mo_coeff[:,active], fock, shf.mo_coeff[:,active])
        mo_energy2 = np.einsum('ai,ab,bi->i', mo2[:,active], fock, mo2[:,active])
        e_hf = shf.e_tot

        eriblocks = ['oooo', 'ovoo', 'oovv', 'ovvo', 'ovov', 'ovvv', 'vvvv', 'vvL']

        for postscfcls in [pyscf.cc.ccsd.CCSD, pyscf.cc.rccsd.RCCSD]:

            postscf = postscfcls(shf, frozen=frozen)
            eris_expected_1 = postscf.ao2mo()
            eris_expected_2 = postscf.ao2mo(mo_coeff=mo2)

            eris_1 = postscf_kao2gmo(postscf, gdf, fock=fock, mo_energy=mo_energy, e_hf=e_hf)
            eris_2 = postscf_kao2gmo(postscf, gdf, fock=fock, mo_energy=mo_energy2, e_hf=e_hf, mo_coeff=mo2)

            for eris, eris_expected in [(eris_1, eris_expected_1), (eris_2, eris_expected_2)]:
                for attr in (eriblocks + ['fock', 'mo_energy', 'e_hf']):
                    expected = getattr(eris_expected, attr, None)
                    if expected is None:
                        continue
                    val = getattr(eris, attr)

                    if attr == 'vvL':
                        expected = np.einsum('iQ,jQ->ij', expected, expected)
                        val = np.einsum('iQ,jQ->ij', val, val)

                    self.assertEqual(val.shape, expected.shape)
                    self.assertIsNone(np.testing.assert_almost_equal(val, expected))

    def test_uhf(self):
        cell = testsystems.lih_k221.mol
        khf = testsystems.lih_k221.uhf()
        kpts = testsystems.lih_k221.kpts
        gdf = khf.with_df
        scell = testsystems.lih_s221.mol
        shf = testsystems.lih_s221.uhf()
        self.assertAlmostEqual(shf.e_tot / len(kpts), khf.e_tot)

        nao = scell.nao
        nocc = np.count_nonzero(shf.mo_occ > 0)
        np.random.seed(4591)
        mo2a = np.random.rand(nao, nao)
        mo2b = np.random.rand(nao, nao)
        mo2 = (mo2a, mo2b)
        nactive_occ = 2
        nactive_vir = 3
        active = list(range(nocc-nactive_occ, nocc+nactive_vir))
        frozen = list(range(0, nocc-nactive_occ)) + list(range(nocc+nactive_vir, nao))

        fock = shf.get_fock()
        mo_energy_a = np.einsum('ai,ab,bi->i', shf.mo_coeff[0][:,active], fock[0], shf.mo_coeff[0][:,active])
        mo_energy_b = np.einsum('ai,ab,bi->i', shf.mo_coeff[1][:,active], fock[1], shf.mo_coeff[1][:,active])
        mo_energy = (mo_energy_a, mo_energy_b)
        mo_energy2_a = np.einsum('ai,ab,bi->i', mo2[0][:,active], fock[0], mo2[0][:,active])
        mo_energy2_b = np.einsum('ai,ab,bi->i', mo2[1][:,active], fock[1], mo2[1][:,active])
        mo_energy2 = (mo_energy2_a, mo_energy2_b)
        e_hf = shf.e_tot

        eriblocks_aa = ['oooo', 'ovoo', 'oovv', 'ovvo', 'ovov', 'ovvv', 'vvvv']
        eriblocks_ab = ['%s%s' % (x[:2], x[2:].upper()) for x in eriblocks_aa]
        eriblocks_bb = [x.upper() for x in eriblocks_aa]
        eriblocks_ba = ['OVoo', 'OOvv', 'OVvo', 'OVvv']
        eriblocks = eriblocks_aa + eriblocks_bb + eriblocks_ab + eriblocks_ba

        for postscfcls in [pyscf.cc.uccsd.UCCSD]:

            postscf = postscfcls(shf, frozen=frozen)
            eris_expected_1 = postscf.ao2mo()
            eris_expected_2 = postscf.ao2mo(mo_coeff=mo2)

            eris_1 = postscf_kao2gmo_uhf(postscf, gdf, fock=fock, mo_energy=mo_energy, e_hf=e_hf)
            eris_2 = postscf_kao2gmo_uhf(postscf, gdf, fock=fock, mo_energy=mo_energy2, e_hf=e_hf, mo_coeff=mo2)

            for eris, eris_expected in [(eris_1, eris_expected_1), (eris_2, eris_expected_2)]:
                for attr in (eriblocks + ['fock', 'mo_energy', 'e_hf']):
                    expected = getattr(eris_expected, attr, None)
                    if expected is None:
                        continue
                    val = getattr(eris, attr)
                    self.assertAllclose(val, expected)


if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
