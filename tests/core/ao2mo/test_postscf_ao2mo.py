import itertools
import unittest

import numpy as np

import pyscf
import pyscf.pbc
import pyscf.pbc.gto
import pyscf.pbc.df
import pyscf.pbc.scf
import pyscf.pbc.cc
import pyscf.pbc.tools

from vayesta.core.ao2mo import postscf_kao2gmo
from vayesta.core.ao2mo import postscf_kao2gmo_uhf


class PostSCF_KAO2GMO_Tests(unittest.TestCase):

    def test_rhf(self):
        cell = pyscf.pbc.gto.Cell()
        cell.atom = 'He 0 0 0'
        cell.basis = 'def2-svp'
        cell.a = 3.0*np.eye(3)
        cell.build()
        kmesh = [3, 2, 1]
        kpts = cell.make_kpts(kmesh)
        gdf = pyscf.pbc.df.GDF(cell, kpts)
        gdf.auxbasis = 'def2-svp-ri'
        gdf.build()
        khf = pyscf.pbc.scf.KRHF(cell, kpts)
        khf.with_df = gdf
        khf.conv_tol = 1e-12
        khf.kernel()

        scell = pyscf.pbc.tools.super_cell(cell, kmesh)
        shf = pyscf.pbc.scf.RHF(scell)
        gdf_sc = pyscf.pbc.df.GDF(scell)
        gdf_sc.auxbasis = 'def2-svp-ri'
        gdf_sc.build()
        shf.with_df = gdf_sc
        shf.conv_tol = 1e-12
        shf.kernel()
        assert np.isclose(shf.e_tot / len(kpts), khf.e_tot)

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
                        print("Attribute %s not found" % attr)
                        continue
                    print("Testing class %s attribute %s" % (postscfcls, attr))
                    val = getattr(eris, attr)

                    if attr == 'vvL':
                        expected = np.einsum('iQ,jQ->ij', expected, expected)
                        val = np.einsum('iQ,jQ->ij', val, val)

                    self.assertEqual(val.shape, expected.shape)
                    self.assertIsNone(np.testing.assert_almost_equal(val, expected))

    def test_rhf_2d(self):
        cell = pyscf.pbc.gto.Cell()
        cell.atom = 'He 0 0 0'
        cell.basis = 'def2-svp'
        cell.a = 3.0*np.eye(3)
        cell.a[2,2] = 20.0
        cell.dimension = 2
        cell.build()
        kmesh = [3, 2, 1]
        kpts = cell.make_kpts(kmesh)
        gdf = pyscf.pbc.df.GDF(cell, kpts)
        gdf.auxbasis = 'def2-svp-ri'
        gdf.build()
        khf = pyscf.pbc.scf.KRHF(cell, kpts)
        khf.with_df = gdf
        khf.conv_tol = 1e-12
        khf.kernel()

        scell = pyscf.pbc.tools.super_cell(cell, kmesh)
        shf = pyscf.pbc.scf.RHF(scell)
        gdf_sc = pyscf.pbc.df.GDF(scell)
        gdf_sc.auxbasis = 'def2-svp-ri'
        gdf_sc.build()
        shf.with_df = gdf_sc
        shf.conv_tol = 1e-12
        shf.kernel()
        assert np.isclose(shf.e_tot / len(kpts), khf.e_tot)

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
                        print("Attribute %s not found" % attr)
                        continue
                    print("Testing class %s attribute %s" % (postscfcls, attr))
                    val = getattr(eris, attr)

                    if attr == 'vvL':
                        expected = np.einsum('iQ,jQ->ij', expected, expected)
                        val = np.einsum('iQ,jQ->ij', val, val)

                    self.assertEqual(val.shape, expected.shape)
                    self.assertIsNone(np.testing.assert_almost_equal(val, expected))

    def test_uhf(self):
        cell = pyscf.pbc.gto.Cell()
        cell.atom = 'Li 0 0 0'
        cell.basis = 'def2-svp'
        cell.a = 5.0*np.eye(3)
        cell.spin = 3
        cell.exp_to_discard = 0.1
        cell.build()
        kmesh = [3, 1, 1]
        kpts = cell.make_kpts(kmesh)
        gdf = pyscf.pbc.df.GDF(cell, kpts)
        gdf.auxbasis = 'def2-svp-ri'
        gdf.build()
        khf = pyscf.pbc.scf.KUHF(cell, kpts)
        khf.with_df = gdf
        khf.conv_tol = 1e-10
        khf.kernel()

        scell = pyscf.pbc.tools.super_cell(cell, kmesh)
        assert (scell.spin == cell.spin)
        assert (scell.exp_to_discard == cell.exp_to_discard)
        shf = pyscf.pbc.scf.UHF(scell)
        gdf_sc = pyscf.pbc.df.GDF(scell)
        gdf_sc.auxbasis = 'def2-svp-ri'
        gdf_sc.build()
        shf.with_df = gdf_sc
        shf.conv_tol = 1e-10
        shf.kernel()
        assert np.isclose(shf.e_tot / len(kpts), khf.e_tot)

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
                        print("Attribute %s not found" % attr)
                        continue
                    print("Testing class %s attribute %s" % (postscfcls, attr))
                    val = getattr(eris, attr)

                    if hasattr(expected, 'shape'):
                        self.assertEqual(val.shape, expected.shape)
                    self.assertIsNone(np.testing.assert_almost_equal(val, expected))

if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
