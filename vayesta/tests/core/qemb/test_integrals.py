import unittest
import numpy as np

import pyscf
import pyscf.ao2mo


import vayesta
import vayesta.core
import vayesta.tests

from vayesta.tests.cache import moles, cells
from vayesta.core import Embedding, UEmbedding



class IntegralTest(unittest.TestCase):

    def test_eris(self):

        for keys in (('h2o_ccpvdz', 'rhf'), ('no2_ccpvdz','uhf')):
            print("Now testing %s %s..." % tuple(keys))
            mf = moles[keys[0]][keys[1]]
            nao = mf.mol.nao
            nmos = [3,4,5,6]
            np.random.seed(0)
            mo_coeffs = [np.random.rand(nao, nmos[i]) for i in range(4)]
            if mf.mol.spin == 0:
                emb = Embedding(mf)
            else:
                emb = UEmbedding(mf)

            # --- Test ERIs
            # Test 1 coefficient:
            eris = emb.get_eris_array(mo_coeffs[0])
            eris_expected = pyscf.ao2mo.kernel(mf.mol, mo_coeffs[0], compact=False).reshape(4*[nmos[0]])
            self.assertIsNone(np.testing.assert_allclose(eris, eris_expected))
            # Test 4 different coefficients:
            eris = emb.get_eris_array(mo_coeffs)
            eris_expected = pyscf.ao2mo.kernel(mf.mol, mo_coeffs).reshape(nmos)
            self.assertIsNone(np.testing.assert_allclose(eris, eris_expected))

    def test_eris_df(self):

        for keys in (('h2o_ccpvdz_df', 'rhf'), ('no2_ccpvdz_df','uhf')):
            print("Now testing %s %s..." % tuple(keys))
            mf = moles[keys[0]][keys[1]]
            nao = mf.mol.nao
            nmos = [3,4,5,6]
            np.random.seed(0)
            mo_coeffs = [np.random.rand(nao, nmos[i]) for i in range(4)]
            if mf.mol.spin == 0:
                emb = Embedding(mf)
            else:
                emb = UEmbedding(mf)

            # --- Test ERIs
            # Test 1 coefficient:
            eris = emb.get_eris_array(mo_coeffs[0])
            eris_expected = mf.with_df.ao2mo(mo_coeffs[0], compact=False).reshape(4*[nmos[0]])
            self.assertIsNone(np.testing.assert_allclose(eris, eris_expected))
            # Test 4 coefficients:
            eris = emb.get_eris_array(mo_coeffs)
            eris_expected = mf.with_df.ao2mo(mo_coeffs).reshape(nmos)
            self.assertIsNone(np.testing.assert_allclose(eris, eris_expected))

            # --- Test CD-ERIs
            # Test 1 coefficient:
            cderi, cderi_neg = emb.get_cderi(mo_coeffs[0])
            self.assertIsNone(cderi_neg)
            eris = np.einsum('Lij,Lkl->ijkl', cderi, cderi)
            eris_expected = mf.with_df.ao2mo(mo_coeffs[0], compact=False).reshape(4*[nmos[0]])
            self.assertIsNone(np.testing.assert_allclose(eris, eris_expected))
            # Test 4 coefficient:
            cderi1, cderi_neg1 = emb.get_cderi(mo_coeffs[:2])
            cderi2, cderi_neg2 = emb.get_cderi(mo_coeffs[2:])
            self.assertIsNone(cderi_neg1)
            self.assertIsNone(cderi_neg2)
            eris = np.einsum('Lij,Lkl->ijkl', cderi1, cderi2)
            eris_expected = mf.with_df.ao2mo(mo_coeffs, compact=False).reshape(nmos)
            self.assertIsNone(np.testing.assert_allclose(eris, eris_expected))

    def test_eris_solid(self):

        for keys in (('lih_k221', 'rhf'), ('boron_cp_k321', 'uhf')):
            print("Now testing %s %s..." % tuple(keys))
            mf = cells[keys[0]][keys[1]]
            nk = len(mf.kpts)
            nao = mf.mol.nao*nk
            nmos = [3,4,5,6]
            np.random.seed(0)
            mo_coeffs = [np.random.rand(nao, nmos[i]) for i in range(4)]
            if mf.mol.spin == 0:
                emb = Embedding(mf)
            else:
                emb = UEmbedding(mf)

            mf_sc = cells[keys[0].replace('_k', '_g')][keys[1]]
            self.assertAlmostEqual(mf_sc.e_tot/nk, mf.e_tot)

            # --- Test ERIs
            # Test 1 coefficient:
            eris = emb.get_eris_array(mo_coeffs[0])
            eris_expected = mf_sc.with_df.ao2mo(mo_coeffs[0], compact=False).reshape(4*[nmos[0]])
            self.assertIsNone(np.testing.assert_allclose(eris, eris_expected))
            # Test 4 coefficients:
            eris = emb.get_eris_array(mo_coeffs)
            eris_expected = mf_sc.with_df.ao2mo(mo_coeffs).reshape(nmos)
            self.assertIsNone(np.testing.assert_allclose(eris, eris_expected))

            # --- Test CD-ERIs
            # Test 1 coefficient:
            cderi, cderi_neg = emb.get_cderi(mo_coeffs[0])
            self.assertIsNone(cderi_neg)
            eris = np.einsum('Lij,Lkl->ijkl', cderi, cderi)
            eris_expected = mf_sc.with_df.ao2mo(mo_coeffs[0], compact=False).reshape(4*[nmos[0]])
            self.assertIsNone(np.testing.assert_allclose(eris, eris_expected))
            # Test 4 coefficient:
            cderi1, cderi_neg1 = emb.get_cderi(mo_coeffs[:2])
            cderi2, cderi_neg2 = emb.get_cderi(mo_coeffs[2:])
            self.assertIsNone(cderi_neg1)
            self.assertIsNone(cderi_neg2)
            eris = np.einsum('Lij,Lkl->ijkl', cderi1, cderi2)
            eris_expected = mf_sc.with_df.ao2mo(mo_coeffs, compact=False).reshape(nmos)
            self.assertIsNone(np.testing.assert_allclose(eris, eris_expected))

    def test_eris_solid_2d(self):

        for keys in (('graphene_k221', 'rhf'), ('hydrogen_cubic_2d_k221', 'uhf')):
            print("Now testing %s %s..." % tuple(keys))
            mf = cells[keys[0]][keys[1]]
            nk = len(mf.kpts)
            nao = mf.mol.nao*nk
            nmos = [3,4,5,6]
            np.random.seed(0)
            mo_coeffs = [np.random.rand(nao, nmos[i]) for i in range(4)]
            if mf.mol.spin == 0:
                emb = Embedding(mf)
            else:
                emb = UEmbedding(mf)

            mf_sc = cells[keys[0].replace('_k', '_g')][keys[1]]
            self.assertAlmostEqual(mf_sc.e_tot/nk, mf.e_tot)

            # --- Test ERIs
            # Test 1 coefficient:
            eris = emb.get_eris_array(mo_coeffs[0])
            eris_expected = mf_sc.with_df.ao2mo(mo_coeffs[0], compact=False).reshape(4*[nmos[0]])
            self.assertIsNone(np.testing.assert_allclose(eris, eris_expected))
            # Test 4 coefficients:
            eris = emb.get_eris_array(mo_coeffs)
            eris_expected = mf_sc.with_df.ao2mo(mo_coeffs).reshape(nmos)
            self.assertIsNone(np.testing.assert_allclose(eris, eris_expected))

            # --- Test CD-ERIs
            # Test 1 coefficient:
            cderi, cderi_neg = emb.get_cderi(mo_coeffs[0])
            eris = (np.einsum('Lij,Lkl->ijkl', cderi, cderi)
                  - np.einsum('Lij,Lkl->ijkl', cderi_neg, cderi_neg))
            eris_expected = mf_sc.with_df.ao2mo(mo_coeffs[0], compact=False).reshape(4*[nmos[0]])
            self.assertIsNone(np.testing.assert_allclose(eris, eris_expected))
            # Test 4 coefficient:
            cderi1, cderi1_neg = emb.get_cderi(mo_coeffs[:2])
            cderi2, cderi2_neg = emb.get_cderi(mo_coeffs[2:])
            eris = (np.einsum('Lij,Lkl->ijkl', cderi1, cderi2)
                  - np.einsum('Lij,Lkl->ijkl', cderi1_neg, cderi2_neg))
            eris_expected = mf_sc.with_df.ao2mo(mo_coeffs, compact=False).reshape(nmos)
            self.assertIsNone(np.testing.assert_allclose(eris, eris_expected))


if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
