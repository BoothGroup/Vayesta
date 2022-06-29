import unittest
import numpy as np

import pyscf
import pyscf.scf
import pyscf.ao2mo


import vayesta
from vayesta.tests.common import TestCase
from vayesta.tests import testsystems
from vayesta.core.qemb import Embedding, UEmbedding
from vayesta.core.util import cache


class Integral_Test(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.mf = testsystems.water_631g.rhf()

    @classmethod
    def tearDownClass(cls):
        del cls.mf

    @classmethod
    @cache
    def get_embedding(cls):
        if isinstance(cls.mf, pyscf.scf.uhf.UHF):
            emb = UEmbedding(cls.mf)
        else:
            emb = Embedding(cls.mf)
        return emb

    def get_eris_ref(self, mf, mo_coeff):
        if isinstance(mo_coeff, np.ndarray) and mo_coeff.ndim == 2:
            shape = 4*[mo_coeff.shape[-1]]
        else:
            shape = [mo.shape[-1] for mo in mo_coeff]
        if hasattr(mf, 'with_df'):
            eris_ref = mf.with_df.ao2mo(mo_coeff, compact=False).reshape(shape)
        else:
            eris_ref = pyscf.ao2mo.kernel(mf.mol, mo_coeff, compact=False).reshape(shape)
        return eris_ref

    def test_eris_1coeff(self):
        emb = self.get_embedding()
        mf = emb.mf
        nao = mf.mol.nao
        nmo = 3
        np.random.seed(0)
        mo_coeff = np.random.rand(nao, 3)

        eris_ref = self.get_eris_ref(mf, mo_coeff)
        eris = emb.get_eris_array(mo_coeff)
        self.assertAllclose(eris, eris_ref)

        # Density-fitting
        if not hasattr(mf, 'with_df'):
            return
        cderi, cderi_neg = emb.get_cderi(mo_coeff)
        eris = np.einsum('Lij,Lkl->ijkl', cderi, cderi)
        if cderi_neg is not None:
            eris -= np.einsum('Lij,Lkl->ijkl', cderi_neg, cderi_neg)
        self.assertAllclose(eris, eris_ref)

    def test_eris_4coeff(self):
        """Test 4 different coefficients"""
        emb = self.get_embedding()
        mf = emb.mf
        nao = mf.mol.nao
        nmos = [2,3,1,4]
        np.random.seed(0)
        mo_coeffs = [np.random.rand(nao, nmos[i]) for i in range(4)]

        eris_ref = self.get_eris_ref(mf, mo_coeffs)
        eris = emb.get_eris_array(mo_coeffs)
        self.assertAllclose(eris, eris_ref)

        # Density-fitting
        if not hasattr(mf, 'with_df'):
            return
        cderi1, cderi_neg1 = emb.get_cderi(mo_coeffs[:2])
        cderi2, cderi_neg2 = emb.get_cderi(mo_coeffs[2:])
        eris = np.einsum('Lij,Lkl->ijkl', cderi1, cderi2)
        if cderi_neg1 is not None:
            eris -= np.einsum('Lij,Lkl->ijkl', cderi_neg1, cderi_neg2)
        self.assertAllclose(eris, eris_ref)


class Integral_DF_Test(Integral_Test):

    @classmethod
    def setUpClass(cls):
        cls.mf = testsystems.water_631g_df.rhf()

class Integral_UHF_Test(Integral_Test):

    @classmethod
    def setUpClass(cls):
        cls.mf = testsystems.water_cation_631g.uhf()

class Integral_DF_UHF_Test(Integral_Test):

    @classmethod
    def setUpClass(cls):
        cls.mf = testsystems.water_cation_631g_df.uhf()

# PBC

class Integral_PBC_Test(Integral_Test):

    @classmethod
    def setUpClass(cls):
        cls.mf = testsystems.h2_sto3g_s311.rhf()

class Integral_PBC_UHF_Test(Integral_Test):

    @classmethod
    def setUpClass(cls):
        cls.mf = testsystems.h3_sto3g_s311.uhf()

# PBC 2D

class Integral_PBC_2D_Test(Integral_Test):

    @classmethod
    def setUpClass(cls):
        cls.mf = testsystems.h2_sto3g_s31.rhf()

class Integral_PBC_2D_UHF_Test(Integral_Test):

    @classmethod
    def setUpClass(cls):
        cls.mf = testsystems.h3_sto3g_s31.uhf()


# TODO: Figure out why this fails:
#from vayesta.tests.cache import moles, cells
#class IntegralTest(unittest.TestCase):
#
#    def test_eris_solid_2d(self):
#
#        keys = ('graphene_k221', 'rhf')
#        mf = cells[keys[0]][keys[1]]
#        nk = len(mf.kpts)
#        #nk = 1
#        nao = mf.mol.nao*nk
#        nmos = [3,4,5,6]
#        np.random.seed(0)
#        mo_coeffs = [np.random.rand(nao, nmos[i]) for i in range(4)]
#        if mf.mol.spin == 0:
#            emb = Embedding(mf)
#        else:
#            emb = UEmbedding(mf)
#
#        #mf_sc = mf
#        mf_sc = cells[keys[0].replace('_k', '_g')][keys[1]]
#        self.assertAlmostEqual(mf_sc.e_tot/nk, mf.e_tot)
#
#        # --- Test ERIs
#        # Test 1 coefficient:
#        eris = emb.get_eris_array(mo_coeffs[0])
#        eris_expected = mf_sc.with_df.ao2mo(mo_coeffs[0], compact=False).reshape(4*[nmos[0]])
#        self.assertIsNone(np.testing.assert_allclose(eris, eris_expected))
#        # Test 4 coefficients:
#        eris = emb.get_eris_array(mo_coeffs)
#        eris_expected = mf_sc.with_df.ao2mo(mo_coeffs).reshape(nmos)
#        self.assertIsNone(np.testing.assert_allclose(eris, eris_expected))
#
#        # --- Test CD-ERIs
#        # Test 1 coefficient:
#        cderi, cderi_neg = emb.get_cderi(mo_coeffs[0])
#        eris = (np.einsum('Lij,Lkl->ijkl', cderi, cderi)
#              - np.einsum('Lij,Lkl->ijkl', cderi_neg, cderi_neg))
#        eris_expected = mf_sc.with_df.ao2mo(mo_coeffs[0], compact=False).reshape(4*[nmos[0]])
#        self.assertIsNone(np.testing.assert_allclose(eris, eris_expected))
#        # Test 4 coefficient:
#        cderi1, cderi1_neg = emb.get_cderi(mo_coeffs[:2])
#        cderi2, cderi2_neg = emb.get_cderi(mo_coeffs[2:])
#        eris = (np.einsum('Lij,Lkl->ijkl', cderi1, cderi2)
#              - np.einsum('Lij,Lkl->ijkl', cderi1_neg, cderi2_neg))
#        eris_expected = mf_sc.with_df.ao2mo(mo_coeffs, compact=False).reshape(nmos)
#        self.assertIsNone(np.testing.assert_allclose(eris, eris_expected))


if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
