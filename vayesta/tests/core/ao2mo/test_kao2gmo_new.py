import pytest
import unittest
import itertools

import numpy as np

import pyscf
import pyscf.pbc
import pyscf.pbc.gto
import pyscf.pbc.df
import pyscf.pbc.tools

from vayesta.core.ao2mo import kao2gmo_cderi
from vayesta.tests.common import TestCase
from vayesta.tests import testsystems


@pytest.mark.slow
class KAO2GMO_Tests(TestCase):

    def test_kao2gmo_cderi_3d(self):
        cell = testsystems.he_k321.mol
        kmf = testsystems.he_k321.rhf()
        kmesh = (3, 2, 1)
        kpts = testsystems.he_k321.kpts
        gdf = kmf.with_df

        nmo1, nmo2 = 1, 2
        np.random.seed(1491)
        mo1 = np.random.rand(len(kpts)*cell.nao, nmo1)
        mo2 = np.random.rand(len(kpts)*cell.nao, nmo2)

        scell = pyscf.pbc.tools.super_cell(cell, kmesh)
        gdf_sc = pyscf.pbc.df.GDF(scell)
        gdf_sc.auxbasis = 'def2-svp-ri'
        gdf_sc.build()
        eri_expected = gdf_sc.ao2mo((mo1, mo2, mo1, mo2)).reshape(nmo1, nmo2, nmo1, nmo2)

        options = {'driver' : ['c', 'python'], 'make_real' : [True, False], 'blksize' : [None, 10],
                'tril_kij' : [True, False]}
        options_comb = [dict(zip(options, val)) for val in itertools.product(*options.values())]

        for opts in options_comb:
            cderi, cderi_neg = kao2gmo_cderi(gdf, (mo1, mo2), **opts)
            eri = np.einsum('Lij,Lkl->ijkl', cderi.conj(), cderi)
            self.assertIsNone(cderi_neg)
            self.assertIsNone(np.testing.assert_almost_equal(eri, eri_expected))

    def test_kao2gmo_cderi_2d(self):
        cell = testsystems.he_k32.mol
        kmf = testsystems.he_k32.rhf()
        kmesh = (3, 2, 1)
        kpts = testsystems.he_k32.kpts
        gdf = kmf.with_df

        nmo1, nmo2 = 1, 2
        np.random.seed(15918)
        mo1 = np.random.rand(len(kpts)*cell.nao, nmo1)
        mo2 = np.random.rand(len(kpts)*cell.nao, nmo2)

        scell = pyscf.pbc.tools.super_cell(cell, kmesh)
        gdf_sc = pyscf.pbc.df.GDF(scell)
        gdf_sc.auxbasis = 'def2-svp-ri'
        gdf_sc.build()
        eri_expected = gdf_sc.ao2mo((mo1, mo2, mo1, mo2)).reshape(nmo1, nmo2, nmo1, nmo2)

        options = {'driver' : ['c', 'python'], 'make_real' : [True, False], 'blksize' : [None, 10],
                'tril_kij' : [True, False]}
        options_comb = [dict(zip(options, val)) for val in itertools.product(*options.values())]

        for opts in options_comb:
            cderi, cderi_neg = kao2gmo_cderi(gdf, (mo1, mo2), **opts)
            eri = (np.einsum('Lij,Lkl->ijkl', cderi.conj(), cderi)
                 - np.einsum('Lij,Lkl->ijkl', cderi_neg.conj(), cderi_neg))
            self.assertIsNone(np.testing.assert_almost_equal(eri, eri_expected))

    def test_kao2gmo_cderi_1d(self):
        cell = testsystems.he_k3.mol
        kmf = testsystems.he_k3.rhf()
        kmesh = (3, 1, 1)
        kpts = testsystems.he_k3.kpts
        gdf = kmf.with_df

        nmo1, nmo2 = 1, 2
        np.random.seed(10518)
        mo1 = np.random.rand(len(kpts)*cell.nao, nmo1)
        mo2 = np.random.rand(len(kpts)*cell.nao, nmo2)

        scell = pyscf.pbc.tools.super_cell(cell, kmesh)
        gdf_sc = pyscf.pbc.df.GDF(scell)
        gdf_sc.auxbasis = 'def2-svp-ri'
        gdf_sc.build()
        eri_expected = gdf_sc.ao2mo((mo1, mo2, mo1, mo2)).reshape(nmo1, nmo2, nmo1, nmo2)

        options = {'driver' : ['c', 'python'], 'make_real' : [True, False], 'blksize' : [None, 10],
                'tril_kij' : [True, False]}
        options_comb = [dict(zip(options, val)) for val in itertools.product(*options.values())]

        for opts in options_comb:
            cderi, cderi_neg = kao2gmo_cderi(gdf, (mo1, mo2), **opts)
            eri = (np.einsum('Lij,Lkl->ijkl', cderi.conj(), cderi)
                 - np.einsum('Lij,Lkl->ijkl', cderi_neg.conj(), cderi_neg))
            self.assertIsNone(np.testing.assert_almost_equal(eri, eri_expected))


if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
