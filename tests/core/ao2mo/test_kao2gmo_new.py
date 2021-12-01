import itertools
import unittest

import numpy as np

import pyscf
import pyscf.pbc
import pyscf.pbc.gto
import pyscf.pbc.df
import pyscf.pbc.tools

from vayesta.core.ao2mo import kao2gmo_cderi


class KAO2GMO_Tests(unittest.TestCase):

    def test_kao2gmo_cderi_3d(self):
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
            print("Options: %r" % opts)
            cderi, cderi_neg = kao2gmo_cderi(gdf, (mo1, mo2), **opts)
            eri = np.einsum('Lij,Lkl->ijkl', cderi.conj(), cderi)
            self.assertIsNone(cderi_neg)
            self.assertIsNone(np.testing.assert_almost_equal(eri, eri_expected))

    def test_kao2gmo_cderi_2d(self):
        cell = pyscf.pbc.gto.Cell()
        cell.atom = 'He 0 0 0'
        cell.basis = 'def2-svp'
        cell.a = 3.0*np.eye(3)
        cell.a[2,2] = 30.0
        cell.dimension = 2
        cell.build()
        kmesh = [3, 2, 1]
        kpts = cell.make_kpts(kmesh)
        gdf = pyscf.pbc.df.GDF(cell, kpts)
        gdf.auxbasis = 'def2-svp-ri'
        gdf.build()
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
            print("Options: %r" % opts)
            cderi, cderi_neg = kao2gmo_cderi(gdf, (mo1, mo2), **opts)
            eri = (np.einsum('Lij,Lkl->ijkl', cderi.conj(), cderi)
                 - np.einsum('Lij,Lkl->ijkl', cderi_neg.conj(), cderi_neg))
            self.assertIsNone(np.testing.assert_almost_equal(eri, eri_expected))

    def test_kao2gmo_cderi_1d(self):
        cell = pyscf.pbc.gto.Cell()
        cell.atom = 'He 0 0 0'
        cell.basis = 'def2-svp'
        cell.a = 3.0*np.eye(3)
        cell.a[1,1] = cell.a[2,2] = 30.0
        cell.dimension = 1
        cell.build()
        kmesh = [3, 1, 1]
        kpts = cell.make_kpts(kmesh)
        gdf = pyscf.pbc.df.GDF(cell, kpts)
        gdf.auxbasis = 'def2-svp-ri'
        gdf.build()
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
            print("Options: %r" % opts)
            cderi, cderi_neg = kao2gmo_cderi(gdf, (mo1, mo2), **opts)
            eri = (np.einsum('Lij,Lkl->ijkl', cderi.conj(), cderi)
                 - np.einsum('Lij,Lkl->ijkl', cderi_neg.conj(), cderi_neg))
            self.assertIsNone(np.testing.assert_almost_equal(eri, eri_expected))

if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
