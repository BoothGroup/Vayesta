import unittest
import numpy as np
from pyscf import lib
from vayesta.misc import gdf
from pyscf import lib
from pyscf.pbc import gto, df
from pyscf.pbc.lib.kpts_helper import get_kconserv
from vayesta import log
log.setLevel(50)


class GDFTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        L = 5.0
        cls.cell = gto.Cell()
        cls.cell.a = np.eye(3) * L
        cls.cell.atom = 'He 3 2 3; He 1 1 1'
        cls.cell.basis = '6-31g'
        cls.cell.verbose = 0
        cls.cell.max_memory = 1e9
        cls.cell.precision = 1e-12
        cls.cell.build()

        cls.kpts = cls.cell.make_kpts([3,2,1])

        cls.df = gdf.GDF(cls.cell)
        cls.df.auxbasis = 'weigend'
        cls.df.kpts = cls.kpts

        cls.df_ref = df.GDF(cls.cell)
        cls.df_ref.auxbasis = 'weigend'
        cls.df_ref.kpts = cls.kpts

    @classmethod
    def tearDownClass(cls):
        del cls.cell, cls.kpts, cls.df, cls.df_ref

    def _test_eris(self, kpts):
        eri0 = self.df_ref.get_eri(kpts)
        eri1 = self.df.get_eri(kpts)
        self.assertAlmostEqual(np.max(np.abs(eri0-eri1)), 0.0, 8)

    test_kpts_1 = lambda self: self._test_eris((self.kpts[0], self.kpts[0], self.kpts[0], self.kpts[0]))
    test_kpts_2 = lambda self: self._test_eris((self.kpts[1], self.kpts[1], self.kpts[1], self.kpts[1]))
    test_kpts_3 = lambda self: self._test_eris((self.kpts[4], self.kpts[4], self.kpts[4], self.kpts[4]))
    test_kpts_4 = lambda self: self._test_eris((self.kpts[0], self.kpts[1], self.kpts[1], self.kpts[0]))
    test_kpts_5 = lambda self: self._test_eris((self.kpts[1]+5e-8, self.kpts[1]+5e-8, self.kpts[1], self.kpts[1]))
    test_kpts_6 = lambda self: self._test_eris((self.kpts[0]+5e-8, self.kpts[1]+5e-8, self.kpts[1], self.kpts[0]))
    test_kpts_7 = lambda self: self._test_eris(self.kpts[:4])

    def test_cholesky_via_eig(self):
        L = 5.0
        cell = gto.Cell()
        cell.a = np.eye(3) * L
        cell.atom = 'He 3 2 3; He 1 1 1'
        cell.basis = 'cc-pvdz'
        cell.verbose = 0
        cell.max_memory = 1e9
        cell.build()

        df0 = gdf.GDF(cell)
        df0.auxbasis = 'weigend'
        df0.kpts = self.kpts
        df0.linear_dep_always = True
        df0.linear_dep_method = 'none'
        df0.linear_dep_threshold = 1e-10
        df0.build()

        df1 = df.GDF(cell)
        df1.auxbasis = 'weigend'
        df1.kpts = self.kpts
        df1.linear_dep_always = True
        df1.linear_dep_method = 'none'
        df1.linear_dep_threshold = 1e-10
        df1.build()

        eri0 = df0.get_eri((self.kpts[0],)*4)
        eri1 = df1.get_eri((self.kpts[0],)*4)
        self.assertAlmostEqual(np.max(np.abs(eri0-eri1)), 0.0, 8)

    def test_exp_to_discard(self):
        L = 5.0
        cell = gto.Cell()
        cell.a = np.eye(3) * L
        cell.atom = 'He 3 2 3; He 1 1 1'
        cell.basis = 'cc-pvdz'
        cell.verbose = 0
        cell.max_memory = 1e9
        cell.precision = 1e-10
        cell.build()

        df0 = gdf.GDF(cell)
        df0.auxbasis = 'weigend'
        df0.kpts = self.kpts
        df0.exp_to_discard = 1.0
        df0.build()

        df1 = df.GDF(cell)
        df1.auxbasis = 'weigend'
        df1.kpts = self.kpts
        df1.exp_to_discard = 1.0
        df1.build()

        eri0 = df0.get_eri(self.kpts[:4])
        eri1 = df1.get_eri(self.kpts[:4])
        self.assertAlmostEqual(np.max(np.abs(eri0-eri1)), 0.0, 8)

    def test_get_jk(self):
        dm = np.random.random((len(self.kpts), self.cell.nao, self.cell.nao))
        dm = dm + np.random.random((len(self.kpts), self.cell.nao, self.cell.nao)) * 1.0j
        dm = dm + dm.transpose(0, 2, 1).conj()

        j0, k0 = self.df_ref.get_jk(dm)
        j1, k1 = self.df.get_jk(dm)

        self.assertAlmostEqual(np.max(np.abs(j0-j1)), 0.0, 8)
        self.assertAlmostEqual(np.max(np.abs(k0-k1)), 0.0, 8)



if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
