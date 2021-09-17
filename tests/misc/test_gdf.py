import unittest
import numpy as np
from pyscf import lib
from vayesta.misc import gdf
from pyscf import lib
from pyscf.pbc import gto, df
from pyscf.pbc.lib.kpts_helper import get_kconserv


class GDFTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        L = 5.0
        n = 11
        cls.cell = gto.Cell()
        cls.cell.a = np.eye(3) * L
        cls.cell.mesh = [n, n, n]
        cls.cell.atom = 'He 3 2 3; He 1 1 1'
        cls.cell.basis = 'cc-pvdz'
        cls.cell.verbose = 0
        cls.cell.max_memory = 1e9
        cls.cell.build()

        np.random.seed(1)
        cls.kpts = np.random.random((5, 3))
        cls.kpts[0] = 0
        cls.kpts[3] = cls.kpts[0] - cls.kpts[1] + cls.kpts[2]
        cls.kpts[4] *= 1e-5

        cls.df = gdf.GDF(cls.cell)
        cls.df.linear_dep_threshold = 1e-7
        cls.df.auxbasis = 'weigend'
        cls.df.kpts = cls.kpts

        cls.df_ref = df.GDF(cls.cell)
        cls.df_ref.linear_dep_threshold = 1e-7
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
        factor = 10
        L = 10.0
        n = 11
        cell = gto.Cell()
        cell.a = np.eye(3) * L
        cell.mesh = [n, n, n]
        cell.atom = 'He {a} {b} {a}; He {c} {c} {c}'.format(a=3*factor, b=2*factor, c=factor)
        cell.basis = 'cc-pvdz'
        cell.verbose = 0
        cell.max_memory = 1e9
        cell.build()

        df0 = gdf.GDF(cell)
        df0.auxbasis = 'weigend'
        df0.kpts = self.kpts
        df0.build()

        df1 = df.GDF(cell)
        df1.auxbasis = 'weigend'
        df1.kpts = self.kpts
        df1.build()

        eri0 = self.df.get_eri(self.kpts[:4])
        eri1 = self.df_ref.get_eri(self.kpts[:4])
        self.assertAlmostEqual(np.max(np.abs(eri0-eri1)), 0.0, 8)


if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
