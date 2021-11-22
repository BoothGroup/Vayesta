import unittest
import numpy as np
from pyscf import lib
from vayesta.misc import gdf
from pyscf import lib
from pyscf.pbc import gto, df, tools
from pyscf.pbc.lib.kpts_helper import get_kconserv
from vayesta import log


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
        cell = self.cell.copy()
        cell.basis = 'cc-pvdz'
        cell.precision = 1e-8
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
        cell = self.cell.copy()
        cell.basis = 'cc-pvdz'
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

        j0, k0 = self.df_ref.get_jk(dm, exxdiv='ewald')
        j1, k1 = self.df.get_jk(dm, exxdiv='ewald')

        self.assertAlmostEqual(np.max(np.abs(j0-j1)), 0.0, 8)
        self.assertAlmostEqual(np.max(np.abs(k0-k1)), 0.0, 8)

    def test_cart(self):
        cell = self.cell.copy()
        cell.cart = True
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

    def test_sr_loop(self):
        df0, df1 = self.df, self.df_ref

        r0, i0, s0 = list(df0.sr_loop(self.kpts[:2], blksize=df0.get_naoaux()))[0]
        r1, i1, s1 = list(df1.sr_loop(self.kpts[:2], blksize=df0.get_naoaux()))[0]
        l0 = (r0 + 1.0j * i0).reshape(-1, self.cell.nao, self.cell.nao)
        l1 = (r1 + 1.0j * i1).reshape(-1, self.cell.nao, self.cell.nao)

        eri0 = np.einsum('Lpq,Lrs->pqrs', l0, l0)
        eri1 = np.einsum('Lpq,Lrs->pqrs', l1, l1)
        self.assertAlmostEqual(np.max(np.abs(eri0-eri1)), 0.0, 8)

        r0, i0, s0 = list(df0.sr_loop(np.zeros((2, 3)), blksize=df0.get_naoaux()))[0]
        r1, i1, s1 = list(df1.sr_loop(np.zeros((2, 3)), blksize=df0.get_naoaux()))[0]
        l0 = (r0 + 1.0j * i0).reshape(-1, self.cell.nao*(self.cell.nao+1)//2)
        l1 = (r1 + 1.0j * i1).reshape(-1, self.cell.nao*(self.cell.nao+1)//2)

        eri0 = np.einsum('Lp,Lq->pq', l0, l0)
        eri1 = np.einsum('Lp,Lq->pq', l1, l1)
        self.assertAlmostEqual(np.max(np.abs(eri0-eri1)), 0.0, 8)

    def test_reset(self):
        df0 = gdf.GDF(self.cell, self.kpts)
        df0.build(with_j3c=False)
        df0.reset()
        self.assertEqual(df0.auxcell, None)

    def test_save_load(self):
        df0 = self.df

        df0.save('test.npy')

        df = gdf.GDF(self.cell)
        df.auxbasis = 'weigend'
        df.kpts = self.kpts
        df.build(with_j3c=False)
        df.load('test.npy')

        eri0 = df.get_eri(self.kpts[:4])
        eri1 = df0.get_eri(self.kpts[:4])
        self.assertAlmostEqual(np.max(np.abs(eri0-eri1)), 0.0, 8)

    def test_ao2mo(self):
        df0, df1 = self.df, self.df_ref

        c1 = np.random.random((self.cell.nao, 4)) + 1.0j * np.random.random((self.cell.nao, 4))
        c2 = np.random.random((self.cell.nao, 8)) + 1.0j * np.random.random((self.cell.nao, 8))
        c3 = np.random.random((self.cell.nao, 4)) + 1.0j * np.random.random((self.cell.nao, 4))
        c4 = np.random.random((self.cell.nao, 8)) + 1.0j * np.random.random((self.cell.nao, 8))

        eri0 = df0.ao2mo((c1, c2, c3, c4), self.kpts[:4])
        eri1 = df1.ao2mo((c1, c2, c3, c4), self.kpts[:4])
        self.assertAlmostEqual(np.max(np.abs(eri0-eri1)), 0.0, 8)

        c1 = np.random.random((self.cell.nao, 4))

        eri0 = df0.ao2mo((c1, c1, c1, c1), np.zeros((4, 3)), compact=True)
        eri1 = df1.ao2mo((c1, c1, c1, c1), np.zeros((4, 3)), compact=True)
        self.assertAlmostEqual(np.max(np.abs(eri0-eri1)), 0.0, 8)

    def test_3c_eri(self):
        df0, df1 = self.df, self.df_ref

        Lij = df0.get_3c_eri(self.kpts[:2], compact=False)
        Lij = Lij.reshape(-1, self.cell.nao, self.cell.nao)
        eri0 = np.einsum('Lij,Lkl->ijkl', Lij, Lij)
        eri1 = df1.get_eri(list(self.kpts[:2])*2).reshape((self.cell.nao,)*4)
        self.assertAlmostEqual(np.max(np.abs(eri0-eri1)), 0.0, 8)

        Lij = df0.get_3c_eri(np.zeros((2, 3)), compact=True)
        Lij = Lij.reshape(-1, self.cell.nao*(self.cell.nao+1)//2)
        eri0 = np.einsum('Li,Lj->ij', Lij, Lij)
        eri1 = df1.get_eri(np.zeros((4, 3))).reshape((self.cell.nao*(self.cell.nao+1)//2,)*2)
        self.assertAlmostEqual(np.max(np.abs(eri0-eri1)), 0.0, 8)

    def test_ao2mo_3c(self):
        df0, df1 = self.df, self.df_ref

        c1 = np.random.random((self.cell.nao, 4)) + 1.0j * np.random.random((self.cell.nao, 4))
        c2 = np.random.random((self.cell.nao, 8)) + 1.0j * np.random.random((self.cell.nao, 8))

        Lij = df0.ao2mo_3c((c1, c2), self.kpts[:2], compact=False)
        Lij = Lij.reshape(-1, c1.shape[-1], c2.shape[-1])
        eri0 = np.einsum('Lij,Lkl->ijkl', Lij, Lij)
        eri1 = df1.ao2mo((c1, c2, c1, c2), list(self.kpts[:2])*2).reshape(c1.shape[-1], c2.shape[-1], c1.shape[-1], c2.shape[-1])
        self.assertAlmostEqual(np.max(np.abs(eri0-eri1)), 0.0, 8)

        c1 = np.random.random((self.cell.nao, 4))

        Lij = df0.ao2mo_3c((c1, c1), np.zeros((2, 3)), compact=True)
        Lij = Lij.reshape(-1, c1.shape[-1]*(c1.shape[-1]+1)//2)
        eri0 = np.einsum('Li,Lj->ij', Lij, Lij)
        eri1 = df1.ao2mo((c1, c1, c1, c1), np.zeros((4, 3))).reshape((c1.shape[-1]*(c1.shape[-1]+1)//2,)*2)
        self.assertAlmostEqual(np.max(np.abs(eri0-eri1)), 0.0, 8)

    def test_properties(self):
        df0, df1 = self.df, self.df_ref

        df0.get_nuc()
        self.assertAlmostEqual(np.max(np.abs(df0.get_nuc()-df1.get_nuc())), 0.0, 8)

        df0.get_pp()
        self.assertAlmostEqual(np.max(np.abs(df0.get_pp()-df1.get_pp())), 0.0, 8)

        df0.get_ovlp()
        ovlp = np.array(self.cell.pbc_intor('int1e_ovlp', hermi=1, kpts=self.kpts))
        self.assertAlmostEqual(np.max(np.abs(np.array(df0.get_ovlp())-ovlp)), 0.0, 8)

        self.assertAlmostEqual(np.abs(tools.pbc.madelung(self.cell, self.kpts) - df0.madelung), 0.0, 8)


if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
