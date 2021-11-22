import unittest
import numpy as np
import pyscf.gto
import pyscf.scf
import pyscf.pbc.gto
import pyscf.pbc.scf
import pyscf.pbc.tools
from pyscf import lib
from vayesta.misc import gdf
from vayesta.core import QEmbeddingFragment, QEmbeddingMethod
from vayesta.core.bath import DMET_Bath
# UHF:
from vayesta.core import UEmbedding, UFragment


class temporary_seed:
    def __init__(self, seed):
        self.seed, self.state = seed, None
    def __enter__(self):
        self.state = np.random.get_state()
        np.random.seed(self.seed)
    def __exit__(self, *args):
        np.random.set_state(self.state)


class MolFragmentTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.mol = pyscf.gto.Mole()
        cls.mol.atom = 'O1 0 0 0.118; O2 0 0.755 -0.471; O3 0 -0.755 -0.471'
        cls.mol.basis = 'cc-pvdz'
        cls.mol.verbose = 0
        cls.mol.build()
        cls.mf = pyscf.scf.RHF(cls.mol)
        cls.mf = cls.mf.density_fit()
        cls.mf.conv_tol = 1e-12
        cls.mf.kernel()
        assert cls.mf.converged
        # UHF
        cls.uhf = pyscf.scf.UHF(cls.mol)
        cls.uhf = cls.uhf.density_fit()
        cls.uhf.conv_tol = 1e-12
        cls.uhf.kernel()
        assert cls.uhf.converged

    @classmethod
    def tearDownClass(cls):
        del cls.mol, cls.mf

    def test_iao_atoms(self):
        qemb = QEmbeddingMethod(self.mf)
        qemb.iao_fragmentation()
        frags = qemb.add_all_atomic_fragments()

        e_elec = sum([f.get_fragment_mf_energy() for f in frags])
        self.assertAlmostEqual(e_elec + self.mf.energy_nuc(), self.mf.e_tot, 8)
        self.assertAlmostEqual(frags[0].get_fragment_mf_energy(), -107.09600995056937, 8)
        self.assertAlmostEqual(frags[1].get_fragment_mf_energy(), -104.94355748773579, 8)
        self.assertAlmostEqual(frags[2].get_fragment_mf_energy(), -104.94355748773579, 8)

        tr = lambda c: np.einsum('xi,xi->', c, c.conj())

        c_bath, c_occenv, c_virenv = DMET_Bath(frags[0], frags[0].opts.dmet_threshold).make_dmet_bath(frags[0].c_env)
        self.assertAlmostEqual(tr(c_bath), 5.305607912416594, 8)
        self.assertAlmostEqual(tr(c_occenv), 6.0044522544970995, 8)
        self.assertAlmostEqual(tr(c_virenv), 349.9785855174963, 8)

        frags[1].opts.wf_partition = 'first-vir'
        c_bath, c_occenv, c_virenv = DMET_Bath(frags[1], frags[1].opts.dmet_threshold).make_dmet_bath(frags[1].c_env)
        self.assertAlmostEqual(tr(c_bath), 7.357949737669514, 8)
        self.assertAlmostEqual(tr(c_occenv), 5.4998286299420664, 8)
        self.assertAlmostEqual(tr(c_virenv), 349.9785855174961, 8)

        frags[2].opts.wf_partition = 'democratic'
        c_bath, c_occenv, c_virenv = DMET_Bath(frags[2], frags[2].opts.dmet_threshold).make_dmet_bath(frags[2].c_env)
        self.assertAlmostEqual(tr(c_bath), 7.357949737669514, 8)
        self.assertAlmostEqual(tr(c_occenv), 5.4998286299420664, 8)
        self.assertAlmostEqual(tr(c_virenv), 349.9785855174961, 8)

        for frag in frags[0].loop_fragments():
            frag.reset()
            self.assertTrue(frag._c_active_occ is None)
            self.assertTrue(frag._c_active_vir is None)
            self.assertTrue(frag._c_frozen_occ is None)
            self.assertTrue(frag._c_frozen_vir is None)

    def test_iao_aos(self):
        qemb = QEmbeddingMethod(self.mf)
        qemb.iao_fragmentation()

        frag = qemb.add_orbital_fragment([0, 1])
        self.assertAlmostEqual(frag.get_fragment_mf_energy(), -72.99138042535633, 8)

        frag = qemb.add_atomic_fragment([0], orbital_filter=['1s', '2s'])
        self.assertAlmostEqual(frag.get_fragment_mf_energy(), -72.99138042535633, 8)

    def test_lowdin_atoms(self):
        qemb = QEmbeddingMethod(self.mf)
        qemb.sao_fragmentation()
        frags = [qemb.add_atomic_fragment(['O%d'%x]) for x in range(1, 4)]

        self.assertAlmostEqual(frags[0].get_fragment_mf_energy(), -108.51371286149299, 8)
        self.assertAlmostEqual(frags[1].get_fragment_mf_energy(), -104.23470603227311, 8)
        self.assertAlmostEqual(frags[2].get_fragment_mf_energy(), -104.23470603227311, 8)
        e_mf = sum([f.get_fragment_mf_energy() for f in frags])
        self.assertAlmostEqual(e_mf, (self.mf.e_tot-self.mf.energy_nuc()), 8)

    def test_sao_atomic_fragment_uhf(self):
        qemb = UEmbedding(self.uhf)
        qemb.sao_fragmentation()
        frags = [qemb.add_atomic_fragment(['O%d'%x]) for x in range(1, 4)]

        self.assertAlmostEqual(frags[0].get_fragment_mf_energy(), -108.51371286149299, 6)
        self.assertAlmostEqual(frags[1].get_fragment_mf_energy(), -104.23470603227311, 6)
        self.assertAlmostEqual(frags[2].get_fragment_mf_energy(), -104.23470603227311, 6)
        e_mf = sum([f.get_fragment_mf_energy() for f in frags])
        self.assertAlmostEqual(e_mf, (self.uhf.e_tot-self.uhf.energy_nuc()), 8)

    def test_lowdin_aos(self):
        qemb = QEmbeddingMethod(self.mf)
        qemb.sao_fragmentation()

        frag = qemb.add_orbital_fragment([0, 1])
        self.assertAlmostEqual(frag.get_fragment_mf_energy(), -66.82653212162008, 8)

        frag = qemb.add_orbital_fragment(0)
        self.assertAlmostEqual(frag.get_fragment_mf_energy(), -57.71353451461683, 8)

        frag = qemb.add_orbital_fragment('1s')
        self.assertAlmostEqual(frag.get_fragment_mf_energy(), -170.7807235289751, 8)

    def test_iaopao_atoms(self):
        qemb = QEmbeddingMethod(self.mf)
        qemb.iaopao_fragmentation()
        frags = [qemb.add_atomic_fragment(['O%d'%x]) for x in range(1, 4)]
        e_mf = sum([f.get_fragment_mf_energy() for f in frags])
        self.assertAlmostEqual(e_mf, (self.mf.e_tot-self.mf.energy_nuc()), 8)

    def test_ghost_atoms(self):
        mol = self.mol.copy()
        mol.atom = 'ghost 0 0 0; %s' % self.mol.atom
        mol.basis = {'O1': 'cc-pvdz', 'O2': 'cc-pvdz', 'O3': 'cc-pvdz', 'ghost': pyscf.gto.basis.load('sto3g', 'H')}
        mol.build()

        mf = pyscf.scf.RHF(mol)
        mf = mf.density_fit()
        mf.conv_tol = 1e-12
        mf.kernel()

        qemb = QEmbeddingMethod(mf)
        qemb.iao_fragmentation()

        frag = qemb.add_orbital_fragment([0, 1])
        self.assertAlmostEqual(frag.get_fragment_mf_energy(), -72.98871119377974, 8)

    def test_project_amplitude_to_fragment(self):
        qemb = QEmbeddingMethod(self.mf)
        qemb.sao_fragmentation()
        frag = qemb.add_atomic_fragment(0)

        nocc = np.sum(self.mf.mo_occ > 0)
        nvir = np.sum(self.mf.mo_occ == 0)
        nmo = nocc + nvir

        with temporary_seed(1):
            c_ij = np.random.random((nocc, nvir))
            c_ijab = np.random.random((nocc, nocc, nvir, nvir))
            c_occ = np.random.random((nmo, nocc))
            c_vir = np.random.random((nmo, nvir))


        f = frag.project_amplitude_to_fragment(c_ij, c_occ=c_occ, c_vir=c_vir, partition='first-occ')
        self.assertAlmostEqual(lib.fp(f), 147.69445873300094, 8)

        f = frag.project_amplitude_to_fragment(c_ij, c_occ=c_occ, c_vir=c_vir, partition='first-vir')
        self.assertAlmostEqual(lib.fp(f), 151.60598331376818, 8)

        f = frag.project_amplitude_to_fragment(c_ij, c_occ=c_occ, c_vir=c_vir, partition='occ-2')
        self.assertAlmostEqual(lib.fp(f), 147.69445873300094, 8)

        f = frag.project_amplitude_to_fragment(c_ij, c_occ=c_occ, c_vir=c_vir, partition='democratic')
        self.assertAlmostEqual(lib.fp(f), 149.65022102336877, 8)


        f = frag.project_amplitude_to_fragment(c_ijab, c_occ=c_occ, c_vir=c_vir, partition='first-occ')
        self.assertAlmostEqual(lib.fp(f), -253.9226566096766, 8)

        f = frag.project_amplitude_to_fragment(c_ijab, c_occ=c_occ, c_vir=c_vir, partition='first-vir')
        self.assertAlmostEqual(lib.fp(f), 62.534014837251874, 8)

        f = frag.project_amplitude_to_fragment(c_ijab, c_occ=c_occ, c_vir=c_vir, partition='occ-2')
        self.assertAlmostEqual(lib.fp(f), -764.3221494218335, 8)

        f = frag.project_amplitude_to_fragment(c_ijab, c_occ=c_occ, c_vir=c_vir, partition='democratic')
        self.assertAlmostEqual(lib.fp(f), 674831740.969954, 8)

        f = frag.project_amplitude_to_fragment(c_ijab, c_occ=c_occ, c_vir=c_vir, partition='democratic', symmetrize=False)
        self.assertAlmostEqual(lib.fp(f), 674820355.3385825, 8)


class CellFragmentTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        L = 5.0
        n = 11
        cls.cell = pyscf.pbc.gto.Cell()
        cls.cell.a = np.eye(3) * L
        cls.cell.mesh = [n, n, n]
        cls.cell.atom = 'He 3 2 3; He 1 1 1'
        cls.cell.basis = '6-31g'
        cls.cell.exp_to_discard = 0.1
        cls.cell.verbose = 0
        cls.cell.max_memory = 1e9
        cls.cell.build()
        cls.scell = pyscf.pbc.tools.super_cell(cls.cell, [2,2,2])
        cls.kpts = cls.cell.make_kpts([2,2,2])
        cls.mf = pyscf.pbc.scf.KRHF(cls.cell, cls.kpts)
        cls.mf.conv_tol = 1e-12
        cls.mf.with_df = gdf.GDF(cls.cell, cls.kpts)
        cls.mf.with_df.linear_dep_threshold = 1e-7
        cls.mf.with_df.build()
        cls.mf.kernel()

    @classmethod
    def tearDownClass(cls):
        del cls.cell, cls.kpts, cls.mf

    def test_iao_atoms(self):
        qemb = QEmbeddingMethod(self.mf)
        qemb.iao_fragmentation()
        frag = qemb.add_atomic_fragment([0, 1])
        frags = [frag,] + frag.add_tsymmetric_fragments([2,2,2])

        for frag in frags[0].loop_fragments():
            self.assertAlmostEqual(frag.get_fragment_mf_energy().real, -4.261995344528774, 8)

        tr = lambda c: np.einsum('xi,xi->', c, c.conj())

        c_bath, c_occenv, c_virenv = DMET_Bath(frags[0], frags[0].opts.dmet_threshold).make_dmet_bath(frags[0].c_env)
        self.assertAlmostEqual(tr(c_bath), 0.0, 8)
        self.assertAlmostEqual(tr(c_occenv), 8.600258699380264, 8)
        self.assertAlmostEqual(tr(c_virenv), 43.70235664564082, 8)

    def test_iao_aos(self):
        qemb = QEmbeddingMethod(self.mf)
        qemb.iao_fragmentation()
        frag = qemb.add_orbital_fragment([0, 1])

        self.assertAlmostEqual(frag.get_fragment_mf_energy().real, -4.261995344528774, 8)

    def test_lowdin_atoms(self):
        qemb = QEmbeddingMethod(self.mf)
        qemb.sao_fragmentation()
        frags = [qemb.add_atomic_fragment([i*2, i*2+1]) for i in range(len(self.kpts))]

        for frag in frags:
            self.assertAlmostEqual(frag.get_fragment_mf_energy().real, -4.261995344528774, 8)

        tr = lambda c: np.einsum('xi,xi->', c, c.conj())

        c_bath, c_occenv, c_virenv = DMET_Bath(frags[0], 1e-5).make_dmet_bath(frags[0].c_env)
        self.assertAlmostEqual(tr(c_bath), 0.0, 8)
        self.assertAlmostEqual(tr(c_occenv), 8.60025893931295, 8)
        self.assertAlmostEqual(tr(c_virenv), 38.23956182500303, 8)

    def test_lowdin_aos(self):
        qemb = QEmbeddingMethod(self.mf)
        qemb.sao_fragmentation()
        frag = qemb.add_orbital_fragment([0, 1, 2, 3])

        self.assertAlmostEqual(frag.get_fragment_mf_energy().real, -4.261995344528774, 8)




if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
