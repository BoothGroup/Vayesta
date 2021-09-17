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

#TODO: fragment_type = 'ao' after bugs are fixed
#TODO: convert_amp_c_to_t
#TODO: get_t1
#TODO: get_t2
#TODO: cover properties
#TODO: get_rot_to_mf
#TODO: get_rot_to_fragment
#TODO: couple_to_fragment(s)
#TODO: inverse get_fragment_project
#TODO: project_ref_orbitals
#TODO: lots of conditions in make_dmet_bath
#TODO: c_ref in dmet bath?
#TODO: 'occ-2' in project_amplitude_to_fragment
#TODO: make_counterpoise_mol


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

    @classmethod
    def tearDownClass(cls):
        del cls.mol, cls.mf

    def test_iao_atoms(self):
        qemb = QEmbeddingMethod(self.mf)
        qemb.init_fragmentation('iao')
        frags = qemb.make_all_atom_fragments()

        self.assertAlmostEqual(frags[0].get_fragment_mf_energy(), -107.09600995056937, 8)
        self.assertAlmostEqual(frags[1].get_fragment_mf_energy(), -104.94355748773579, 8)
        self.assertAlmostEqual(frags[2].get_fragment_mf_energy(), -104.94355748773579, 8)

        tr = lambda c: np.einsum('xi,xi->', c, c.conj())

        c_bath, c_occenv, c_virenv = frags[0].make_dmet_bath(frags[0].c_env)
        self.assertAlmostEqual(tr(c_bath), 5.305607912416594, 8)
        self.assertAlmostEqual(tr(c_occenv), 6.0044522544970995, 8)
        self.assertAlmostEqual(tr(c_virenv), 349.9785855174963, 8)

        frags[1].opts.wf_partition = 'first-vir'
        c_bath, c_occenv, c_virenv = frags[1].make_dmet_bath(frags[1].c_env)
        self.assertAlmostEqual(tr(c_bath), 7.357949737669514, 8)
        self.assertAlmostEqual(tr(c_occenv), 5.4998286299420664, 8)
        self.assertAlmostEqual(tr(c_virenv), 349.9785855174961, 8)

        frags[2].opts.wf_partition = 'democratic'
        c_bath, c_occenv, c_virenv = frags[2].make_dmet_bath(frags[2].c_env)
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
        qemb.init_fragmentation('iao')

        frag = qemb.make_ao_fragment([0, 1])
        self.assertAlmostEqual(frag.get_fragment_mf_energy(), -72.99138042535633, 8)

        frag = qemb.make_atom_fragment([0], aos=['1s', '2s'])
        self.assertAlmostEqual(frag.get_fragment_mf_energy(), -72.99138042535633, 8)

    def test_lowdin_atoms(self):
        qemb = QEmbeddingMethod(self.mf)
        qemb.init_fragmentation('lowdin-ao')
        frags = [qemb.make_atom_fragment(['O%d'%x]) for x in range(1, 4)]

        self.assertAlmostEqual(frags[0].get_fragment_mf_energy(), -108.51371286149299, 8)
        self.assertAlmostEqual(frags[1].get_fragment_mf_energy(), -104.23470603227311, 8)
        self.assertAlmostEqual(frags[2].get_fragment_mf_energy(), -104.23470603227311, 8)

    def test_lowdin_aos(self):
        qemb = QEmbeddingMethod(self.mf)
        qemb.init_fragmentation('lowdin-ao')

        frag = qemb.make_ao_fragment([0, 1])
        self.assertAlmostEqual(frag.get_fragment_mf_energy(), -66.82653212162008, 8)

        frag = qemb.make_ao_fragment(0)
        self.assertAlmostEqual(frag.get_fragment_mf_energy(), -57.71353451461683, 8)

        frag = qemb.make_ao_fragment('1s')
        self.assertAlmostEqual(frag.get_fragment_mf_energy(), -170.7807235289751, 8)

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
        qemb.init_fragmentation('iao')

        frag = qemb.make_ao_fragment([0, 1])
        self.assertAlmostEqual(frag.get_fragment_mf_energy(), -72.98871119377974, 8)


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
        qemb.init_fragmentation('iao')
        frag = qemb.make_atom_fragment([0, 1])
        frags = [frag,] + frag.make_tsymmetric_fragments([2,2,2])

        for frag in frags[0].loop_fragments():
            self.assertAlmostEqual(frag.get_fragment_mf_energy().real, -4.261995344528774, 8)

        tr = lambda c: np.einsum('xi,xi->', c, c.conj())

        c_bath, c_occenv, c_virenv = frags[0].make_dmet_bath(frags[0].c_env)
        self.assertAlmostEqual(tr(c_bath), 0.0, 8)
        self.assertAlmostEqual(tr(c_occenv), 8.600258699380264, 8)
        self.assertAlmostEqual(tr(c_virenv), 43.70235664564082, 8)

    def test_iao_aos(self):
        qemb = QEmbeddingMethod(self.mf)
        qemb.init_fragmentation('iao')
        frag = qemb.make_ao_fragment([0, 1])

        self.assertAlmostEqual(frag.get_fragment_mf_energy().real, -4.261995344528774, 8)

    def test_lowdin_atoms(self):
        qemb = QEmbeddingMethod(self.mf)
        qemb.init_fragmentation('lowdin-ao')
        frags = [qemb.make_atom_fragment([i*2, i*2+1]) for i in range(len(self.kpts))]

        for frag in frags:
            self.assertAlmostEqual(frag.get_fragment_mf_energy().real, -4.261995344528774, 8)

        tr = lambda c: np.einsum('xi,xi->', c, c.conj())

        c_bath, c_occenv, c_virenv = frags[0].make_dmet_bath(frags[0].c_env)
        self.assertAlmostEqual(tr(c_bath), 0.0, 8)
        self.assertAlmostEqual(tr(c_occenv), 8.60025893931295, 8)
        self.assertAlmostEqual(tr(c_virenv), 38.23956182500303, 8)

    def test_lowdin_aos(self):
        qemb = QEmbeddingMethod(self.mf)
        qemb.init_fragmentation('lowdin-ao')
        frag = qemb.make_ao_fragment([0, 1, 2, 3])

        self.assertAlmostEqual(frag.get_fragment_mf_energy().real, -4.261995344528774, 8)




if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
