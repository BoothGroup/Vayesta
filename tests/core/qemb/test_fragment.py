import unittest
import numpy as np
import pyscf.gto
import pyscf.scf
import pyscf.pbc.gto
import pyscf.pbc.scf
import pyscf.pbc.tools
import pyscf.mp
import pyscf.pbc.mp
from pyscf import lib
from vayesta.misc import gdf
from vayesta.core import QEmbeddingFragment, QEmbeddingMethod
from vayesta.core.bath import DMET_Bath, MP2_BNO_Bath
from vayesta.core import UEmbedding, UFragment
from vayesta.core.bath import UDMET_Bath
from vayesta.core.actspace import ActiveSpace


class temporary_seed:
    def __init__(self, seed):
        self.seed, self.state = seed, None
    def __enter__(self):
        self.state = np.random.get_state()
        np.random.seed(self.seed)
    def __exit__(self, *args):
        np.random.set_state(self.state)


class MolFragmentTests(unittest.TestCase):

    PLACES = 8
    Embedding = QEmbeddingMethod
    HF = pyscf.scf.hf.RHF
    DMET_Bath = DMET_Bath

    @classmethod
    def setUpClass(cls):
        cls.mol = pyscf.gto.Mole()
        cls.mol.atom = 'O1 0 0 0.118; O2 0 0.755 -0.471; O3 0 -0.755 -0.471'
        cls.mol.basis = 'cc-pvdz'
        cls.mol.verbose = 0
        cls.mol.build()
        cls.mf = cls.HF(cls.mol)
        cls.mf = cls.mf.density_fit()
        cls.mf.conv_tol = 1e-12
        cls.mf.kernel()
        assert cls.mf.converged

    @classmethod
    def tearDownClass(cls):
        del cls.mol, cls.mf

    def trace(self, c):
        return np.einsum('xi,xi->', c, c.conj())

    def test_properties(self):
        qemb = self.Embedding(self.mf)
        qemb.iao_fragmentation()
        frag = qemb.add_atomic_fragment(0)

        self.assertIs(frag.mol,           self.mol)
        self.assertIs(frag.mf,            frag.base.mf)
        self.assertIs(frag.n_frag,        frag.c_frag.shape[-1])
        self.assertIs(frag.boundary_cond, frag.base.boundary_cond)

        self.assertIsNone(frag.c_active)
        self.assertIsNone(frag.c_active_occ)
        self.assertIsNone(frag.c_active_vir)
        self.assertIsNone(frag.c_frozen)
        self.assertIsNone(frag.c_frozen_occ)
        self.assertIsNone(frag.c_frozen_vir)

    def test_iao_atoms(self):
        qemb = self.Embedding(self.mf)
        qemb.iao_fragmentation()
        frags = qemb.add_all_atomic_fragments()

        e_elec = sum([f.get_fragment_mf_energy() for f in frags])
        self.assertAlmostEqual(e_elec + self.mf.energy_nuc(), self.mf.e_tot, self.PLACES)
        self.assertAlmostEqual(frags[0].get_fragment_mf_energy(), -107.09600995056937, self.PLACES)
        self.assertAlmostEqual(frags[1].get_fragment_mf_energy(), -104.94355748773579, self.PLACES)
        self.assertAlmostEqual(frags[2].get_fragment_mf_energy(), -104.94355748773579, self.PLACES)

    def test_iao_aos(self):
        qemb = self.Embedding(self.mf)
        qemb.iao_fragmentation()

        frag = qemb.add_orbital_fragment([0, 1])
        self.assertAlmostEqual(frag.get_fragment_mf_energy(), -72.99138042535633, self.PLACES)

        frag = qemb.add_atomic_fragment([0], orbital_filter=['1s', '2s'])
        self.assertAlmostEqual(frag.get_fragment_mf_energy(), -72.99138042535633, self.PLACES)

    def test_iao_aos_uhf(self):
        qemb = self.Embedding(self.mf)
        qemb.iao_fragmentation()

        frag = qemb.add_orbital_fragment([0, 1])
        self.assertAlmostEqual(frag.get_fragment_mf_energy(), -72.99138042535633, self.PLACES)

        frag = qemb.add_atomic_fragment([0], orbital_filter=['1s', '2s'])
        self.assertAlmostEqual(frag.get_fragment_mf_energy(), -72.99138042535633, self.PLACES)

    def test_iao_minao(self):
        qemb = self.Embedding(self.mf)
        qemb.iao_fragmentation(minao='sto3g')
        frag = qemb.add_atomic_fragment(0)

        self.assertAlmostEqual(frag.get_fragment_mf_energy(), -107.04952448358654, self.PLACES)

    def test_sao_atoms(self):
        qemb = self.Embedding(self.mf)
        qemb.sao_fragmentation()
        frags = [qemb.add_atomic_fragment(['O%d'%x]) for x in range(1, 4)]

        self.assertAlmostEqual(frags[0].get_fragment_mf_energy(), -108.51371286149299, self.PLACES)
        self.assertAlmostEqual(frags[1].get_fragment_mf_energy(), -104.23470603227311, self.PLACES)
        self.assertAlmostEqual(frags[2].get_fragment_mf_energy(), -104.23470603227311, self.PLACES)
        e_mf = sum([f.get_fragment_mf_energy() for f in frags])
        self.assertAlmostEqual(e_mf, (self.mf.e_tot-self.mf.energy_nuc()), self.PLACES)

    def test_sao_aos(self):
        qemb = self.Embedding(self.mf)
        qemb.sao_fragmentation()

        frag = qemb.add_orbital_fragment([0, 1])
        self.assertAlmostEqual(frag.get_fragment_mf_energy(), -66.82653212162008, self.PLACES)

        frag = qemb.add_orbital_fragment(0)
        self.assertAlmostEqual(frag.get_fragment_mf_energy(), -57.71353451461683, self.PLACES)

        frag = qemb.add_orbital_fragment('1s')
        self.assertAlmostEqual(frag.get_fragment_mf_energy(), -170.7807235289751, self.PLACES)

    def test_iaopao_atoms(self):
        qemb = self.Embedding(self.mf)
        qemb.iaopao_fragmentation()
        frags = [qemb.add_atomic_fragment(['O%d'%x]) for x in range(1, 4)]
        e_mf = sum([f.get_fragment_mf_energy() for f in frags])
        self.assertAlmostEqual(e_mf, (self.mf.e_tot-self.mf.energy_nuc()), self.PLACES)

    def test_ghost_atoms(self):
        mol = self.mol.copy()
        mol.atom = 'ghost 0 0 0; %s' % self.mol.atom
        mol.basis = {'O1': 'cc-pvdz', 'O2': 'cc-pvdz', 'O3': 'cc-pvdz', 'ghost': pyscf.gto.basis.load('sto3g', 'H')}
        mol.build()

        mf = self.HF(mol)
        mf = mf.density_fit()
        mf.conv_tol = 1e-12
        mf.kernel()

        qemb = self.Embedding(mf)
        qemb.iao_fragmentation()

        frag = qemb.add_orbital_fragment([0, 1])
        self.assertAlmostEqual(frag.get_fragment_mf_energy(), -72.98871119377974, self.PLACES)

    def test_project_amplitude_to_fragment(self):
        qemb = self.Embedding(self.mf)
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
        self.assertAlmostEqual(lib.fp(f), 147.69445873300094, self.PLACES)

        f = frag.project_amplitude_to_fragment(c_ij, c_occ=c_occ, c_vir=c_vir, partition='first-vir')
        self.assertAlmostEqual(lib.fp(f), 151.60598331376818, self.PLACES)

        f = frag.project_amplitude_to_fragment(c_ij, c_occ=c_occ, c_vir=c_vir, partition='occ-2')
        self.assertAlmostEqual(lib.fp(f), 147.69445873300094, self.PLACES)

        f = frag.project_amplitude_to_fragment(c_ij, c_occ=c_occ, c_vir=c_vir, partition='democratic')
        self.assertAlmostEqual(lib.fp(f), 149.65022102336877, self.PLACES)


        f = frag.project_amplitude_to_fragment(c_ijab, c_occ=c_occ, c_vir=c_vir, partition='first-occ')
        self.assertAlmostEqual(lib.fp(f), -253.9226566096766, self.PLACES)

        f = frag.project_amplitude_to_fragment(c_ijab, c_occ=c_occ, c_vir=c_vir, partition='first-vir')
        self.assertAlmostEqual(lib.fp(f), 62.534014837251874, self.PLACES)

        f = frag.project_amplitude_to_fragment(c_ijab, c_occ=c_occ, c_vir=c_vir, partition='occ-2')
        self.assertAlmostEqual(lib.fp(f), -764.3221494218335, self.PLACES)

        f = frag.project_amplitude_to_fragment(c_ijab, c_occ=c_occ, c_vir=c_vir, partition='democratic')
        self.assertAlmostEqual(lib.fp(f), 674831740.969954, self.PLACES-5)

        f = frag.project_amplitude_to_fragment(c_ijab, c_occ=c_occ, c_vir=c_vir, partition='democratic', symmetrize=False)
        self.assertAlmostEqual(lib.fp(f), 674820355.3385825, self.PLACES-5)

    def test_project_ref_orbitals(self):
        qemb = self.Embedding(self.mf)
        qemb.sao_fragmentation()
        frag = qemb.add_atomic_fragment(0)

        nmo = self.mf.mo_occ.size

        with temporary_seed(1):
            c_ref = np.random.random((nmo, nmo))
            c = np.random.random((nmo, nmo))

        w, v = frag.project_ref_orbitals(c_ref, c)

        self.assertAlmostEqual(lib.fp(w),   8.1870185388302, self.PLACES)
        self.assertAlmostEqual(lib.fp(v), 694.4976696239447, self.PLACES)

    def test_dmet_bath(self):
        qemb = self.Embedding(self.mf)
        qemb.iao_fragmentation()
        frags = qemb.add_all_atomic_fragments()

        bath = self.DMET_Bath(frags[0], frags[0].opts.dmet_threshold)
        bath.kernel()
        self.assertAlmostEqual(self.trace(bath.c_dmet),      5.3056079124166, self.PLACES)
        self.assertAlmostEqual(self.trace(bath.c_env_occ),   6.0044522544971, self.PLACES)
        self.assertAlmostEqual(self.trace(bath.c_env_vir), 349.9785855174963, self.PLACES)

        frags[1].opts.wf_partition = 'first-vir'
        bath = self.DMET_Bath(frags[1], frags[1].opts.dmet_threshold)
        bath.kernel()
        self.assertAlmostEqual(self.trace(bath.c_dmet),      7.3579497376695, self.PLACES)
        self.assertAlmostEqual(self.trace(bath.c_env_occ),   5.4998286299421, self.PLACES)
        self.assertAlmostEqual(self.trace(bath.c_env_vir), 349.9785855174961, self.PLACES)

        frags[2].opts.wf_partition = 'democratic'
        bath = self.DMET_Bath(frags[2], frags[2].opts.dmet_threshold)
        bath.kernel()
        self.assertAlmostEqual(self.trace(bath.c_dmet),      7.3579497376695, self.PLACES)
        self.assertAlmostEqual(self.trace(bath.c_env_occ),   5.4998286299421, self.PLACES)
        self.assertAlmostEqual(self.trace(bath.c_env_vir), 349.9785855174961, self.PLACES)

        frag = frags[2]
        self.assertIs(bath.mf,     frag.mf)
        self.assertIs(bath.mol,    frag.mol)
        self.assertIs(bath.log,    frag.log)
        self.assertIs(bath.base,   frag.base)
        self.assertIs(bath.c_frag, frag.c_frag)
        self.assertIs(bath.get_dmet_bath(), bath.c_dmet)
        self.assertIs(bath.get_environment()[0], bath.c_env_occ)
        self.assertIs(bath.get_environment()[1], bath.c_env_vir)

        frag.reset()
        self.assertIsNone(frag._c_active_occ)
        self.assertIsNone(frag._c_active_vir)
        self.assertIsNone(frag._c_frozen_occ)
        self.assertIsNone(frag._c_frozen_vir)

    def test_mp2_bno_bath(self):
        qemb = self.Embedding(self.mf)
        qemb.iao_fragmentation()
        frags = qemb.add_all_atomic_fragments()

        nocc = np.sum(self.mf.mo_occ > 0)
        nvir = np.sum(self.mf.mo_occ == 0)
        with temporary_seed(1):
            t2a = np.random.random((nocc, nocc, nvir, nvir)) - 0.5
            t2a = 0.25 * (t2a + t2a.swapaxes(0,1) + t2a.swapaxes(2,3) + t2a.transpose(1,0,3,2))
            t2b = np.random.random((nocc, nocc, nvir, nvir)) - 0.5
            t2b = 0.25 * (t2b + t2b.swapaxes(0,1) + t2b.swapaxes(2,3) + t2b.transpose(1,0,3,2))

        bath = MP2_BNO_Bath(frags[0])
        dm1o = bath.make_dm1('occ', t2a, t2b)
        dm1v = bath.make_dm1('vir', t2a, t2b)
        self.assertIs(bath.get_mp2_class(), pyscf.mp.MP2)
        self.assertAlmostEqual(lib.fp(dm1o), -821.3024549573227, self.PLACES)
        self.assertAlmostEqual(lib.fp(dm1v),   82.9358131759875, self.PLACES)
        self.assertIsNone(bath.c_bno_occ)
        self.assertIsNone(bath.c_bno_vir)
        self.assertIsNone(bath.n_bno_occ)
        self.assertIsNone(bath.n_bno_vir)
        self.assertIs(bath.mf,     frags[0].mf)
        self.assertIs(bath.mol,    frags[0].mol)
        self.assertIs(bath.log,    frags[0].log)
        self.assertIs(bath.base,   frags[0].base)
        self.assertIs(bath.c_frag, frags[0].c_frag)

        bath = MP2_BNO_Bath(frags[0], local_dm='semi', canonicalize=False)
        dm1o = bath.make_dm1('occ', t2a, t2b)
        dm1v = bath.make_dm1('vir', t2a, t2b)
        self.assertIs(bath.get_mp2_class(), pyscf.mp.MP2)
        self.assertAlmostEqual(lib.fp(dm1o),  -38.8930725735660, self.PLACES)
        self.assertAlmostEqual(lib.fp(dm1v),  -19.2835444257561, self.PLACES)

        bath = MP2_BNO_Bath(frags[0], local_dm=True, canonicalize=False)
        dm1o = bath.make_dm1('occ', t2a, t2b)
        dm1v = bath.make_dm1('vir', t2a, t2b)
        self.assertIs(bath.get_mp2_class(), pyscf.mp.MP2)
        self.assertAlmostEqual(lib.fp(dm1o), -936.2602579531159, self.PLACES)
        self.assertAlmostEqual(lib.fp(dm1v),   20.0117780184031, self.PLACES)

        #FIXME: bugs #6 and #7 - then add to CellFragmentTests

        #tr = lambda c: np.einsum('xi,xi->', c, c.conj())

        #bath = MP2_BNO_Bath(frags[1])
        #bath.kernel()
        #c_bno_occ, n_bno_occ = bath.make_bno_coeff('occ')
        #c_bno_vir, n_bno_vir = bath.make_bno_coeff('vir')
        #self.assertAlmostEqual(tr(c_bno_occ),     0.0, self.PLACES)
        #self.assertAlmostequal(lib.fp(n_bno_occ), 0.0, self.PLACES)

        #bath = MP2_BNO_Bath(frags[1], canonicalize=(False, False))
        #bath.kernel()
        #mp2 = pyscf.mp.mp2.MP2(self.mf)
        #eris = mp2.ao2mo()
        #c_bno_occ, n_bno_occ = bath.make_bno_coeff('occ', eris=eris)
        #c_bno_vir, n_bno_vir = bath.make_bno_coeff('vir', eris=eris)


class UMolFragmentTests(MolFragmentTests):

    PLACES = 6
    Embedding = UEmbedding
    HF = pyscf.scf.uhf.UHF
    DMET_Bath = UDMET_Bath

    def trace(self, c):
        ca, cb = c
        return (np.einsum('xi,xi->', ca, ca.conj()) + np.einsum('xi,xi->', cb, cb.conj())) / 2

    def test_properties(self):
        pass

    test_project_amplitude_to_fragment = None
    test_project_ref_orbitals = None
    test_mp2_bno_bath = None


class CellFragmentTests(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        L = 5.0
        n = 11
        cls.cell = pyscf.pbc.gto.Cell()
        cls.cell.a = np.eye(3) * L
        cls.cell.mesh = [n, n, n]
        cls.cell.atom = 'He 3 2 3; He 1 1 1'
        cls.cell.basis = 'cc-pvdz'
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
            self.assertAlmostEqual(frag.get_fragment_mf_energy().real, -4.261997589296356, 8)

    def test_iao_aos(self):
        qemb = QEmbeddingMethod(self.mf)
        qemb.iao_fragmentation()
        frag = qemb.add_orbital_fragment([0, 1])

        self.assertAlmostEqual(frag.get_fragment_mf_energy().real, -4.261997589296356, 8)

    def test_sao_atoms(self):
        qemb = QEmbeddingMethod(self.mf)
        qemb.sao_fragmentation()
        frags = [qemb.add_atomic_fragment([i*2, i*2+1]) for i in range(len(self.kpts))]

        for frag in frags:
            self.assertAlmostEqual(frag.get_fragment_mf_energy().real, -4.261997589296352, 8)

    def test_sao_aos(self):
        qemb = QEmbeddingMethod(self.mf)
        qemb.sao_fragmentation()
        frag = qemb.add_orbital_fragment([0, 1, 2, 3])

        self.assertAlmostEqual(frag.get_fragment_mf_energy().real, -2.130998605308573, 8)

    def test_dmet_bath(self):
        tr = lambda c: np.einsum('xi,xi->', c, c.conj())

        qemb = QEmbeddingMethod(self.mf)
        qemb.sao_fragmentation()
        frag = qemb.add_atomic_fragment([0])
        frags = [frag,] + frag.add_tsymmetric_fragments([2,2,2])

        bath = DMET_Bath(frags[1], frags[1].opts.dmet_threshold)
        c_bath, c_occenv, c_virenv = bath.make_dmet_bath(frags[0].c_env)
        self.assertAlmostEqual(tr(c_bath),    3.30703579288843, 8)
        self.assertAlmostEqual(tr(c_occenv),  8.60108764820888, 8)
        self.assertAlmostEqual(tr(c_virenv), 83.27964350816293, 8)

        #bath = DMET_Bath(frags[0], frags[0].opts.dmet_threshold)
        #c_bath, c_occenv, c_virenv = bath.make_dmet_bath(frags[0].c_env, nbath=c_bath.shape[-1])  #FIXME bug #5
        #self.assertAlmostEqual(tr(c_bath),    3.30703579288843, 8)
        #self.assertAlmostEqual(tr(c_occenv),  8.60108764820888, 8)
        #self.assertAlmostEqual(tr(c_virenv), 83.27964350816293, 8)

        qemb = QEmbeddingMethod(self.mf)
        qemb.sao_fragmentation()
        frags = [qemb.add_atomic_fragment([i*2, i*2+1]) for i in range(len(self.kpts))]

        bath = DMET_Bath(frags[0], 1e-5)
        c_bath, c_occenv, c_virenv = bath.make_dmet_bath(frags[0].c_env, verbose=False)
        self.assertAlmostEqual(tr(c_bath),    0.00000000000000, 8)
        self.assertAlmostEqual(tr(c_occenv),  8.60108602101449, 8)
        self.assertAlmostEqual(tr(c_virenv), 80.24082979829498, 8)


if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
