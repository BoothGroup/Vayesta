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

from vayesta.core import Embedding, UEmbedding
from vayesta.core.bath import DMET_Bath, MP2_BNO_Bath, UDMET_Bath
from vayesta.tests.cache import moles, cells
from vayesta.tests.common import temporary_seed

#TODO readd some tests for ghost atoms


class MolFragmentTests(unittest.TestCase):
    key = 'h2o_ccpvdz_df'
    mf_key = 'rhf'
    PLACES = 8
    Embedding = Embedding
    DMET_Bath = DMET_Bath

    def trace(self, c):
        return np.einsum('xi,xi->', c, c.conj())

    def test_properties(self):
        """Test methods registered with @property decorator.
        """

        qemb = self.Embedding(moles[self.key][self.mf_key])
        qemb.iao_fragmentation()
        frag = qemb.add_atomic_fragment(0)

        self.assertIs(frag.mol,           moles[self.key]['mol'])
        self.assertIs(frag.mf,            frag.base.mf)
        self.assertIs(frag.n_frag,        frag.c_frag.shape[-1])
        self.assertIs(frag.boundary_cond, frag.base.boundary_cond)

    def test_iao_atoms(self):
        """Test IAO atomic fragmentation.
        """

        qemb = self.Embedding(moles[self.key][self.mf_key])
        qemb.iao_fragmentation()
        frags = qemb.add_all_atomic_fragments()

        e_elec = sum([f.get_fragment_mf_energy() for f in frags])
        self.assertAlmostEqual(e_elec + moles[self.key][self.mf_key].energy_nuc(), moles[self.key][self.mf_key].e_tot, self.PLACES)
        self.assertAlmostEqual(frags[0].get_fragment_mf_energy(), -81.55618063172534, self.PLACES)
        self.assertAlmostEqual(frags[1].get_fragment_mf_energy(),  -1.83005213220285, self.PLACES)
        self.assertAlmostEqual(frags[2].get_fragment_mf_energy(),  -1.83005213220285, self.PLACES)

    def test_iao_aos(self):
        """Test IAO orbital fragmentation.
        """

        qemb = self.Embedding(moles[self.key][self.mf_key])
        qemb.iao_fragmentation()

        frag = qemb.add_orbital_fragment([0, 1])
        self.assertAlmostEqual(frag.get_fragment_mf_energy(), -61.40491055875449, self.PLACES)

        frag = qemb.add_atomic_fragment([0], orbital_filter=['1s', '2s'])
        self.assertAlmostEqual(frag.get_fragment_mf_energy(), -61.40491055875449, self.PLACES)

    def test_iao_minao(self):
        """Test IAO fragmentation with a custom `minao` keyword.
        """

        qemb = self.Embedding(moles[self.key][self.mf_key])
        qemb.iao_fragmentation(minao='sto3g')
        frag = qemb.add_atomic_fragment(0)

        self.assertAlmostEqual(frag.get_fragment_mf_energy(), -80.8244967526962, self.PLACES)

    def test_sao_atoms(self):
        """Test SAO atomic fragmentation.
        """

        qemb = self.Embedding(moles[self.key][self.mf_key])
        qemb.sao_fragmentation()
        frags  = [qemb.add_atomic_fragment(['O'])]
        frags += [qemb.add_atomic_fragment(i) for i in [1, 2]]

        self.assertAlmostEqual(frags[0].get_fragment_mf_energy(), -78.51384197417300, self.PLACES)
        self.assertAlmostEqual(frags[1].get_fragment_mf_energy(),  -3.35122146047156, self.PLACES)
        self.assertAlmostEqual(frags[2].get_fragment_mf_energy(),  -3.35122146047156, self.PLACES)
        e_mf = sum([f.get_fragment_mf_energy() for f in frags])
        self.assertAlmostEqual(e_mf, (moles[self.key][self.mf_key].e_tot-moles[self.key][self.mf_key].energy_nuc()), self.PLACES)

    def test_sao_aos(self):
        """Test SAO orbital fragmentation.
        """

        qemb = self.Embedding(moles[self.key][self.mf_key])
        qemb.sao_fragmentation()

        frag = qemb.add_orbital_fragment([0, 1])
        self.assertAlmostEqual(frag.get_fragment_mf_energy(), -56.62602304567066, self.PLACES)

        frag = qemb.add_orbital_fragment(0)
        self.assertAlmostEqual(frag.get_fragment_mf_energy(), -50.92864009746159, self.PLACES)

        frag = qemb.add_orbital_fragment('1s')
        self.assertAlmostEqual(frag.get_fragment_mf_energy(), -54.5697235132222, self.PLACES)

    def test_iaopao_atoms(self):
        """Test IAO+PAO atomic fragmentation.
        """

        qemb = self.Embedding(moles[self.key][self.mf_key])
        qemb.iaopao_fragmentation()

        frags = qemb.add_all_atomic_fragments()
        e_mf = sum([f.get_fragment_mf_energy() for f in frags])
        self.assertAlmostEqual(e_mf, (moles[self.key][self.mf_key].e_tot-moles[self.key][self.mf_key].energy_nuc()), self.PLACES)

    def test_project_amplitude_to_fragment(self):
        """Test the project_amplitude_to_fragment function.
        """

        qemb = self.Embedding(moles[self.key][self.mf_key])
        qemb.sao_fragmentation()
        frag = qemb.add_atomic_fragment(0)

        nocc = np.sum(moles[self.key][self.mf_key].mo_occ > 0)
        nvir = np.sum(moles[self.key][self.mf_key].mo_occ == 0)
        nmo = nocc + nvir

        with temporary_seed(1):
            c_ij = np.random.random((nocc, nvir)) - 0.5
            c_ijab = np.random.random((nocc, nocc, nvir, nvir)) - 0.5
            c_occ = np.random.random((nmo, nocc)) - 0.5
            c_vir = np.random.random((nmo, nvir)) - 0.5

        f = frag.project_amplitude_to_fragment(c_ij, c_occ=c_occ, c_vir=c_vir, partition='first-occ')
        self.assertAlmostEqual(lib.fp(f), -1.22094291483245, self.PLACES)

        f = frag.project_amplitude_to_fragment(c_ij, c_occ=c_occ, c_vir=c_vir, partition='first-vir')
        self.assertAlmostEqual(lib.fp(f),  1.45786726001951, self.PLACES)

        f = frag.project_amplitude_to_fragment(c_ij, c_occ=c_occ, c_vir=c_vir, partition='occ-2')
        self.assertAlmostEqual(lib.fp(f), -1.22094291483245, self.PLACES)

        f = frag.project_amplitude_to_fragment(c_ij, c_occ=c_occ, c_vir=c_vir, partition='democratic')
        self.assertAlmostEqual(lib.fp(f),  0.11846217259353, self.PLACES)

        f = frag.project_amplitude_to_fragment(c_ijab, c_occ=c_occ, c_vir=c_vir, partition='first-occ')
        self.assertAlmostEqual(lib.fp(f),  1.06523031903778, self.PLACES)

        f = frag.project_amplitude_to_fragment(c_ijab, c_occ=c_occ, c_vir=c_vir, partition='first-vir')
        self.assertAlmostEqual(lib.fp(f), -4.58067349696292, self.PLACES)

        f = frag.project_amplitude_to_fragment(c_ijab, c_occ=c_occ, c_vir=c_vir, partition='occ-2')
        self.assertAlmostEqual(lib.fp(f), 19.99235570216633, self.PLACES)

        f = frag.project_amplitude_to_fragment(c_ijab, c_occ=c_occ, c_vir=c_vir, partition='democratic')
        self.assertAlmostEqual(lib.fp(f), 99.05649141218053, self.PLACES)

        f = frag.project_amplitude_to_fragment(c_ijab, c_occ=c_occ, c_vir=c_vir, partition='democratic', symmetrize=False)
        self.assertAlmostEqual(lib.fp(f), 86.58052426394231, self.PLACES)

    def test_project_ref_orbitals(self):
        """Test the project_ref_orbitals function.
        """

        qemb = self.Embedding(moles[self.key][self.mf_key])
        qemb.sao_fragmentation()
        frag = qemb.add_atomic_fragment(0)

        nmo = moles[self.key][self.mf_key].mo_occ.size

        with temporary_seed(1):
            c_ref = np.random.random((nmo, nmo))
            c = np.random.random((nmo, nmo))

        w, v = frag.project_ref_orbitals(c_ref, c)

        self.assertAlmostEqual(lib.fp(w),          5.63864201070, self.PLACES)
        self.assertAlmostEqual(np.sum(v*v)**0.5, 235.00085653529, self.PLACES)

    def test_dmet_bath(self):
        """Test the DMET bath.
        """

        qemb = self.Embedding(moles[self.key][self.mf_key])
        qemb.iao_fragmentation()
        frags = qemb.add_all_atomic_fragments()

        bath = self.DMET_Bath(frags[0], frags[0].opts.dmet_threshold)
        bath.kernel()
        self.assertAlmostEqual(self.trace(bath.c_dmet),     2.53936714739812, self.PLACES)
        self.assertAlmostEqual(self.trace(bath.c_env_occ),  0.00000000000000, self.PLACES)
        self.assertAlmostEqual(self.trace(bath.c_env_vir), 73.98633044345692, self.PLACES)

        frags[1].opts.wf_partition = 'first-vir'
        bath = self.DMET_Bath(frags[1], frags[1].opts.dmet_threshold)
        bath.kernel()
        self.assertAlmostEqual(self.trace(bath.c_dmet),     0.92508900878229, self.PLACES)
        self.assertAlmostEqual(self.trace(bath.c_env_occ),  2.79277815395295, self.PLACES)
        self.assertAlmostEqual(self.trace(bath.c_env_vir), 75.71109293760985, self.PLACES)

        frags[2].opts.wf_partition = 'democratic'
        bath = self.DMET_Bath(frags[2], frags[2].opts.dmet_threshold)
        bath.kernel()
        self.assertAlmostEqual(self.trace(bath.c_dmet),     0.92508900878229, self.PLACES)
        self.assertAlmostEqual(self.trace(bath.c_env_occ),  2.79277815395295, self.PLACES)
        self.assertAlmostEqual(self.trace(bath.c_env_vir), 75.71109293760989, self.PLACES)

        frag = frags[2]
        self.assertIs(bath.mf,     frag.mf)
        self.assertIs(bath.mol,    frag.mol)
        self.assertIs(bath.log,    frag.log)
        self.assertIs(bath.base,   frag.base)
        self.assertIs(bath.c_frag, frag.c_frag)
        self.assertIs(bath.get_environment()[0], bath.c_env_occ)
        self.assertIs(bath.get_environment()[1], bath.c_env_vir)

        frag.reset()

    def test_mp2_bno_bath(self):
        """Test the MP2 BNO bath.
        """

        qemb = self.Embedding(moles[self.key][self.mf_key])
        qemb.iao_fragmentation()
        frags = qemb.add_all_atomic_fragments()

        nocc = np.sum(moles[self.key][self.mf_key].mo_occ > 0)
        nvir = np.sum(moles[self.key][self.mf_key].mo_occ == 0)
        with temporary_seed(1):
            t2a = np.random.random((nocc, nocc, nvir, nvir)) - 0.5
            t2a = 0.25 * (t2a + t2a.swapaxes(0, 1) + t2a.swapaxes(2, 3) + t2a.transpose(1, 0, 3, 2))
            t2b = np.random.random((nocc, nocc, nvir, nvir)) - 0.5
            t2b = 0.25 * (t2b + t2b.swapaxes(0, 1) + t2b.swapaxes(2, 3) + t2b.transpose(1, 0, 3, 2))

        dmet_bath = DMET_Bath(frags[0])
        bath = MP2_BNO_Bath(frags[0], dmet_bath)
        dm1o = bath.make_delta_dm1('occ', t2a, t2b)
        dm1v = bath.make_delta_dm1('vir', t2a, t2b)
        self.assertAlmostEqual(lib.fp(dm1o), 366.180724570873/2, self.PLACES)
        self.assertAlmostEqual(lib.fp(dm1v),  -1.956209725591/2, self.PLACES)
        self.assertIsNone(bath.c_bno_occ)
        self.assertIsNone(bath.c_bno_vir)
        self.assertIsNone(bath.n_bno_occ)
        self.assertIsNone(bath.n_bno_vir)
        self.assertIs(bath.mf,     frags[0].mf)
        self.assertIs(bath.mol,    frags[0].mol)
        self.assertIs(bath.log,    frags[0].log)
        self.assertIs(bath.base,   frags[0].base)
        self.assertIs(bath.c_frag, frags[0].c_frag)

        #bath = MP2_BNO_Bath(frags[0], dmet_bath, local_dm='semi', canonicalize=False)
        #dm1o = bath.make_delta_dm1('occ', t2a, t2b)
        #dm1v = bath.make_delta_dm1('vir', t2a, t2b)
        #self.assertIs(bath.get_mp2_class(), pyscf.mp.MP2)
        #self.assertAlmostEqual(lib.fp(dm1o),  -11.0426240019808/2, self.PLACES)
        #self.assertAlmostEqual(lib.fp(dm1v),   11.2234922296148/2, self.PLACES)

        #bath = MP2_BNO_Bath(frags[0], dmet_bath, local_dm=True, canonicalize=False)
        #dm1o = bath.make_delta_dm1('occ', t2a, t2b)
        #dm1v = bath.make_delta_dm1('vir', t2a, t2b)
        #self.assertIs(bath.get_mp2_class(), pyscf.mp.MP2)
        #self.assertAlmostEqual(lib.fp(dm1o), 382.7366604206307/2, self.PLACES)
        #self.assertAlmostEqual(lib.fp(dm1v),  16.0277297008466/2, self.PLACES)

        #FIXME: bugs #6 and #7 - then add to CellFragmentTests
        #NOTE these values were for an old system

        #tr = lambda c: np.einsum('xi,xi->', c, c.conj())

        #bath = MP2_BNO_Bath(frags[1])
        #bath.kernel()
        #c_bno_occ, n_bno_occ = bath.make_bno_coeff('occ')
        #c_bno_vir, n_bno_vir = bath.make_bno_coeff('vir')
        #self.assertAlmostEqual(tr(c_bno_occ),     0.0, self.PLACES)
        #self.assertAlmostequal(lib.fp(n_bno_occ), 0.0, self.PLACES)

        #bath = MP2_BNO_Bath(frags[1], canonicalize=(False, False))
        #bath.kernel()
        #mp2 = pyscf.mp.mp2.MP2(moles[self.key][self.mf_key])
        #eris = mp2.ao2mo()
        #c_bno_occ, n_bno_occ = bath.make_bno_coeff('occ', eris=eris)
        #c_bno_vir, n_bno_vir = bath.make_bno_coeff('vir', eris=eris)


class UMolFragmentTests(MolFragmentTests):
    mf_key = 'uhf'
    PLACES = 6
    Embedding = UEmbedding
    DMET_Bath = UDMET_Bath

    def trace(self, c):
        ca, cb = c
        return (np.einsum('xi,xi->', ca, ca.conj()) + np.einsum('xi,xi->', cb, cb.conj())) / 2

    def test_properties(self):
        #TODO
        pass

    test_project_amplitude_to_fragment = None
    test_project_ref_orbitals = None
    test_mp2_bno_bath = None


class CellFragmentTests(unittest.TestCase):
    key = 'he2_631g_222'
    PLACES = 8

    def trace(self, c):
        return np.einsum('xi,xi->', c, c.conj())

    def test_iao_atoms(self):
        """Test IAO atomic fragmentation.
        """

        qemb = Embedding(cells[self.key]['rhf'])
        qemb.iao_fragmentation()
        frag = qemb.add_atomic_fragment([0, 1])
        frags = [frag] + frag.add_tsymmetric_fragments([2, 2, 2])

        for frag in frags[0].loop_fragments():
            self.assertAlmostEqual(frag.get_fragment_mf_energy().real, -4.261995344528813, self.PLACES)

    def test_iao_aos(self):
        """Test IAO orbital fragmentation.
        """

        qemb = Embedding(cells[self.key]['rhf'])
        qemb.iao_fragmentation()
        frag = qemb.add_orbital_fragment([0, 1])

        self.assertAlmostEqual(frag.get_fragment_mf_energy().real, -4.261995344528813, self.PLACES)

    def test_sao_atoms(self):
        qemb = Embedding(cells[self.key]['rhf'])
        qemb.sao_fragmentation()
        frags = [qemb.add_atomic_fragment([i*2, i*2+1]) for i in range(len(qemb.kpts))]

        for frag in frags:
            self.assertAlmostEqual(frag.get_fragment_mf_energy().real, -4.261995344528689, self.PLACES)

    def test_sao_aos(self):
        """Test SAO orbital fragmentation.
        """

        qemb = Embedding(cells[self.key]['rhf'])
        qemb.sao_fragmentation()
        frag = qemb.add_orbital_fragment([0, 1, 2, 3])

        self.assertAlmostEqual(frag.get_fragment_mf_energy().real, -4.261995344528689, self.PLACES)

    def test_dmet_bath(self):
        """Test the DMET bath.
        """

        qemb = Embedding(cells[self.key]['rhf'])
        qemb.sao_fragmentation()
        frag = qemb.add_atomic_fragment([0])
        frags = [frag] + frag.add_tsymmetric_fragments([2, 2, 2])

        bath = DMET_Bath(frags[1], frags[1].opts.dmet_threshold)
        c_bath, c_occenv, c_virenv = bath.make_dmet_bath(frags[0].c_env)
        self.assertAlmostEqual(self.trace(c_bath),    3.34569601263718, self.PLACES)
        self.assertAlmostEqual(self.trace(c_occenv),  8.60026059294578, self.PLACES)
        self.assertAlmostEqual(self.trace(c_virenv), 38.23956564189844, self.PLACES)

        #bath = DMET_Bath(frags[0], frags[0].opts.dmet_threshold)
        #c_bath, c_occenv, c_virenv = bath.make_dmet_bath(frags[0].c_env, nbath=c_bath.shape[-1])  #FIXME bug #5 (these values are old)
        #self.assertAlmostEqual(self.trace(c_bath),    3.30703579288843, self.PLACES)
        #self.assertAlmostEqual(self.trace(c_occenv),  8.60108764820888, self.PLACES)
        #self.assertAlmostEqual(self.trace(c_virenv), self.PLACES3.27964350816293, self.PLACES)

        qemb = Embedding(cells[self.key]['rhf'])
        qemb.sao_fragmentation()
        frags = [qemb.add_atomic_fragment([i*2, i*2+1]) for i in range(len(qemb.kpts))]

        bath = DMET_Bath(frags[0], 1e-5)
        c_bath, c_occenv, c_virenv = bath.make_dmet_bath(frags[0].c_env, verbose=False)
        self.assertAlmostEqual(self.trace(c_bath),    0.00000000000000, self.PLACES)
        self.assertAlmostEqual(self.trace(c_occenv),  8.60025893931299, self.PLACES)
        self.assertAlmostEqual(self.trace(c_virenv), 38.23956182500297, self.PLACES)

    def test_mp2_bno_bath(self):
        #TODO
        pass


if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
