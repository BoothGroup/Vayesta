import logging
from timeit import default_timer as timer
from datetime import datetime
import dataclasses
import copy

import numpy as np

import pyscf
import pyscf.gto
import pyscf.mp
import pyscf.cc
import pyscf.lo
import pyscf.pbc
import pyscf.pbc.df
import pyscf.pbc.tools
import pyscf.lib
try:
    import pyscf.pbc.df.df_incore
    from pyscf.pbc.df.df_incore import IncoreGDF
except:
    IncoreGDF = pyscf.pbc.df.GDF

from vayesta.core import vlog
from vayesta.core.k2bvk import UnfoldedSCF, unfold_scf
from vayesta.core.util import *
from vayesta.core.kao2gmo import gdf_to_pyscf_eris
from vayesta.misc.gdf import GDF

from .fragment import QEmbeddingFragment


class QEmbeddingMethod:

    # Shadow these in inherited methods:
    Fragment = QEmbeddingFragment

    @dataclasses.dataclass
    class Options(OptionsBase):
        recalc_veff: bool = False
        wf_partition: str = 'first-occ'     # ['first-occ', 'first-vir', 'democratic']


    def __init__(self, mf, options=None, log=None, **kwargs):
        """Abstract base class for quantum embedding methods.

        Parameters
        ----------
        mf : pyscf.scf.SCF
            PySCF mean-field object.
        log : logging.Logger, optional
            Logger object. Default: None

        Attributes
        ----------
        mol
        has_lattice_vectors
        boundary_cond
        nao
        ncells
        nmo
        nfrag
        e_mf
        log : logging.Logger
            Logger object.
        self.mf : pyscf.scf.SCF
            PySCF mean-field object.
        self.mo_energy : (nMO) array
            MO energies.
        self.mo_occ : (nMO) array
            MO occupation numbers.
        self.mo_coeff : (nAO, nMO) array
            MO coefficients.
        self.default_fragment_type : {'IAO', 'Lowdin-AO', 'AO', 'Site'}
            The default type for fragment orbitals, when a new fragment is created.
        self.fragments : list
            List of fragments for embedding calculation.
        self.kcell : pyscf.pbc.gto.Cell
            For k-point sampled mean-field calculation, which have been unfolded to the supercell,
            this will hold the original primitive unit cell.
        self.kpts : (nK, 3) array
            For k-point sampled mean-field calculation, which have been unfolded to the supercell,
            this will hold the original k-points.
        self.kdf : pyscf.pbc.df.GDF
            For k-point sampled mean-field calculation, which have been unfolded to the supercell,
            this will hold the original Gaussian density-fitting object.

        Depending on which fragmentation is initialized, the following attributes my also be present:

        IAO
        ---
        iao_minao : str
            Minimal reference basis set for IAOs.
        iao_labels : list
            IAO labels.
        iao_coeff : (nAO, nIAO) array
            IAO coefficients.
        iao_rest_coeff : (nAO, nRest) array
            Remaining MO coefficients
        iao_occup : (nIAO) array
            IAO occupation numbers.

        Lowdin-AO
        ---------
        lao_labels : list
            Lowdin-AO labels.
        lao_coeff : (nAO, nLAO) array
            Lowdin-AO coefficients.
        lao_occup : (nLAO) array
            Lowdin-AO occupation numbers.

        AO
        --
        ao_labels : list
            AO labels.

        Site
        ----
        site_labels : list
            Site labels.
        """

        # 1) Logging
        # ----------
        self.log = log or logging.getLogger(__name__)
        self.log.info("Initializing %s" % self.__class__.__name__)
        self.log.info("=============%s" % (len(str(self.__class__.__name__))*"="))

        # Options
        # -------
        if options is None:
            options = self.Options(**kwargs)
        else:
            options = options.replace(kwargs)
        self.opts = options

        # 2) Mean-field
        # -------------
        self._mf_orig = mf      # Keep track of original mean-field object - be careful not to modify in any way, to avoid side effects!
        mf = copy.copy(mf)
        self.log.debug("type(MF)= %r", type(mf))
        if hasattr(mf, 'kpts') and mf.kpts is not None:
            mf = unfold_scf(mf)
        if isinstance(mf, UnfoldedSCF):
            self.kcell, self.kpts, self.kdf = mf.kmf.mol, mf.kmf.kpts, mf.kmf.with_df
        else:
            self.kcell = self.kpts = self.kdf = None
        self.mf = mf
        # Cached AO integral matrices (to improve efficiency):
        self._ovlp = None
        self._hcore = None
        self._veff = None
        # HF potential
        if self.opts.recalc_veff:
            self.log.debug("Recalculating effective potential from MF object.")
        else:
            self.log.debug("Determining effective potential from MOs.")
            cs = np.dot(self.mo_coeff.T, self.get_ovlp())
            fock = np.dot(cs.T*self.mo_energy, cs)
            self._veff = fock - self.get_hcore()
        # Some MF output
        if self.mf.converged:
            self.log.info("E(MF)= %+16.8f Ha", self.e_mf)
        else:
            self.log.warning("E(MF)= %+16.8f Ha (not converged!)", self.e_mf)
        self.log.info("n(AO)= %4d  n(MO)= %4d  n(linear dep.)= %4d", self.nao, self.nmo, self.nao-self.nmo)
        idterr = self.mo_coeff.T.dot(self.get_ovlp()).dot(self.mo_coeff) - np.eye(self.nmo)
        self.log.log(logging.ERROR if np.linalg.norm(idterr) > 1e-5 else logging.DEBUG,
                "Orthogonality error of MF orbitals: L(2)= %.2e  L(inf)= %.2e", np.linalg.norm(idterr), abs(idterr).max())
        self.log.debugv("MO energies (occ):\n%r", self.mo_energy[self.mo_occ > 0])
        self.log.debugv("MO energies (vir):\n%r", self.mo_energy[self.mo_occ == 0])

        # 3) Fragments
        # ------------
        self.default_fragment_type = None
        self.fragments = []
        self._nfrag_tot = 0 # Total number of fragments created with `add_fragment` method.

        # 4) Other
        # --------
        self.c_lo = None  # Local orthogonal orbitals (e.g. Lowdin)


    # --- Basic properties and methods
    # ================================

    def __repr__(self):
        keys = ['mf']
        fmt = ('%s(' + len(keys)*'%s: %r, ')[:-2] + ')'
        values = [self.__dict__[k] for k in keys]
        return fmt % (self.__class__.__name__, *[x for y in zip(keys, values) for x in y])

    # Mol/Cell properties

    @property
    def mol(self):
        """Mole or Cell object."""
        return self.mf.mol

    @property
    def has_lattice_vectors(self):
        """Flag if self.mol has lattice vectors defined."""
        return (hasattr(self.mol, 'a') and self.mol.a is not None)
        # This would be better, but would trigger PBC code for Hubbard models, which have lattice vectors defined,
        # but not 'a':
        #return hasattr(self.mol, 'lattice_vectors')

    @property
    def boundary_cond(self):
        """Type of boundary condition."""
        if not self.has_lattice_vectors:
            return 'open'
        if self.mol.dimension == 1:
            return 'periodic-1D'
        if self.mol.dimension == 2:
            return 'periodic-2D'
        return 'periodic'

    @property
    def nao(self):
        """Number of atomic orbitals."""
        return self.mol.nao_nr()

    @property
    def ncells(self):
        """Number of primitive cells within supercell."""
        if self.kpts is None:
            return 1
        return len(self.kpts)

    # Mean-field properties

    @property
    def mo_energy(self):
        """Molecular orbital energies."""
        return self.mf.mo_energy

    @property
    def mo_coeff(self):
        """Molecular orbital coefficients."""
        return self.mf.mo_coeff

    @property
    def mo_occ(self):
        """Molecular orbital occupations."""
        return self.mf.mo_occ

    # MOs setters:

    @mo_energy.setter
    def mo_energy(self, mo_energy):
        """Updating the MOs resets the effective potential cache `_veff`."""
        self.log.debugv("MF attribute 'mo_energy' is updated; deleting cached _veff.")
        self._veff = None
        self.mf.mo_energy = mo_energy

    @mo_coeff.setter
    def mo_coeff(self, mo_coeff):
        """Updating the MOs resets the effective potential cache `_veff`."""
        self.log.debugv("MF attribute 'mo_coeff' is updated; deleting chached _veff.")
        self._veff = None
        self.mf.mo_coeff = mo_coeff

    @mo_occ.setter
    def mo_occ(self, mo_occ):
        """Updating the MOs resets the effective potential cache `_veff`."""
        self.log.debugv("MF attribute 'mo_occ' is updated; deleting chached _veff.")
        self._veff = None
        self.mf.mo_occ = mo_occ

    @property
    def nmo(self):
        """Total number of molecular orbitals (MOs)."""
        return len(self.mo_energy)

    @property
    def nocc(self):
        """Number of occupied MOs."""
        return np.count_nonzero(self.mo_occ > 0)

    @property
    def nvir(self):
        """Number of virtual MOs."""
        return np.count_nonzero(self.mo_occ == 0)

    @property
    def mo_coeff_occ(self):
        """Occupied MO coefficients."""
        return self.mo_coeff[:,:self.nocc]

    @property
    def mo_coeff_vir(self):
        """Virtual MO coefficients."""
        return self.mo_coeff[:,self.nocc:]

    @property
    def e_mf(self):
        """Total mean-field energy per unit cell (not unfolded supercell!).
        Note that the input unit cell itself can be a supercell, in which case
        `e_mf` refers to this cell.
        """
        return self.mf.e_tot/self.ncells

    # Embedding properties

    @property
    def nfrag(self):
        """Number of fragments."""
        return len(self.fragments)

    def loop(self):
        """Loop over fragments."""
        for frag in self.fragments:
            yield frag

    # --- Integral methods
    # ====================

    @cached_method('_ovlp')
    def get_ovlp(self, *args, **kwargs):
        return self.mf.get_ovlp(*args, **kwargs)

    def get_ovlp_power(self, power):
        """get power of AO overlap matrix.

        For unfolded calculations, this uses the k-point sampled overlap, for better performance and accuracy.

        Parameters
        ----------
        power : float
            Matrix power.

        Returns
        -------
        spow : (n(AO), n(AO)) array
            Matrix power of AO overlap matrix
        """
        if power == 1: return self.get_ovlp()
        if self.kcell is None:
            e, v = np.linalg.eigh(self.get_ovlp())
            return np.dot(v*(e**power), v.T.conj())
        sk = self.kcell.pbc_intor('int1e_ovlp', hermi=1, kpts=self.kpts, pbcopt=pyscf.lib.c_null_ptr())
        ek, vk = np.linalg.eigh(sk)
        spowk = einsum('kai,ki,kbi->kab', vk, ek**power, vk.conj())
        spow = pyscf.pbc.tools.k2gamma.to_supercell_ao_integrals(self.kcell, self.kpts, spowk)
        return spow

    @cached_method('_hcore')
    def get_hcore(self, *args, **kwargs):
        return self.mf.get_hcore(*args, **kwargs)

    @cached_method('_veff')
    def get_veff(self, *args, **kwargs):
        """Hartree-Fock Coulomb and exchange potential."""
        return self.mf.get_veff(*args, **kwargs)

    def get_fock(self, recalc=False):
        """Fock matrix in AO basis."""
        return self.get_hcore() + self.get_veff()

    def get_eris(self, cm):
        """Get ERIS for post-HF methods.

        For unfolded PBC calculations, this folds the MO back into k-space
        and contracts with the k-space three-center integrals..

        Parameters
        ----------
        cm: pyscf.mp.mp2.MP2, pyscf.cc.ccsd.CCSD, or pyscf.cc.rccsd.RCCSD
            Correlated method, must have mo_coeff set.

        Returns
        -------
        eris: pyscf.mp.mp2._ChemistsERIs, pyscf.cc.ccsd._ChemistsERIs, or pyscf.cc.rccsd._ChemistsERIs
            ERIs which can be used for the respective correlated method.
        """
        # Molecules or supercell:
        if self.kdf is None:
            self.log.debugv("ao2mo method: %r", cm.ao2mo)
            if isinstance(cm, pyscf.mp.dfmp2.DFMP2):
                # TODO: This is a hack, requiring modified PySCF - normal DFMP2 does not store 4c (ov|ov) integrals
                return cm.ao2mo(store_eris=True)
            else:
                return cm.ao2mo()
        # k-point sampled primitive cell:
        eris = gdf_to_pyscf_eris(self.mf, self.kdf, cm, fock=self.get_fock())
        return eris

    # Fragmentation specific routines
    # -------------------------------
    from .fragmentation import init_fragmentation
    from .fragmentation import init_iao_fragmentation
    from .fragmentation import init_lowdin_fragmentation
    from .fragmentation import init_ao_fragmentation
    from .fragmentation import init_site_fragmentation
    from .fragmentation import make_atom_fragment
    from .fragmentation import get_ao_labels
    from .fragmentation import get_ao_indices
    from .fragmentation import make_ao_fragment
    from .fragmentation import add_fragment
    from .fragmentation import make_all_atom_fragments
    from .fragmentation import make_iao_coeffs
    from .fragmentation import get_fragment_occupancy
    from .fragmentation import get_iao_labels
    from .fragmentation import get_subset_ao_projector
    from .fragmentation import get_ao_projector

    # Symmetry between fragments
    # --------------------------

    def get_symmetry_parent_fragments(self):
        """Returns a list of all fragments, which are parents to symmetry related child fragments.

        Returns
        -------
        parents: list
            A list of all parent fragments, ordered in the same way as they appear in `self.fragments`.
        """
        parents = []
        for f in self.fragments:
            if f.sym_parent is None:
                parents.append(f)
        return parents


    def get_symmetry_child_fragments(self, include_parents=False):
        """Returns a list of all fragments, which are children to symmetry related parent fragments.

        Parameters
        ----------
        include_parents: bool, optional
            If true, the parent fragment of each symmetry group is prepended to each symmetry sublist.

        Returns
        -------
        children: list of lists
            A list with the length of the number of parent fragments in the system, each element
            being another list containing all the children fragments of the given parent fragment.
            Both the outer and inner lists are ordered in the same way that the fragments appear in `self.fragments`.
        """
        parents = self.get_symmetry_parent_fragments()
        if include_parents:
            children = [[p] for p in parents]
        else:
            children = [[] for p in parents]
        parent_ids = [p.id for p in parents]
        for f in self.fragments:
            if f.sym_parent is None: continue
            pid = f.sym_parent.id
            assert (pid in parent_ids)
            idx = parent_ids.index(pid)
            children[idx].append(f)
        return children

    # Results
    # -------


    def get_wf_ccsd(self, get_lambda=False, partition=None):
        """Get global CCSD wave function (T1 and T2 amplitudes) from fragment calculations.

        Parameters
        ----------
        partition: ['first-occ', 'first-vir', 'democratic']
            Partitioning scheme of the T amplitudes. Default: 'first-occ'.

        Returns
        -------
        t1: (n(occ), n(vir)) array
            Global T1 amplitudes.
        t2: (n(occ), n(occ), n(vir), n(vir)) array
            Global T2 amplitudes.
        """
        if partition is None: partition = self.opts.wf_partition
        t1 = np.zeros((self.nocc, self.nvir))
        t2 = np.zeros((self.nocc, self.nocc, self.nvir, self.nvir))
        ovlp = self.get_ovlp()
        # Add fragment WFs in intermediate normalization
        for f in self.fragments:
            if f.results.t2 is not None and not get_lambda:
                a1, a2 = f.results.t1, f.results.t2
            elif f.results.l2 is not None and get_lambda:
                a1, a2 = f.results.l1, f.results.l2
            elif f.results.c2 is not None:
                a1, a2 = f.results.convert_amp_c_to_t()
            else:
                raise RuntimeError("No amplitudes found for fragment %s" % f)
            a1 = f.project_amplitude_to_fragment(a1, partition=partition)
            a2 = f.project_amplitude_to_fragment(a2, partition=partition, symmetrize=True)
            ro = np.linalg.multi_dot((f.c_active_occ.T, ovlp, self.mo_coeff_occ))
            rv = np.linalg.multi_dot((f.c_active_vir.T, ovlp, self.mo_coeff_vir))
            t1 += einsum('ia,iI,aA->IA', a1, ro, rv)
            t2 += einsum('ijab,iI,jJ,aA,bB->IJAB', a2, ro, ro, rv, rv)
        return t1, t2


    def make_rdm1(self, partition='dm', ao_basis=False, add_mf=True, symmetrize=True,
            wf_partition=None):
        """Recreate global one-particle reduced density-matrix from fragment calculations.

        Warning: A democratically partitioned DM is only expected to yield good results
        for Lowdin-AO or site fragmentation.

        Parameters
        ----------
        partition: ['dm', 'wf-ccsd']
            Type of density-matrix partioning:
                'dm': The density-matrix of each cluster is projected in it's first index
                      onto the corresponding fragment space.
                'wf-t12': The wave-function amplitudes (only T1, T2) of each cluster are projected in it's first
                      occupied index onto the corresponding fragment space.
            Default: 'dm'
        ao_basis: bool, optional
            Return the density-matrix in the AO basis. Default: False.
        add_mf: bool, optional
            Add the mean-field contribution to the density-matrix (double counting is accounted for).
            Is only used if `partition = 'dm'`. Default: True.
        symmetrize: bool, optional
            Symmetrize the density-matrix at the end of the calculation. Default: True.

        Returns
        -------
        dm1: (n, n) array
            One-particle reduced density matrix in AO (if `ao_basis=True`) or MO basis (default).
        """
        partition = partition.lower()
        if partition == 'dm':
            if add_mf:
                #sc = np.dot(self.get_ovlp(), self.mo_coeff)
                sc = np.dot(self.get_ovlp(), self.mf.mo_coeff)
                dm1_mf = np.linalg.multi_dot((sc.T, self.mf.make_rdm1(), sc))
                dm1 = dm1_mf.copy()
            else:
                dm1 = np.zeros((self.nmo, self.nmo))
            for f in self.fragments:
                if f.results.dm1 is None:
                    raise RuntimeError("DM1 not calculated for fragment %s!" % f)
                if self.opts.dm_with_frozen:
                    cf = f.mo_coeff
                else:
                    cf = f.c_active
                #rf = np.linalg.multi_dot((self.mo_coeff.T, self.get_ovlp(), cf))
                rf = np.linalg.multi_dot((self.mf.mo_coeff.T, self.get_ovlp(), cf))
                if add_mf:
                    # Subtract double counting:
                    ddm = (f.results.dm1 - np.linalg.multi_dot((rf.T, dm1_mf, rf)))
                else:
                    ddm = f.results.dm1
                pf = f.get_fragment_projector(cf)
                dm1 += einsum('xi,ij,px,qj->pq', pf, ddm, rf, rf)
        elif partition.startswith('wf-t12'):
            t1, t2 = self.get_wf_ccsd(partition=wf_partition)
            cc = pyscf.cc.ccsd.CCSD(self.mf)
            if 'l12_full' in partition:
                l1 = l2 = None
            elif 'l12' in partition:
                l1, l2 = self.get_wf_ccsd(get_lambda=True, partition=wf_partition)
            else:
                l1, l2 = t1, t2
            dm1 = cc.make_rdm1(t1=t1, t2=t2, l1=l1, l2=l2, with_frozen=False)
        else:
            raise NotImplementedError("Unknown make_rdm1 partition= %r", partition)

        if ao_basis:
            #dm1 = np.linalg.multi_dot((self.mo_coeff, dm1, self.mo_coeff.T))
            dm1 = np.linalg.multi_dot((self.mf.mo_coeff, dm1, self.mf.mo_coeff.T))
        if symmetrize:
            dm1 = (dm1 + dm1.T)/2
        return dm1


    def make_rdm2(self, partition='dm', ao_basis=False, add_mf=True, symmetrize=True,
            wf_partition=None):
        """Recreate global two-particle reduced density-matrix from fragment calculations.

        Warning: A democratically partitioned DM is only expected to yield good results
        for Lowdin-AO or site fragmentation.

        Parameters
        ----------
        partition: ['dm', 'wf-t12', 'wf-t12-l12', 'wf-t12-l12_full']
            Type of density-matrix partioning:
                'dm': The density-matrix of each cluster is projected in it's first index
                      onto the corresponding fragment space.
                'wf-t12': The wave-function amplitudes (only T1, T2) of each cluster are projected in it's first
                      occupied index onto the corresponding fragment space.
            Default: 'dm'
        ao_basis: bool, optional
            Return the density-matrix in the AO basis. Default: False.
        add_mf: bool, optional
            Add the mean-field contribution to the density-matrix (double counting is accounted for).
            Is only used if `partition = 'dm'`. Default: True.
        symmetrize: bool, optional
            Symmetrize the density-matrix at the end of the calculation. Default: True.

        Returns
        -------
        dm2: (n, n, n, n) array
            Two-particle reduced density matrix in AO (if `ao_basis=True`) or MO basis (default).
        """
        partition = partition.lower()
        if partition == 'dm':
            if add_mf:
                #dm2_mf = np.zeros(4*[self.nmo])
                #for i in range(self.nocc):
                #    for j in range(self.nocc):
                #        dm2_mf[i,i,j,j] += 4.0
                #        dm2_mf[i,j,j,i] -= 2.0
                sc = np.dot(self.get_ovlp(), self.mo_coeff)
                dm1_mf = np.linalg.multi_dot((sc.T, self.mf.make_rdm1(), sc))
                dm2_mf = einsum('ij,kl->ijkl', dm1_mf, dm1_mf) - einsum('il,jk->ijkl', dm1_mf, dm1_mf)/2
                dm2 = dm2_mf.copy()
            else:
                dm2 = np.zeros((self.nmo, self.nmo, self.nmo, self.nmo))

            for f in self.fragments:
                if f.results.dm2 is None:
                    raise RuntimeError("DM2 not calculated for fragment %s!" % f)
                if self.opts.dm_with_frozen:
                    cf = f.mo_coeff
                else:
                    cf = f.c_active
                rf = np.linalg.multi_dot((self.mo_coeff.T, self.get_ovlp(), cf))
                if add_mf:
                    # Subtract double counting:
                    ddm = (f.results.dm2 - einsum('IJKL,Ii,Jj,Kk,Ll->ijkl', dm2_mf, rf, rf, rf, rf))
                else:
                    ddm = f.results.dm2
                pf = f.get_fragment_projector(cf)
                dm2 += einsum('xi,ijkl,px,qj,rk,sl->pqrs', pf, ddm, rf, rf, rf, rf)
        elif partition.startswith('wf-t12'):
            t1, t2 = self.get_wf_ccsd(partition=wf_partition)
            cc = pyscf.cc.ccsd.CCSD(self.mf)
            if 'l12_full' in partition:
                l1 = l2 = None
            elif 'l12' in partition:
                l1, l2 = self.get_wf_ccsd(get_lambda=True, partition=wf_partition)
            else:
                l1, l2 = t1, t2
            dm2 = cc.make_rdm2(t1=t1, t2=t2, l1=l1, l2=l2, with_frozen=False)
        else:
            raise NotImplementedError("Unknown make_rdm2 partition= %r", partition)

        if ao_basis:
            dm2 = einsum('ijkl,pi,qj,rk,sl->pqrs', dm2, *(4*[self.mo_coeff]))
        if symmetrize:
            dm2 = (dm2 + dm2.transpose(1,0,3,2))/2
        return dm2


    # Utility
    # -------

    def pop_analysis(self, dm1, mo_coeff=None, kind='lo', c_lo=None, filename=None, filemode='a', verbose=True):
        """
        Parameters
        ----------
        dm1 : (N, N) array
            If `mo_coeff` is None, AO representation is assumed!
        kind : {'mulliken', 'lo'}
            Kind of population analysis. Default: 'lo'.
        c_lo :
            Local orbital coefficients, only used if kind=='lo'. Default: Lowdin AOs.
        """
        if mo_coeff is not None:
            dm1 = einsum('ai,ij,bj->ab', mo_coeff, dm1, mo_coeff)
        if kind.lower() == 'mulliken':
            pop = einsum('ab,ba->a', dm1, self.get_ovlp())
            name = "Mulliken"
        elif kind.lower() == 'lo':
            name = "Local orbital"
            if c_lo is None:
                c_lo = self.c_lo
                name = "Lowdin"
            if c_lo is None:
                # Lowdin population analysis:
                # Avoid pre_orth_ao step!
                #self.c_lo = c_lo = pyscf.lo.orth_ao(self.mol, 'lowdin')
                #self.c_lo = c_lo = pyscf.lo.orth_ao(self.mol, 'meta-lowdin', pre_orth_ao=None)
                self.c_lo = c_lo = self.get_ovlp_power(power=-0.5)
            cs = np.dot(c_lo.T, self.get_ovlp())
            pop = einsum('ia,ab,ib->i', cs, dm1, cs)
        else:
            raise ValueError("Unknown population analysis kind: %s" % kind)
        # Get atomic charges
        elecs = np.zeros(self.mol.natm)
        for i, label in enumerate(self.mol.ao_labels(fmt=None)):
            elecs[label[0]] += pop[i]
        chg = self.mol.atom_charges() - elecs

        if not verbose:
            return pop, chg

        if filename is None:
            write = lambda *args : self.log.info(*args)
            write("%s population analysis", name)
            write("%s--------------------", len(name)*'-')
        else:
            f = open(filename, filemode)
            write = lambda fmt, *args : f.write((fmt+'\n') % args)
            tstamp = datetime.now()
            self.log.info("[%s] Writing population analysis to file \"%s\"", tstamp, filename)
            write("[%s] %s population analysis" % (tstamp, name))
            write("-%s--%s--------------------" % (26*'-', len(name)*'-'))

        aoslices = self.mol.aoslice_by_atom()[:,2:]
        aolabels = self.mol.ao_labels()

        for atom in range(self.mol.natm):
            write("> Charge of atom %d%-6s= % 11.8f (% 11.8f electrons)", atom, self.mol.atom_symbol(atom), chg[atom], elecs[atom])
            aos = aoslices[atom]
            for ao in range(aos[0], aos[1]):
                label = aolabels[ao]
                write("    %4d %-16s= % 11.8f" % (ao, label, pop[ao]))

        if filename is not None:
            f.close()
        return pop, chg

