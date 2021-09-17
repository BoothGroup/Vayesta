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
from pyscf.mp.mp2 import _mo_without_core
try:
    import pyscf.pbc.df.df_incore
    from pyscf.pbc.df.df_incore import IncoreGDF
except:
    IncoreGDF = pyscf.pbc.df.GDF

from vayesta.core import vlog
from vayesta.core.k2bvk import UnfoldedSCF, unfold_scf
from vayesta.core.util import *
from vayesta.core.ao2mo.dfccsd import make_eris as make_eris_ccsd
from vayesta.core.ao2mo.kao2gmo import gdf_to_pyscf_eris
from vayesta.misc.gdf import GDF
from vayesta import lattmod

from .fragment import QEmbeddingFragment
from .scmf import PDMET_SCMF, Brueckner_SCMF


class QEmbeddingMethod:

    # Shadow these in inherited methods:
    Fragment = QEmbeddingFragment

    @dataclasses.dataclass
    class Options(OptionsBase):
        recalc_vhf: bool = True
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
        if self.opts.recalc_vhf:
            self.log.debug("Recalculating HF potential from MF object.")
        else:
            self.log.debug("Determining HF potential from MO energies and coefficients.")
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
        if self.mo_energy is not None:
            self.log.debugv("MO energies (occ):\n%r", self.mo_energy[self.mo_occ > 0])
            self.log.debugv("MO energies (vir):\n%r", self.mo_energy[self.mo_occ == 0])

        # 3) Fragments
        # ------------
        self.default_fragment_type = None
        self.fragments = []
        self._nfrag_tot = 0 # Total number of fragments created with `add_fragment` method.

        # 4) Other
        # --------
        self.with_scmf = None   # Self-consistent mean-field
        self.c_lo = None        # Local orthogonal orbitals (e.g. Lowdin)


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
        return self.mo_coeff.shape[-1]

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

    def get_eris(self, mo_or_cm):
        """Get ERIS for post-HF methods.

        For unfolded PBC calculations, this folds the MO back into k-space
        and contracts with the k-space three-center integrals..

        Parameters
        ----------
        mo_or_cm: array or one of the following post-HF methods:
                pyscf.mp.mp2.MP2,
                pyscf.cc.ccsd.CCSD,
                or pyscf.cc.rccsd.RCCSD
                or pyscf.cc.dfccsd.DFCCSD
            MO coefficients or correlated method with mo_coeff set.

        Returns
        -------
        eris: array or cm._ChemistsERIs
            ERIs which can be used for the respective correlated method.
        """
        # 1) Input = MO coefficients
        # ==========================
        if isinstance(mo_or_cm, np.ndarray):
            mo = mo_or_cm
            # Temporary fix for Hubbard-models with only onsite repulsion!
            # TODO: k-DF...?
            if hasattr(self.mf, 'with_df') and self.mf.with_df is not None:
                eris = self.mf.with_df.ao2mo(mo, compact=False)
            elif self.mf._eri is not None:
                eris = pyscf.ao2mo.full(self.mf._eri, mo, compact=False)
            else:
                eris = self.mol.ao2mo(mo, compact=False)
            eris = eris.reshape(4*[mo.shape[-1]])
            return eris
        # 2) Input = Correlated method
        # ============================
        # Molecules or supercell:
        cm = mo_or_cm
        if self.kdf is None:
            self.log.debugv("ao2mo method: %r", cm.ao2mo)
            # For PBC DFCCSD calculation we need to use the right Fock matrix (without exxdiv correction!)
            # -> use custom ao2mo
            if self.boundary_cond.startswith('periodic') and isinstance(cm, pyscf.cc.ccsd.CCSD):
                c_act = _mo_without_core(cm, cm.mo_coeff)
                fock = dot(c_act.T, self.get_fock(), c_act)
                mo_energy = fock.diagonal().copy()
                madelung = pyscf.pbc.tools.madelung(self.mol, self.mf.kpt)
                for i in range(cm.get_nocc()):
                    fock[i,i] += madelung
                return make_eris_ccsd(cm, fock=fock, mo_energy=mo_energy)
            elif isinstance(cm, pyscf.mp.dfmp2.DFMP2):
                # TODO: This is a hack, requiring modified PySCF - normal DFMP2 does not store 4c (ov|ov) integrals
                return cm.ao2mo(store_eris=True)
            else:
                return cm.ao2mo()
        # k-point sampled primitive cell:
        eris = gdf_to_pyscf_eris(self.mf, self.kdf, cm, fock=self.get_fock())
        return eris


    def kernel(self, *args, **kwargs):
        # If with_scmf is set, go via the SCMF kernel instead:
        if self.with_scmf is not None:
            return self.with_scmf.kernel(*args, **kwargs)
        return self.kernel_one_iteration(*args, **kwargs)

    def kernel_one_iteration(self, *args, **kwargs):
        raise NotImplementedError("Abstract method")


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

    def get_dmet_energy(self):
        e_dmet = self.mol.energy_nuc()
        for f in self.fragments:
            e_dmet += f.get_fragment_dmet_energy()
        return e_dmet

    def get_t1(self, get_lambda=False, partition=None):
        """Get global CCSD T1- or L1-amplitudes from fragment calculations.

        Parameters
        ----------
        partition: ['first-occ', 'first-vir', 'democratic']
            Partitioning scheme of the T amplitudes. Default: 'first-occ'.

        Returns
        -------
        t1: (n(occ), n(vir)) array
            Global T1- or L1-amplitudes.
        """
        if partition is None: partition = self.opts.wf_partition
        t1 = np.zeros((self.nocc, self.nvir))
        ovlp = self.get_ovlp()
        # Add fragment WFs in intermediate normalization
        for f in self.fragments:
            self.log.debugv("Now adding projected %s-amplitudes of fragment %s", ("L" if get_lambda else "T"), f)
            ro, rv = f.get_rot_to_mf()
            t1f = (f.results.l1 if get_lambda else f.results.get_t1())
            if t1f is None: raise RuntimeError("Amplitudes not found for %s" % f)
            t1f = f.project_amplitude_to_fragment(t1f, partition=partition)
            t1 += einsum('ia,iI,aA->IA', t1f, ro, rv)
        return t1

    def get_t12(self, calc_t1=True, calc_t2=True, get_lambda=False, partition=None, symmetrize=True):
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
        t1 = np.zeros((self.nocc, self.nvir)) if calc_t1 else None
        t2 = np.zeros((self.nocc, self.nocc, self.nvir, self.nvir)) if calc_t2 else None
        ovlp = self.get_ovlp()
        # Add fragment WFs in intermediate normalization
        for f in self.fragments:
            self.log.debugv("Now adding projected %s-amplitudes of fragment %s", ("L" if get_lambda else "T"), f)
            ro, rv = f.get_rot_to_mf()
            if calc_t1:
                t1f = (f.results.l1 if get_lambda else f.results.get_t1())
                if t1f is None: raise RuntimeError("Amplitudes not found for %s" % f)
                t1f = f.project_amplitude_to_fragment(t1f, partition=partition)
                t1 += einsum('ia,iI,aA->IA', t1f, ro, rv)
            if calc_t2:
                t2f = (f.results.l2 if get_lambda else f.results.get_t2())
                if t2f is None: raise RuntimeError("Amplitudes not found for %s" % f)
                t2f = f.project_amplitude_to_fragment(t2f, partition=partition, symmetrize=symmetrize)
                t2 += einsum('ijab,iI,jJ,aA,bB->IJAB', t2f, ro, ro, rv, rv)
        #t2 = (t2 + t2.transpose(1,0,3,2))/2
        #assert np.allclose(t2, t2.transpose(1,0,3,2))
        return t1, t2

    #get_wf_ccsd = get_ccsd_t12

    def make_rdm1_demo(self, ao_basis=False, add_mf=True, symmetrize=True):
        """Make democratically partitioned one-particle reduced density-matrix from fragment calculations.

        Warning: A democratically partitioned DM is only expected to yield good results
        for Lowdin-AO or site fragmentation.

        Parameters
        ----------
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
        if add_mf:
            sc = np.dot(self.get_ovlp(), self.mo_coeff)
            dm1_mf = dot(sc.T, self.mf.make_rdm1(), sc)
            dm1 = dm1_mf.copy()
        else:
            dm1 = np.zeros((self.nmo, self.nmo))
        for f in self.fragments:
            self.log.debugv("Now adding projected DM of fragment %s", f)
            if f.results.dm1 is None:
                raise RuntimeError("DM1 not calculated for fragment %s!" % f)
            if self.opts.dm_with_frozen:
                cf = f.mo_coeff
            else:
                cf = f.c_active
            rf = dot(self.mo_coeff.T, self.get_ovlp(), cf)
            if add_mf:
                # Subtract double counting:
                ddm = (f.results.dm1 - dot(rf.T, dm1_mf, rf))
            else:
                ddm = f.results.dm1
            pf = f.get_fragment_projector(cf)
            dm1 += einsum('xi,ij,px,qj->pq', pf, ddm, rf, rf)
        if ao_basis:
            dm1 = dot(self.mo_coeff, dm1, self.mo_coeff.T)
        if symmetrize:
            dm1 = (dm1 + dm1.T)/2
        return dm1

    def make_rdm1_ccsd(self, ao_basis=False, symmetrize=True, partition=None, t_as_lambda=False, slow=False):
        """Make one-particle reduced density-matrix from partitioned fragment CCSD wave functions.

        Parameters
        ----------
        ao_basis: bool, optional
            Return the density-matrix in the AO basis. Default: False.
        symmetrize: bool, optional
            Symmetrize the density-matrix at the end of the calculation. Default: True.
        partition: ['first-occ', 'first-vir', 'democratic']
        t_as_lambda: bool, optional
            Use T-amplitudes instead of Lambda-amplitudes for CCSD density matrix. Default: False.
        slow: bool, optional
            Combine to global CCSD wave function first, then build density matrix.
            Equivalent, but does not scale well. Default: False

        Returns
        -------
        dm1: (n, n) array
            One-particle reduced density matrix in AO (if `ao_basis=True`) or MO basis (default).
        """

        if slow:
            t1, t2 = self.get_t12(partition=partition)
            cc = pyscf.cc.ccsd.CCSD(self.mf)
            #cc.conv_tol = 1e-12
            #cc.conv_tol_normt = 1e-10
            #if 'full-l' in partition:
            #    l1 = l2 = None
            if t_as_lambda:
                l1, l2 = t1, t2
            else:
                l1, l2 = self.get_t12(get_lambda=True, partition=partition)
            dm1 = cc.make_rdm1(t1=t1, t2=t2, l1=l1, l2=l2, with_frozen=False)

        else:
            # T1/L1-amplitudes can be summed directly
            t1 = self.get_t1(partition=partition)
            l1 = (t1 if t_as_lambda else self.get_t1(get_lambda=True, partition=partition))

            # --- Preconstruct some C^T.S.C rotation matrices:
            # Fragment orbital projectors
            pf = []
            # Fragment to mean-field occupied/virtual
            f2mfo = []
            f2mfv = []
            # Fragment to other fragment occupied/virtual
            f2fo = [[] for i in range(self.nfrag)]
            f2fv = [[] for i in range(self.nfrag)]
            ovlp = self.get_ovlp()
            for i1, f1 in enumerate(self.fragments):
                pf.append(f1.get_fragment_projector(f1.c_active_occ))
                cso = np.dot(f1.c_active_occ.T, ovlp)
                csv = np.dot(f1.c_active_vir.T, ovlp)
                f2mfo.append(np.dot(cso, self.mo_coeff_occ))
                f2mfv.append(np.dot(csv, self.mo_coeff_vir))
                for i2, f2 in enumerate(self.fragments):
                    f2fo[i1].append(np.dot(cso, f2.c_active_occ))
                    f2fv[i1].append(np.dot(csv, f2.c_active_vir))

            # --- Loop over pairs of fragments and add projected density-matrix contributions
            nocc, nvir = t1.shape
            doo = np.zeros((nocc, nocc))
            dvv = np.zeros((nvir, nvir))
            dov = (t1 + l1 - einsum('ie,me,ma->ia', t1, l1, t1))
            for i1, f1 in enumerate(self.fragments):
                theta = f1.results.get_t2()
                theta = (2*theta - theta.transpose(0,1,3,2))
                theta = f1.project_amplitude_to_fragment(theta, partition=partition)
                # Intermediates [leave left index in cluster basis]:
                doo_f1 = np.zeros((f1.n_active_occ, nocc))
                dvv_f1 = np.zeros((f1.n_active_vir, nvir))
                dov += einsum('imae,ip,mM,aq,eE,ME->pq', theta, f2mfo[i1], f2mfo[i1], f2mfv[i1], f2mfv[i1], l1)
                for i2, f2 in enumerate(self.fragments):
                    #l2 = (f2.results.t2 if t_as_lambda else f2.results.l2)
                    l2 = (f2.results.get_t2() if t_as_lambda else f2.results.l2)
                    l2 = f2.project_amplitude_to_fragment(l2, partition=partition)
                    ## Theta_jk^ab * l_ik^ab -> ij
                    #doo -= einsum('jkab,IKAB,jp,kK,aA,bB,Iq->pq', theta_f1, l2_f2,
                    #        f2mfo[i1], f2fo[i1][i2], f2fv[i1][i2], f2fv[i1][i2], f2mfo[i2])
                    ## Theta_ji^ca * l_ji^cb -> ab
                    #dvv += einsum('jica,JICB,jJ,iI,cC,ap,Bq->pq', theta_f1, l2_f2,
                    #        f2fo[i1][i2], f2fo[i1][i2], f2fv[i1][i2], f2mfv[i1], f2mfv[i2])

                    # Theta_jk^ab * l_ik^ab -> ij
                    doo_f1 -= einsum('jkab,IKAB,kK,aA,bB,Iq->jq', theta, l2,
                            f2fo[i1][i2], f2fv[i1][i2], f2fv[i1][i2], f2mfo[i2])
                    # Theta_ji^ca * l_ji^cb -> ab
                    dvv_f1 += einsum('jica,JICB,jJ,iI,cC,Bq->aq', theta, l2,
                            f2fo[i1][i2], f2fo[i1][i2], f2fv[i1][i2], f2mfv[i2])

                    # Theta_jk^ab * l_ik^ab -> ij
                    #doo -= einsum('jx,Iy,jkab,IKAB,xp,kK,aA,bB,yq->pq',
                    #        pf[i1], pf[i2], theta_f1, l2_f2,
                    #        f2mfo[i1], f2fo[i1][i2], f2fv[i1][i2], f2fv[i1][i2], f2mfo[i2])
                    ### Theta_ji^ca * l_ji^cb -> ab
                    #dvv += einsum('jx,Jy,jica,JICB,xy,iI,cC,ap,Bq->pq',
                    #        pf[i1], pf[i2], theta_f1, l2_f2,
                    #        f2fo[i1][i2], f2fo[i1][i2], f2fv[i1][i2], f2mfv[i1], f2mfv[i2])
                doo += np.dot(f2mfo[i1].T, doo_f1)
                dvv += np.dot(f2mfv[i1].T, dvv_f1)

            dov += einsum('im,ma->ia', doo, t1)
            dov -= einsum('ie,ae->ia', t1, dvv)
            doo -= einsum('ja,ia->ij', t1, l1)
            dvv += einsum('ia,ib->ab', t1, l1)

            nmo = (nocc + nvir)
            occ, vir = np.s_[:nocc], np.s_[nocc:]
            dm1 = np.zeros((nmo, nmo))
            dm1[occ,occ] = (doo + doo.T)
            dm1[vir,vir] = (dvv + dvv.T)
            dm1[occ,vir] = dov
            dm1[vir,occ] = dov.T
            dm1[np.diag_indices(nocc)] += 2

        if ao_basis:
            dm1 = dot(self.mo_coeff, dm1, self.mo_coeff.T)
        if symmetrize:
            dm1 = (dm1 + dm1.T)/2
        return dm1

    def make_rdm2_demo(self, ao_basis=False, add_mf=True, symmetrize=True):
        """Make democratically partitioned two-particle reduced density-matrix from fragment calculations.

        Warning: A democratically partitioned DM is only expected to yield good results
        for Lowdin-AO or site fragmentation.

        Parameters
        ----------
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
        if ao_basis:
            dm2 = einsum('ijkl,pi,qj,rk,sl->pqrs', dm2, *(4*[self.mo_coeff]))
        if symmetrize:
            dm2 = (dm2 + dm2.transpose(1,0,3,2))/2
        return dm2


    def make_rdm2_ccsd(self, ao_basis=False, symmetrize=True, partition=None, t_as_lambda=False, slow=True):
        """Recreate global two-particle reduced density-matrix from fragment calculations.

        Parameters
        ----------
        ao_basis: bool, optional
            Return the density-matrix in the AO basis. Default: False.
        symmetrize: bool, optional
            Symmetrize the density-matrix at the end of the calculation. Default: True.
        partition: ['first-occ', 'first-vir', 'democratic']
        t_as_lambda: bool, optional
            Use T-amplitudes instead of Lambda-amplitudes for CCSD density matrix. Default: False.
        slow: bool, optional
            Combine to global CCSD wave function first, then build density matrix.
            Equivalent, but does not scale well. Default: True.

        Returns
        -------
        dm2: (n, n, n, n) array
            Two-particle reduced density matrix in AO (if `ao_basis=True`) or MO basis (default).
        """

        if slow:
            t1, t2 = self.get_t12(partition=partition)
            cc = pyscf.cc.ccsd.CCSD(self.mf)
            #if 'l12_full' in partition:
            #    l1 = l2 = None
            if t_as_lambda:
                l1, l2 = t1, t2
            else:
                l1, l2 = self.get_t12(get_lambda=True, partition=partition)
            dm2 = cc.make_rdm2(t1=t1, t2=t2, l1=l1, l2=l2, with_frozen=False)
        else:
            raise NotImplementedError()
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
            raise ValueError("Unknown population analysis kind: %r" % kind)
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

    # --- Mean-field updates

    def reset_fragments(self, *args, **kwargs):
        for f in self.fragments:
            f.reset(*args, **kwargs)

    def pdmet_scmf(self, *args, **kwargs):
        """Decorator for p-DMET."""
        self.with_scmf = PDMET_SCMF(self, *args, **kwargs)
        return self

    def brueckner_scmf(self, *args, **kwargs):
        """Decorator for Brueckner-DMET."""
        self.with_scmf = Brueckner_SCMF(self, *args, **kwargs)
        return self
