import dataclasses
import itertools
import copy
import os.path

import numpy as np
import scipy
import scipy.linalg

import pyscf
import pyscf.lib
import pyscf.lo

from vayesta.core.util import *
from vayesta.core.symmetry import SymmetryIdentity
from vayesta.core.symmetry import SymmetryTranslation
import vayesta.core.ao2mo
import vayesta.core.ao2mo.helper

from vayesta.misc.cubefile import CubeFile
from vayesta.core.mpi import mpi
from vayesta.core.types import WaveFunction

# Get MPI rank of fragment
get_fragment_mpi_rank = lambda *args : args[0].mpi_rank


class Fragment:

    @dataclasses.dataclass
    class Options(OptionsBase):
        dmet_threshold: float = NotSet
        solver_options: dict = NotSet
        coupled_fragments: list = dataclasses.field(default_factory=list)
        # Symmetry
        sym_factor: float = 1.0
        wf_partition: str = NotSet  # ['first-occ', 'first-vir', 'democratic']
        store_eris: bool = NotSet   # If True, ERIs will be stored in Fragment._eris

    @dataclasses.dataclass
    class Results:
        fid: int = None             # Fragment ID
        converged: bool = None      # True, if solver reached convergence criterion or no convergence required (eg. MP2 solver)
        # --- Energies
        e_corr: float = None        # Fragment correlation energy contribution
        # --- Wave-function
        wf: WaveFunction = None     # WaveFunction object (MP2, CCSD,...)
        pwf: WaveFunction = None    # Fragment-projected wave function
        # --- Density-matrices
        #dm1: np.ndarray = None      # One-particle reduced density matrix (dm1[i,j] = <i^+ j>
        #dm2: np.ndarray = None      # Two-particle reduced density matrix (dm2[i,j,k,l] = <i^+ k^+ l j>)

        # OLD / Deprecated:
        @property
        def c0(self):
            return self.wf.c0
        @property
        def c1(self):
            return self.wf.c1
        @property
        def c2(self):
            return self.wf.c2
        @property
        def t1(self):
            return self.wf.t1
        @property
        def t2(self):
            return self.wf.t2
        @property
        def l1(self):
            return self.wf.l1
        @property
        def l2(self):
            return self.wf.l2
        @property
        def c1x(self):
            return self.pwf.c1
        @property
        def c2x(self):
            return self.pwf.c2
        @property
        def t1x(self):
            return self.pwf.t1
        @property
        def t2x(self):
            return self.pwf.t2
        @property
        def l1x(self):
            return self.pwf.l1
        @property
        def l2x(self):
            return self.pwf.l2

        #c0: float = None            # Reference determinant CI coefficient
        #c1: np.ndarray = None       # CI single coefficients
        #c2: np.ndarray = None       # CI double coefficients
        #t1: np.ndarray = None       # CC single amplitudes
        #t2: np.ndarray = None       # CC double amplitudes
        #l1: np.ndarray = None       # CC single Lambda-amplitudes
        #l2: np.ndarray = None       # CC double Lambda-amplitudes
        ## Fragment-projected amplitudes:
        #c1x: np.ndarray = None      # Fragment-projected CI single coefficients
        #c2x: np.ndarray = None      # Fragment-projected CI double coefficients
        #t1x: np.ndarray = None      # Fragment-projected CC single amplitudes
        #t2x: np.ndarray = None      # Fragment-projected CC double amplitudes
        #l1x: np.ndarray = None      # Fragment-projected CC single Lambda-amplitudes
        #l2x: np.ndarray = None      # Fragment-projected CC double Lambda-amplitudes

        #def convert_amp_c_to_t(self):
        #    self.t1 = self.c1/self.c0
        #    self.t2 = self.c2/self.c0 - einsum('ia,jb->ijab', self.t1, self.t1)
        #    return self.t1, self.t2

        def get_t1(self, default=None):
            if self.t1 is not None:
                return self.t1
            if self.c1 is not None:
                return self.c1 / self.c0
            return default

        def get_t2(self, default=None):
            if self.t2 is not None:
                return self.t2
            if self.c0 is not None and self.c1 is not None and self.c2 is not None:
                c1 = self.c1/self.c0
                return self.c2/self.c0 - einsum('ia,jb->ijab', c1, c1)
            return default

        def get_c1(self, intermed_norm=False, default=None):
            if self.c1 is not None:
                norm = 1/self.c0 if intermed_norm else 1
                return norm * self.c1
            if self.t1 is not None:
                if not intermed_norm:
                    raise ValueError("Cannot deduce C1 amplitudes from T1: normalization not known.")
                return self.t1
            return default

        def get_c2(self, intermed_norm=False, default=None):
            if self.c2 is not None:
                norm = 1/self.c0 if intermed_norm else 1
                return norm * self.c2
            if self.t1 is not None and self.t2 is not None:
                if not intermed_norm:
                    raise ValueError("Cannot deduce C2 amplitudes from T1,T2: normalization not known.")
                return self.t2 + einsum('ia,jb->ijab', self.t1, self.t1)
            return default

    class Exit(Exception):
        """Raise for controlled early exit."""
        pass

    @staticmethod
    def stack_mo(*mo_coeff):
        """
        Use stack_mo in parts of the code which are used both in RHF and UHF.
        Use hstack in parts of the code which are only used in RHF, but may be called
        from UHF per spin channel.
        """
        return hstack(*mo_coeff)

    def __init__(self, base, fid, name, c_frag, c_env, #fragment_type,
            atoms=None, aos=None,
            sym_parent=None, sym_op=None,
            mpi_rank=0,
            log=None, options=None, **kwargs):
        """Abstract base class for quantum embedding fragments.

        The fragment may keep track of associated atoms or atomic orbitals, using
        the `atoms` and `aos` attributes, respectively.

        Parameters
        ----------
        base : Embedding
            Quantum embedding method the fragment is part of.
        fid : int
            Fragment ID.
        name : str
            Name of fragment.
        c_frag : (nAO, nFrag) array
            Fragment orbital coefficients.
        c_env : (nAO, nEnv) array
            Environment (non-fragment) orbital coefficients.
        fragment_type : {'IAO', 'Lowdin-AO', 'AO'}
            Fragment orbital type.
        atoms : list or int, optional
            Associated atoms. Default: None
        aos : list or int, optional
            Associated atomic orbitals. Default: None
        sym_factor : float, optional
            Symmetry factor (number of symmetry equivalent fragments). Default: 1.0.
        sym_parent : Fragment, optional
            Symmetry related parent fragment. Default: None.
        sym_op : Callable, optional
            Symmetry operation on AO basis function, representing the symmetry to the `sym_parent` object. Default: None.
        log : logging.Logger
            Logger object. If None, the logger of the `base` object is used. Default: None.

        Attributes
        ----------
        mol
        mf
        size
        nelectron
        id_name
        boundary_cond
        log : logging.Logger
            Logger object.
        base : Embedding
            Quantum embedding method, the fragment is part of.
        id : int
            Unique fragment ID.
        name : str
            Name of framgnet.
        c_frag : (nAO, nFrag) array
            Fragment orbital coefficients.
        c_env : (nAO, nEnv) array
            Environment (non-fragment) orbital coefficients.
        fragment_type : {'IAO', 'Lowdin-AO', 'AO'}
            Fragment orbital type.
        sym_factor : float
            Symmetry factor (number of symmetry equivalent fragments).
        atoms : list
            Atoms in fragment.
        aos : list
            Atomic orbitals in fragment
        coupled_fragments : list
            List of fragments, the current fragment is coupled to.
        """
        self.log = log or base.log
        self.id = fid
        self.name = name

        # Options
        self.base = base
        if options is None:
            options = self.Options(**kwargs)
        else:
            options = options.replace(kwargs)
        options = options.replace(self.base.opts, select=NotSet)
        self.opts = options

        self.c_frag = c_frag
        self.c_env = c_env
        self.sym_factor = self.opts.sym_factor
        self.sym_parent = sym_parent
        self.sym_op = sym_op
        # For some embeddings, it may be necessary to keep track of any associated atoms or basis functions (AOs)
        # TODO: Is this still used?
        self.atoms = atoms
        self.aos = aos

        # MPI
        self.mpi_rank = mpi_rank

        # This set of orbitals is used in the projection to evaluate expectation value contributions
        # of the fragment. By default it is equal to `self.c_frag`.
        self.c_proj = self.c_frag

        # Initialize self.bath, self._cluster, self._results, self._eris
        self.reset()

        self.log.info("Creating %r", self)
        #self.log.info(break_into_lines(str(self.opts), newline='\n    '))

    def __repr__(self):
        #keys = ['id', 'name']
        #fmt = ('%s(' + len(keys)*'%s: %r, ')[:-2] + ')'
        #values = [self.__dict__[k] for k in keys]
        #return fmt % (self.__class__.__name__, *[x for y in zip(keys, values) for x in y])
        return '%s(id= %d, name= %s, mpi_rank= %d, n_frag= %d, n_elec= %.8f, sym_factor= %f)' % (
                self.__class__.__name__, self.id, self.name, self.mpi_rank,
                self.n_frag, self.nelectron, self.sym_factor)

    def __str__(self):
        return '%s %d: %s' % (self.__class__.__name__, self.id, self.name)

    def log_info(self):
        # Some output
        fmt = '  > %-24s     '
        self.log.info(fmt+'%d', "Fragment orbitals:", self.n_frag)
        self.log.info(fmt+'%f', "Symmetry factor:", self.sym_factor)
        self.log.info(fmt+'%.10f', "Number of electrons:", self.nelectron)
        if self.atoms is not None:
            self.log.info(fmt+'%r', "Associated atoms:", self.atoms)
        if self.aos is not None:
            self.log.info(fmt+'%r', "Associated AOs:", self.aos)

    @property
    def mol(self):
        return self.base.mol

    @property
    def mf(self):
        return self.base.mf

    @property
    def n_frag(self):
        """Number of fragment orbitals."""
        return self.c_frag.shape[-1]

    @property
    def nelectron(self):
        """Number of mean-field electrons."""
        sc = np.dot(self.base.get_ovlp(), self.c_frag)
        ne = einsum('ai,ab,bi->', sc, self.mf.make_rdm1(), sc)
        return ne

    def trimmed_name(self, length=10, add_dots=True):
        """Fragment name trimmed to a given maximum length."""
        if len(self.name) <= length:
            return self.name
        if add_dots:
            return self.name[:(length-3)] + "..."
        return self.name[:length]

    @property
    def id_name(self):
        """Use this whenever a unique name is needed (for example to open a separate file for each fragment)."""
        return "%s-%s" % (self.id, self.trimmed_name())

    @property
    def boundary_cond(self):
        return self.base.boundary_cond

    # --- Overlap matrices
    # --------------------

    def _csc_dot(self, c1, c2):
        ovlp = self.base.get_ovlp()
        return dot(c1.T, ovlp, c2)

    @cache
    def get_overlap(self, key):
        """Get overlap between cluster orbitals, fragment orbitals, or MOs.

        Examples:
        >>> s = self.get_overlap('cluster|mo')
        >>> s = self.get_overlap('cluster|frag')
        >>> s = self.get_overlap('mo[occ]|cluster[occ]')
        >>> s = self.get_overlap('mo[vir]|cluster[vir]')
        """
        if key.count('|') != 1:
            raise ValueError("key needs to have exactly one '|'")

        # Standardize key to reduce cache misses:
        key_mod = key.lower().replace(' ', '')
        if key_mod != key:
            return self.get_overlap(key_mod)

        def _get_coeff(key):
            if 'frag' in key:
                return self.c_frag
            if 'occ' in key:
                part = '_occ'
            elif 'vir' in key:
                part = '_vir'
            else:
                part = ''
            if 'mo' in key:
                return getattr(self.base, 'mo_coeff%s' % part)
            if 'cluster' in key:
                return getattr(self.cluster, 'c_active%s' % part)
            raise ValueError("Invalid key: '%s'")

        left, right = key.split('|')
        c_left = _get_coeff(left)
        c_right = _get_coeff(right)
        return self._csc_dot(c_left, c_right)

    @property
    def results(self):
        return self.get_symmetry_parent()._results
        #if self.sym_parent is not None:
        #    return self.sym_parent.results
        #return self._results

    @results.setter
    def results(self, value):
        if self.sym_parent is not None:
            raise RuntimeError("Cannot set attribute results in symmetry derived fragment.")
        self._results = value

    @property
    def cluster(self):
        if self.sym_parent is not None:
            return self.sym_parent.cluster.transform(self.sym_op)
        return self._cluster

    @cluster.setter
    def cluster(self, value):
        if self.sym_parent is not None:
            raise RuntimeError("Cannot set attribute cluster in symmetry derived fragment.")
        self._cluster = value

    def reset(self):
        self.log.debugv("Resetting %s", self)
        self.bath = None
        self._cluster = None
        self._eris = None
        self._results = self.Results(fid=self.id)
        self.get_overlap.cache_clear()

    def couple_to_fragment(self, frag):
        if frag is self:
            raise RuntimeError("Cannot couple fragment with itself.")
        self.log.debugv("Coupling %s with %s", self, frag)
        self.opts.coupled_fragments.append(frag)

    def couple_to_fragments(self, frags):
        for frag in frags:
            self.couple_to_fragment(frag)

    def get_fragment_mf_energy(self):
        """Calculate the part of the mean-field energy associated with the fragment.

        Does not include nuclear-nuclear repulsion!
        """
        px = self.get_fragment_projector(self.base.mo_coeff)
        hveff = dot(px, self.base.mo_coeff.T, 2*self.base.get_hcore()+self.base.get_veff(), self.base.mo_coeff)
        occ = (self.base.mo_occ > 0)
        e_mf = np.sum(np.diag(hveff)[occ])
        return e_mf

    def get_fragment_projector(self, coeff, c_proj=None, inverse=False):
        """Projector for one index of amplitudes local energy expression.

        Cost: N^2 if O(1) coeffs , N^3 if O(N) coeffs

        Parameters
        ----------
        coeff : ndarray, shape(n(AO), N)
            Occupied or virtual orbital coefficients.
        inverse : bool, optional
            Return 1-p instead. Default: False.

        Returns
        -------
        p : (n, n) array
            Projection matrix.
        """
        if c_proj is None: c_proj = self.c_proj
        r = dot(coeff.T, self.base.get_ovlp(), c_proj)
        p = np.dot(r, r.T)
        if inverse:
            p = (np.eye(p.shape[-1]) - p)
        return p

    def get_mo_occupation(self, *mo_coeff, dm1=None):
        """Get mean-field occupation numbers (diagonal of 1-RDM) of orbitals.

        Parameters
        ----------
        mo_coeff : ndarray, shape(N, M)
            Orbital coefficients.

        Returns
        -------
        occup : ndarray, shape(M)
            Occupation numbers of orbitals.
        """
        mo_coeff = hstack(*mo_coeff)        # Do NOT use self.stack_mo!
        if dm1 is None: dm1 = self.mf.make_rdm1()
        sc = np.dot(self.base.get_ovlp(), mo_coeff)
        occup = einsum('ai,ab,bi->i', sc, dm1, sc)
        return occup

    #def check_mo_occupation(self, expected, *mo_coeff, tol=None):
    #    if tol is None: tol = 2*self.opts.dmet_threshold
    #    occup = self.get_mo_occupation(*mo_coeff)
    #    if not np.allclose(occup, expected, atol=tol):
    #        raise RuntimeError("Incorrect occupation of orbitals (expected %f):\n%r" % (expected, occup))
    #    return occup

    def canonicalize_mo(self, *mo_coeff, fock=None, eigvals=False, sign_convention=True):
        """Diagonalize Fock matrix within subspace.

        Parameters
        ----------
        *mo_coeff : ndarrays
            Orbital coefficients.
        eigenvalues : ndarray
            Return MO energies of canonicalized orbitals.

        Returns
        -------
        mo_canon : ndarray
            Canonicalized orbital coefficients.
        rot : ndarray
            Rotation matrix: np.dot(mo_coeff, rot) = mo_canon.
        """
        if fock is None: fock = self.base.get_fock()
        mo_coeff = hstack(*mo_coeff)    # Called from UHF: do NOT use stack_mo!
        fock = dot(mo_coeff.T, fock, mo_coeff)
        mo_energy, rot = np.linalg.eigh(fock)
        self.log.debugv("Canonicalized MO energies:\n%r", mo_energy)
        mo_can = np.dot(mo_coeff, rot)
        if sign_convention:
            mo_can, signs = fix_orbital_sign(mo_can)
            rot = rot*signs[np.newaxis]
        assert np.allclose(np.dot(mo_coeff, rot), mo_can)
        assert np.allclose(np.dot(mo_can, rot.T), mo_coeff)
        if eigvals:
            return mo_can, rot, mo_energy
        return mo_can, rot

    def diagonalize_cluster_dm(self, *mo_coeff, dm1=None, norm=2, tol=1e-4):
        """Diagonalize cluster (fragment+bath) DM to get fully occupied and virtual orbitals.

        Parameters
        ----------
        *mo_coeff: array or list of arrays
            Orbital coefficients. If multiple are given, they will be stacked along their second dimension.
        dm1: array, optional
            Mean-field density matrix, used to separate occupied and virtual cluster orbitals.
            If None, `self.mf.make_rdm1()` is used. Default: None.
        tol: float, optional
            If set, check that all eigenvalues of the cluster DM are close
            to 0 or 2, with the tolerance given by tol. Default= 1e-4.

        Returns
        -------
        c_cluster_occ: (n(AO), n(occ cluster)) array
            Occupied cluster orbital coefficients.
        c_cluster_vir: (n(AO), n(vir cluster)) array
            Virtual cluster orbital coefficients.
        """
        if dm1 is None: dm1 = self.mf.make_rdm1()
        c_cluster = hstack(*mo_coeff)
        sc = np.dot(self.base.get_ovlp(), c_cluster)
        dm = dot(sc.T, dm1, sc)
        e, r = np.linalg.eigh(dm)
        if tol and not np.allclose(np.fmin(abs(e), abs(e-norm)), 0, atol=tol, rtol=0):
            raise RuntimeError("Eigenvalues of cluster-DM not all close to 0 or %d:\n%s" % (norm, e))
        e, r = e[::-1], r[:,::-1]
        c_cluster = np.dot(c_cluster, r)
        c_cluster = fix_orbital_sign(c_cluster)[0]
        nocc = np.count_nonzero(e >= (norm/2))
        c_cluster_occ, c_cluster_vir = np.hsplit(c_cluster, [nocc])

        return c_cluster_occ, c_cluster_vir

    def project_ref_orbitals(self, c_ref, c):
        """Project reference orbitals into available space in new geometry.

        The projected orbitals will be ordered according to their eigenvalues within the space.

        Parameters
        ----------
        c : ndarray
            Orbital coefficients.
        c_ref : ndarray
            Orbital coefficients of reference orbitals.
        """
        nref = c_ref.shape[-1]
        assert (nref > 0)
        assert (c.shape[-1] > 0)
        self.log.debug("Projecting %d reference orbitals into space of %d orbitals", nref, c.shape[-1])
        s = self.base.get_ovlp()
        # Diagonalize reference orbitals among themselves (due to change in overlap matrix)
        c_ref_orth = pyscf.lo.vec_lowdin(c_ref, s)
        assert (c_ref_orth.shape == c_ref.shape)
        # Diagonalize projector in space
        csc = np.linalg.multi_dot((c_ref_orth.T, s, c))
        p = np.dot(csc.T, csc)
        e, r = np.linalg.eigh(p)
        e, r = e[::-1], r[:,::-1]
        c = np.dot(c, r)

        return c, e

    # --- Symmetry
    # ============

    def add_tsymmetric_fragments(self, tvecs, charge_tol=1e-6):
        """

        Parameters
        ----------
        tvecs: array(3) of integers
            Each element represent the number of translation vector corresponding to the a0, a1, and a2 lattice vectors of the cell.
        charge_tol: float, optional
            Tolerance for the error of the mean-field density matrix between symmetry related fragments.
            If the largest absolute difference in the density-matrix is above this value,
            and exception will be raised. Default: 1e-6.

        Returns
        -------
        fragments: list
            List of T-symmetry related fragments. These will be automatically added to base.fragments and
            have the attributes `sym_parent` and `sym_op` set.
        """
        ovlp = self.base.get_ovlp()
        dm1 = self.mf.make_rdm1()

        fragments = []
        for i, (dx, dy, dz) in enumerate(itertools.product(range(tvecs[0]), range(tvecs[1]), range(tvecs[2]))):
            if i == 0: continue
            tvec = (dx/tvecs[0], dy/tvecs[1], dz/tvecs[2])
            sym_op = SymmetryTranslation(self.mol, tvec)

            if sym_op is None:
                self.log.error("No T-symmetric fragment found for translation (%d,%d,%d) of fragment %s", dx, dy, dz, self.name)
                continue
            # Name for translationally related fragments
            name = '%s_T(%d,%d,%d)' % (self.name, dx, dy, dz)
            # Translated coefficients
            c_frag_t = sym_op(self.c_frag)
            c_env_t = sym_op(self.c_env)
            # Check that translated fragment does not overlap with current fragment:
            fragovlp = abs(dot(self.c_frag.T, ovlp, c_frag_t)).max()
            if (fragovlp > 1e-8):
                self.log.critical("Translation (%d,%d,%d) of fragment %s not orthogonal to original fragment (overlap= %.3e)!",
                            dx, dy, dz, self.name, fragovlp)
                raise RuntimeError("Overlapping fragment spaces.")
            # Deprecated:
            if hasattr(self.base, 'add_fragment'):  # pragma: no cover
                frag = self.base.add_fragment(name, c_frag_t, c_env_t, options=self.opts,
                        sym_parent=self, sym_op=sym_op)
            else:
                frag_id = self.base.register.get_next_id()
                frag = self.base.Fragment(self.base, frag_id, name, c_frag_t, c_env_t, options=self.opts,
                        sym_parent=self, sym_op=sym_op, mpi_rank=self.mpi_rank)
                self.base.fragments.append(frag)

            # Check symmetry
            charge_err = self.get_tsymmetry_error(frag, dm1=dm1)
            if charge_err > charge_tol:
                self.log.critical("Mean-field DM not symmetric for translation (%d,%d,%d) of %s (charge error= %.3e)!",
                    dx, dy, dz, self.name, charge_err)
                raise RuntimeError("MF not symmetric under translation (%d,%d,%d)" % (dx, dy, dz))
            else:
                self.log.debugv("Mean-field DM symmetry error for translation (%d,%d,%d) of %s = %.3e",
                    dx, dy, dz, self.name, charge_err)

            fragments.append(frag)
        return fragments

    def make_tsymmetric_fragments(self, *args, **kwargs):  # pragma: no cover
        self.log.warning("make_tsymmetric_fragments is deprecated - use add_tsymmetric_fragments")
        return self.add_tsymmetric_fragments(*args, **kwargs)

    def get_symmetry_parent(self):
        if self.sym_parent is None:
            return self
        return self.sym_parent

    def get_symmetry_operation(self):
        if self.sym_parent is None:
            return SymmetryIdentity(self.mol)
        return self.sym_op

    def get_symmetry_children(self):
        return self.base.get_fragments(sym_parent=self)

    @property
    def n_symmetry_children(self):
        return len(self.get_symmetry_children())

    @property
    def symmetry_factor(self):
        return (self.n_symmetry_children+1)

    def get_tsymmetry_error(self, frag, dm1=None):
        """Get translational symmetry error between two fragments."""
        if dm1 is None: dm1 = self.mf.make_rdm1()
        ovlp = self.base.get_ovlp()
        # This fragment (x)
        cx = np.hstack((self.c_frag, self.c_env))
        dmx = dot(cx.T, ovlp, dm1, ovlp, cx)
        # Other fragment (y)
        cy = np.hstack((frag.c_frag, frag.c_env))
        dmy = dot(cy.T, ovlp, dm1, ovlp, cy)
        err = abs(dmx - dmy).max()
        return err

    #def check_mf_tsymmetry(self):
    #    """Check translational symmetry of the mean-field between fragment and its children."""
    #    ovlp = self.base.get_ovlp()
    #    sds = dot(ovlp, self.mf.make_rdm1(), ovlp)
    #    c0 = np.hstack((self.c_frag, self.c_env))
    #    dm0 = dot(c0.T, sds, c0)
    #    for frag in self.get_symmetry_children():
    #        c1 = np.hstack((frag.c_frag, frag.c_env))
    #        dm1 = dot(c1.T, sds, c1)
    #        err = abs(dm1 - dm0).max()
    #        if err > mf_tol:
    #            self.log.error("Mean-field not T-symmetric between %s and %s (error= %.3e)!",
    #                    self.name, frag.name, err)
    #            continue
    #        else:
    #            self.log.debugv("Mean-field T-symmetry error between %s and %s = %.3e", self.name, frag.name, err)

    # --- Results
    # ===========

    def get_fragment_mo_energy(self, c_active=None, fock=None):
        """Returns approximate MO energies, using the the diagonal of the Fock matrix.

        Parameters
        ----------
        c_active: array, optional
        fock: array, optional
        """
        if c_active is None: c_active = self.cluster.c_active
        if fock is None: fock = self.base.get_fock()
        mo_energy = einsum('ai,ab,bi->i', c_active, fock, c_active)
        return mo_energy

    @mpi.with_send(source=get_fragment_mpi_rank)
    def get_fragment_dmet_energy(self, dm1=None, dm2=None, h1e_eff=None, eris=None):
        """Get fragment contribution to whole system DMET energy from cluster DMs.

        After fragment summation, the nuclear-nuclear repulsion must be added to get the total energy!

        Parameters
        ----------
        dm1: array, optional
            Cluster one-electron reduced density-matrix in cluster basis. If `None`, `self.results.dm1` is used. Default: None.
        dm2: array, optional
            Cluster two-electron reduced density-matrix in cluster basis. If `None`, `self.results.dm2` is used. Default: None.
        eris: array, optional
            Cluster electron-repulsion integrals in cluster basis. If `None`, the ERIs are reevaluated. Default: None.

        Returns
        -------
        e_dmet: float
            Electronic fragment DMET energy.
        """
        assert (mpi.rank == self.mpi_rank)
        if dm1 is None: dm1 = self.results.dm1
        if dm2 is None: dm2 = self.results.dm2
        if dm1 is None: raise RuntimeError("DM1 not found for %s" % self)
        if dm2 is None: raise RuntimeError("DM2 not found for %s" % self)
        c_act = self.cluster.c_active
        t0 = timer()
        if eris is None:
            eris = self._eris
        if eris is None:
            with log_time(self.log.timing, "Time for AO->MO transformation: %s"):
                eris = self.base.get_eris_array(c_act)
        if not isinstance(eris, np.ndarray):
            self.log.debugv("Extracting ERI array from CCSD ERIs object.")
            eris = vayesta.core.ao2mo.helper.get_full_array(eris, c_act)

        # Get effective core potential
        if h1e_eff is None:
            # Use the original Hcore (without chemical potential modifications), but updated mf-potential!
            h1e_eff = self.base.get_hcore_for_energy() + self.base.get_veff_for_energy(with_exxdiv=False)/2
            h1e_eff = dot(c_act.T, h1e_eff, c_act)
            occ = np.s_[:self.cluster.nocc_active]
            v_act = einsum('iipq->pq', eris[occ,occ,:,:]) - einsum('iqpi->pq', eris[occ,:,:,occ])/2
            h1e_eff -= v_act

        p_frag = self.get_fragment_projector(c_act)
        # Check number of electrons
        ne = einsum('ix,ij,jx->', p_frag, dm1, p_frag)
        self.log.debug("Number of electrons for DMET energy in fragment %12s: %.8f", self, ne)

        # Evaluate energy
        e1b = einsum('xj,xi,ij->', h1e_eff, p_frag, dm1)
        #e1b = einsum('xj,xi,ij->', (h_core + h_eff), p_frag, dm1)/2
        e2b = einsum('xjkl,xi,ijkl->', eris, p_frag, dm2)/2
        e_dmet = self.opts.sym_factor*(e1b + e2b)
        self.log.debug("Fragment E(DMET)= %+16.8f Ha", e_dmet)
        self.log.timing("Time for DMET energy: %s", time_string(timer()-t0))
        return e_dmet

    # --- Counterpoise
    # ================

    def make_counterpoise_mol(self, rmax, nimages=1, unit='A', **kwargs):
        """Make molecule object for counterposise calculation.

        WARNING: This has only been tested for periodic systems so far!

        Parameters
        ----------
        rmax : float
            All atom centers within range `rmax` are added as ghost-atoms in the counterpoise correction.
        nimages : int, optional
            Number of neighboring unit cell in each spatial direction. Has no effect in open boundary
            calculations. Default: 5.
        unit : ['A', 'B']
            Unit for `rmax`, either Angstrom (`A`) or Bohr (`B`).
        **kwargs :
            Additional keyword arguments for returned PySCF Mole/Cell object.

        Returns
        -------
        mol_cp : pyscf.gto.Mole or pyscf.pbc.gto.Cell
            Mole or Cell object with periodic boundary conditions removed
            and with ghost atoms added depending on `rmax` and `nimages`.
        """
        if len(self.atoms) > 1:
            raise NotImplementedError()
        import vayesta.misc
        return vayesta.misc.counterpoise.make_mol(self.mol, self.atoms[1], rmax=rmax, nimages=nimages, unit=unit, **kwargs)

    # --- Orbital plotting
    # --------------------

    @mpi.with_send(source=get_fragment_mpi_rank)
    def pop_analysis(self, cluster=None, dm1=None, **kwargs):
        if cluster is None: cluster = self.cluster
        if dm1 is None: dm1 = self.results.dm1
        if dm1 is None: raise ValueError("DM1 not found for %s" % self)
        # Add frozen mean-field contribution:
        dm1 = cluster.add_frozen_rdm1(dm1)
        return self.base.pop_analysis(dm1, mo_coeff=cluster.coeff, **kwargs)

    def plot3d(self, filename, gridsize=(100, 100, 100), **kwargs):
        """Write cube density data of fragment orbitals to file."""
        nx, ny, nz = gridsize
        directory = os.path.dirname(filename)
        if directory:
            os.makedirs(directory, exist_ok=True)
        cube = CubeFile(self.mol, filename=filename, nx=nx, ny=ny, nz=nz, **kwargs)
        cube.add_orbital(self.c_frag)
        cube.write()
