# --- Standard library
import dataclasses
import itertools
import os.path
import typing
# --- External
import numpy as np
import pyscf
import pyscf.lo
# --- Internal
from vayesta.core.util import (OptionsBase, cache, deprecated, dot, einsum, energy_string, fix_orbital_sign, hstack,
                               log_time, time_string, timer, AbstractMethodError)
from vayesta.core import spinalg
from vayesta.core.types import Cluster
from vayesta.core.symmetry import SymmetryIdentity
from vayesta.core.symmetry import SymmetryTranslation
import vayesta.core.ao2mo
import vayesta.core.ao2mo.helper
from vayesta.core.types import WaveFunction
# Bath
from vayesta.core.bath import BNO_Threshold
from vayesta.core.bath import DMET_Bath
from vayesta.core.bath import EwDMET_Bath
from vayesta.core.bath import MP2_Bath
from vayesta.core.bath import Full_Bath
from vayesta.core.bath import R2_Bath
from vayesta.core.bath import RPA_Bath, RPAcorr_Bath

# Other
from vayesta.misc.cubefile import CubeFile
from vayesta.mpi import mpi
from vayesta.solver import get_solver_class, check_solver_config, ClusterHamiltonian
from vayesta.core.screening.screening_crpa import get_frag_W

# Get MPI rank of fragment
get_fragment_mpi_rank = lambda *args : args[0].mpi_rank


@dataclasses.dataclass
class Options(OptionsBase):
    # Inherited from Embedding
    # ------------------------
    # --- Bath options
    bath_options: dict = None
    # --- Solver options
    solver_options: dict = None
    # --- Other
    store_eris: bool = None     # If True, ERIs will be cached in Fragment.hamil
    dm_with_frozen: bool = None # TODO: is still used?
    screening: typing.Optional[str] = None
    match_cluster_fock: bool = None
    # Fragment specific
    # -----------------
    # Auxiliary fragments are treated before non-auxiliary fragments, but do not contribute to expectation values
    auxiliary: bool = False
    coupled_fragments: list = dataclasses.field(default_factory=list)
    sym_factor: float = 1.0


class Fragment:

    Options = Options

    @dataclasses.dataclass
    class Flags:
        # If multiple cluster exist for a given atom, the envelop fragment is
        # the one with the largest (super) cluster space. This is used, for example,
        # for the finite-bath and ICMP2 corrections
        is_envelop: bool = True
        is_secfrag: bool = False
        # Secondary fragment parameter
        bath_parent_fragment_id: typing.Optional[int] = None

    @dataclasses.dataclass
    class Results:
        fid: int = None             # Fragment ID
        converged: bool = None      # True, if solver reached convergence criterion or no convergence required (eg. MP2 solver)
        # --- Energies
        e_corr: float = None        # Fragment correlation energy contribution
        e_corr_rpa: float = None    # Fragment correlation energy contribution at the level of RPA.
        # --- Wave-function
        wf: WaveFunction = None     # WaveFunction object (MP2, CCSD,...)
        pwf: WaveFunction = None    # Fragment-projected wave function
        moms: tuple = None


    def __init__(self, base, fid, name, c_frag, c_env,
                 solver=None,
                 atoms=None, aos=None, active=True,
                 sym_parent=None, sym_op=None,
                 mpi_rank=0, flags=None,
                 log=None, **kwargs):
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
        self.base = base

        # Options
        self.opts = self.Options()                                  # Default options
        self.opts.update(**self.base.opts.asdict(deepcopy=True))    # Update with embedding class options
        self.opts.replace(**kwargs)                                 # Replace with keyword arguments

        # Flags
        self.flags = self.Flags(**(flags or {}))

        solver = solver or self.base.solver
        self.check_solver(solver)

        self.solver = solver
        self.c_frag = c_frag
        self.c_env = c_env
        self.sym_factor = self.opts.sym_factor
        self.sym_parent = sym_parent
        self.sym_op = sym_op
        # For some embeddings, it may be necessary to keep track of any associated atoms or basis functions (AOs)
        self.atoms = atoms
        # TODO: Is aos still used?
        self.aos = aos
        self.active = active

        # MPI
        self.mpi_rank = mpi_rank

        # This set of orbitals is used in the projection to evaluate expectation value contributions
        # of the fragment. By default it is equal to `self.c_frag`.
        self.c_proj = self.c_frag

        # Initialize self.bath, self._cluster, self._results, self.hamil
        self.reset()

        self.log.debugv("Creating %r", self)
        #self.log.info(break_into_lines(str(self.opts), newline='\n    '))

    def __repr__(self):
        if mpi:
            return '%s(id= %d, name= %s, mpi_rank= %d)' % (
                self.__class__.__name__, self.id, self.name, self.mpi_rank)
        return '%s(id= %d, name= %s)' % (self.__class__.__name__, self.id, self.name)

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
        return "%d-%s" % (self.id, self.trimmed_name())

    def change_options(self, **kwargs):
        self.opts.replace(**kwargs)

    @property
    def hamil(self):
        # Cache the hamiltonian object; note that caching or not of eris is handled inside the hamiltonian object.
        if self._hamil is None:
            self.hamil = self.get_frag_hamil()
        return self._hamil

    @hamil.setter
    def hamil(self, value):
        self._hamil = value

    # --- Overlap matrices
    # --------------------

    def _csc_dot(self, c1, c2, ovlp=True, transpose_left=True, transpose_right=False):
        if transpose_left:
            c1 = c1.T
        if transpose_right:
            c2 = c2.T
        if ovlp is True:
            ovlp = self.base.get_ovlp()
        if ovlp is None:
            return dot(c1, c2)
        return dot(c1, ovlp, c2)

    @cache
    def get_overlap(self, key):
        """Get overlap between cluster orbitals, fragment orbitals, or MOs.

        The return value is cached but not copied; do not modify the array in place without
        creating a copy!

        Examples:
        >>> s = self.get_overlap('cluster|mo')
        >>> s = self.get_overlap('cluster|frag')
        >>> s = self.get_overlap('mo[occ]|cluster[occ]')
        >>> s = self.get_overlap('mo[vir]|cluster[vir]')
        """
        if key.count('|') > 1:
            left, center, right = key.rsplit('|', maxsplit=2)
            overlap_left = self.get_overlap('|'.join((left, center)))
            overlap_right = self.get_overlap('|'.join((center, right)))
            return self._csc_dot(overlap_left, overlap_right, ovlp=None, transpose_left=False)

        # Standardize key to reduce cache misses:
        key_mod = key.lower().replace(' ', '')
        if key_mod != key:
            return self.get_overlap(key_mod)

        def _get_coeff(key):
            if 'frag' in key:
                return self.c_frag
            if 'proj' in key:
                return self.c_proj
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

    def get_coeff_env(self):
        if self.c_env is not None:
            return self.c_env
        return self.sym_op(self.sym_parent.get_coeff_env())

    @property
    def results(self):
        return self.get_symmetry_parent()._results

    @results.setter
    def results(self, value):
        if self.sym_parent is not None:
            raise RuntimeError("Cannot set attribute results in symmetry derived fragment.")
        self._results = value

    @property
    def cluster(self):
        if self.sym_parent is not None:
            return self.sym_parent.cluster.basis_transform(self.sym_op)
        return self._cluster

    @cluster.setter
    def cluster(self, value):
        if self.sym_parent is not None:
            raise RuntimeError("Cannot set attribute cluster in symmetry derived fragment.")
        self._cluster = value

    def reset(self, reset_bath=True, reset_cluster=True, reset_eris=True, reset_inactive=True):
        self.log.debugv("Resetting %s (reset_bath= %r, reset_cluster= %r, reset_eris= %r, reset_inactive= %r)",
                self, reset_bath, reset_cluster, reset_eris, reset_inactive)
        if not reset_inactive and not self.active:
            return
        if reset_bath:
            self._dmet_bath = None
            self._bath_factory_occ = None
            self._bath_factory_vir = None
        if reset_cluster:
            self._cluster = None
            self.get_overlap.cache_clear()
        if reset_eris:
            self.hamil = None
            self._seris_ov = None
        self._results = None

    def get_fragments_with_overlap(self, tol=1e-8, **kwargs):
        """Get list of fragments which overlap both in occupied and virtual space."""
        c_occ = self.get_overlap('mo[occ]|cluster[occ]')
        c_vir = self.get_overlap('mo[vir]|cluster[vir]')
        def svd(cx, cy):
            rxy = np.dot(cx.T, cy)
            return np.linalg.svd(rxy, compute_uv=False)
        frags = []
        for fx in self.base.get_fragments(**kwargs):
            if (fx.id == self.id):
                continue
            cx_occ = fx.get_overlap('mo[occ]|cluster[occ]')
            s_occ = svd(c_occ, cx_occ)
            if s_occ.max() < tol:
                continue
            cx_vir = fx.get_overlap('mo[vir]|cluster[vir]')
            s_vir = svd(c_vir, cx_vir)
            if s_vir.max() < tol:
                continue
            frags.append(fx)
        return frags

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

    @property
    def contributes(self):
        """True if fragment contributes to expectation values, else False."""
        return (self.active and not self.opts.auxiliary)

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
        mo_coeff = hstack(*mo_coeff)
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

        TODO: move to Embedding class

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
        mo_coeff = hstack(*mo_coeff)
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
        if tol and not np.allclose(np.fmin(abs(e), abs(e-norm)), 0, atol=2*tol, rtol=0):
            self.log.warn("Eigenvalues of cluster-DM not all close to 0 or %d:\n%s" % (norm, e))
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

    def copy(self, fid=None, name=None,  **kwargs):
        """Create copy of fragment, without adding it to the fragments list."""
        if fid is None:
            fid = self.base.register.get_next_id()
        name = name or ('%s(copy)' % self.name)
        kwargs_copy = self.opts.asdict().copy()
        kwargs_copy.update(kwargs)
        attrs = ['c_frag', 'c_env', 'solver', 'atoms', 'aos', 'active', 'sym_parent', 'sym_op', 'mpi_rank', 'log']
        for attr in attrs:
            if attr in kwargs_copy:
                continue
            kwargs_copy[attr] = getattr(self, attr)
        frag = type(self)(self.base, fid, name, **kwargs_copy)
        return frag

    def add_tsymmetric_fragments(self, tvecs, symtol=1e-6):
        """

        Parameters
        ----------
        tvecs: array(3) of integers
            Each element represent the number of translation vector corresponding to the a0, a1, and a2 lattice vectors of the cell.
        symtol: float, optional
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
            sym_op = SymmetryTranslation(self.base.symmetry, tvec)
            if sym_op is None:
                self.log.error("No T-symmetric fragment found for translation (%d,%d,%d) of fragment %s", dx, dy, dz, self.name)
                continue
            # Name for translationally related fragments
            name = '%s_T(%d,%d,%d)' % (self.name, dx, dy, dz)
            # Translated coefficients
            c_frag_t = sym_op(self.c_frag)
            c_env_t = None  # Avoid expensive symmetry operation on environment orbitals
            # Check that translated fragment does not overlap with current fragment:
            fragovlp = self._csc_dot(self.c_frag, c_frag_t, ovlp=ovlp)
            if self.base.spinsym == 'restricted':
                fragovlp = abs(fragovlp).max()
            elif self.base.spinsym == 'unrestricted':
                fragovlp = max(abs(fragovlp[0]).max(), abs(fragovlp[1]).max())
            if (fragovlp > 1e-8):
                self.log.critical("Translation (%d,%d,%d) of fragment %s not orthogonal to original fragment (overlap= %.3e)!",
                                  dx, dy, dz, self.name, fragovlp)
                raise RuntimeError("Overlapping fragment spaces.")

            # Add fragment
            frag_id = self.base.register.get_next_id()
            frag = self.base.Fragment(self.base, frag_id, name, c_frag_t, c_env_t,
                                      sym_parent=self, sym_op=sym_op, mpi_rank=self.mpi_rank,
                                      **self.opts.asdict())
            self.base.fragments.append(frag)
            # Check symmetry
            # (only for the primitive translations (1,0,0), (0,1,0), and (0,0,1) to reduce number of sym_op(c_env) calls)
            if (abs(dx)+abs(dy)+abs(dz) == 1):
                charge_err, spin_err = self.get_symmetry_error(frag, dm1=dm1)
                if max(charge_err, spin_err) > symtol:
                    self.log.critical("Mean-field DM1 not symmetric for translation (%d,%d,%d) of %s (errors: charge= %.3e, spin= %.3e)!",
                                      dx, dy, dz, self.name, charge_err, spin_err)
                    raise RuntimeError("MF not symmetric under translation (%d,%d,%d)" % (dx, dy, dz))
                else:
                    self.log.debugv("Mean-field DM symmetry error for translation (%d,%d,%d) of %s: charge= %.3e, spin= %.3e",
                                    dx, dy, dz, self.name, charge_err, spin_err)

            fragments.append(frag)
        return fragments

    def get_symmetry_parent(self):
        if self.sym_parent is None:
            return self
        return self.sym_parent.get_symmetry_parent()

    def get_symmetry_operation(self):
        if self.sym_parent is None:
            return SymmetryIdentity(self.base.symmetry)
        return self.sym_op

    def get_symmetry_generations(self, maxgen=None, **filters):
        if maxgen == 0:
            return []
        generations = []
        fragments = self.base.get_fragments(**filters)
        # Direct children:
        lastgen = self.base.get_fragments(fragments, sym_parent=self)
        generations.append(lastgen)
        # Children of children, etc:
        for gen in range(1, maxgen or 1000):
            newgen = []
            for fx in lastgen:
                newgen += self.base.get_fragments(fragments, sym_parent=fx)
            if not newgen:
                break
            generations.append(newgen)
            lastgen = newgen
        return generations

    def get_symmetry_children(self, maxgen=None, **filters):
        gens = self.get_symmetry_generations(maxgen, **filters)
        # Flatten list of lists:
        children = list(itertools.chain.from_iterable(gens))
        return children

    def get_symmetry_tree(self, maxgen=None, **filters):
        """Returns a recursive tree:

        [(x, [children of x]), (y, [children of y]), ...]
        """
        if maxgen is None:
            maxgen = 1000
        if maxgen == 0:
            return []
        # Get direct children:
        children = self.get_symmetry_children(maxgen=1, **filters)
        # Build tree recursively:
        tree = [(x, x.get_symmetry_tree(maxgen=maxgen-1, **filters)) for x in children]
        return tree

    def loop_symmetry_children(self, arrays=None, axes=None, symtree=None, maxgen=None, include_self=False):
        """Loop over all symmetry related fragments, including children of children, etc.

        Parameters
        ----------
        arrays : ndarray or list[ndarray], optional
            If arrays are passed, the symmetry operation of each symmetry related fragment will be
            applied to this array along the axis given in `axes`.
        axes : list[int], optional
            List of axes, along which the symmetry operation is applied for each element of `arrays`.
            If None, the first axis will be used.
        """

        def get_yield(fragment, arrays):
            if arrays is None or len(arrays) == 0:
                return fragment
            if len(arrays) == 1:
                return fragment, arrays[0]
            return fragment, arrays

        if include_self:
            yield get_yield(self, arrays)
        if maxgen == 0:
            return
        elif maxgen is None:
            maxgen = 1000
        if arrays is None:
            arrays = []
        if axes is None:
            axes = len(arrays)*[0]
        if symtree is None:
            symtree = self.get_symmetry_tree()
        for child, grandchildren in symtree:
            intermediates = [child.sym_op(arr, axis=axis) for (arr, axis) in zip(arrays, axes)]
            yield get_yield(child, intermediates)
            if grandchildren and maxgen > 1:
                yield from child.loop_symmetry_children(intermediates, axes=axes, symtree=grandchildren, maxgen=(maxgen-1))

    @property
    def n_symmetry_children(self):
        """Includes children of children, etc."""
        return len(self.get_symmetry_children())

    @property
    def symmetry_factor(self):
        """Includes children of children, etc."""
        return (self.n_symmetry_children+1)

    def get_symmetry_error(self, frag, dm1=None):
        """Get translational symmetry error between two fragments."""
        if dm1 is None:
            dm1 = self.mf.make_rdm1()
        ovlp = self.base.get_ovlp()
        # This fragment (x)
        cx = np.hstack((self.c_frag, self.get_coeff_env()))
        dmx = dot(cx.T, ovlp, dm1, ovlp, cx)
        # Other fragment (y)
        cy = np.hstack((frag.c_frag, frag.get_coeff_env()))
        dmy = dot(cy.T, ovlp, dm1, ovlp, cy)
        err = abs(dmx - dmy).max()
        return err, 0.0

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

    # Bath and cluster
    # ----------------

    def _get_bath_option(self, key, occtype):
        """Get bath-option, checking if a occupied-virtual specific option has been set."""
        opts = self.opts.bath_options
        opt = opts.get('%s_%s' % (key, occtype[:3]), None)
        if opt is not None:
            return opt
        return opts[key]

    def make_bath(self):

        # --- DMET bath
        dmet = DMET_Bath(self, dmet_threshold=self.opts.bath_options['dmet_threshold'])
        dmet.kernel()
        self._dmet_bath = dmet

        # --- Additional bath
        def get_bath(occtype):
            otype = occtype[:3]
            assert otype in ('occ', 'vir')
            btype = self._get_bath_option('bathtype', occtype)
            # DMET bath only
            if btype == 'dmet':
                return None
            # Full bath (for debugging)
            if btype == 'full':
                return Full_Bath(self, dmet_bath=dmet, occtype=occtype)
            # Energy-weighted DMET bath
            if btype == 'ewdmet':
                threshold = self._get_bath_option('threshold', occtype)
                max_order = self._get_bath_option('max_order', occtype)
                return EwDMET_Bath(self, dmet_bath=dmet, occtype=occtype, threshold=threshold, max_order=max_order)
            # Spatially close orbitals
            if btype == 'r2':
                return R2_Bath(self, dmet, occtype=occtype)
            # MP2 bath natural orbitals
            if btype == 'mp2':
                project_dmet_order = self._get_bath_option('project_dmet_order', occtype)
                project_dmet_mode = self._get_bath_option('project_dmet_mode', occtype)
                addbuffer = self._get_bath_option('addbuffer', occtype) and occtype == 'virtual'
                if addbuffer:
                    other = 'occ' if (otype == 'vir') else 'vir'
                    c_buffer = getattr(dmet, 'c_env_%s' % other)
                else:
                    c_buffer = None
                return MP2_Bath(self, dmet_bath=dmet, occtype=occtype, c_buffer=c_buffer,
                                project_dmet_order=project_dmet_order, project_dmet_mode=project_dmet_mode)
            if btype == 'rpacorr':
                project_dmet_order = self._get_bath_option('project_dmet_order', occtype)
                project_dmet_mode = self._get_bath_option('project_dmet_mode', occtype)
                return RPAcorr_Bath(self, dmet_bath=dmet, occtype=occtype, c_buffer=None,
                                project_dmet_order=project_dmet_order, project_dmet_mode=project_dmet_mode)
            if btype == 'rpa':
                project_dmet_order = self._get_bath_option('project_dmet_order', occtype)
                project_dmet_mode = self._get_bath_option('project_dmet_mode', occtype)
                return RPA_Bath(self, dmet_bath=dmet, occtype=occtype, c_buffer=None,
                                project_dmet_order=project_dmet_order, project_dmet_mode=project_dmet_mode)
            raise NotImplementedError('bathtype= %s' % btype)
        self._bath_factory_occ = get_bath(occtype='occupied')
        self._bath_factory_vir = get_bath(occtype='virtual')

    def make_cluster(self):

        def get_orbitals(occtype):
            factory = getattr(self, '_bath_factory_%s' % occtype[:3])
            print("!!", factory)
            btype = self._get_bath_option('bathtype', occtype)
            if btype == 'dmet':
                c_bath = None
                c_frozen = getattr(self._dmet_bath, 'c_env_%s' % occtype[:3])
            elif btype == 'full':
                c_bath, c_frozen = factory.get_bath()
            elif btype == 'r2':
                rcut = self._get_bath_option('rcut', occtype)
                unit = self._get_bath_option('unit', occtype)
                c_bath, c_frozen = factory.get_bath(rcut=rcut)
            elif btype == 'ewdmet':
                order = self._get_bath_option('order', occtype)
                c_bath, c_frozen = factory.get_bath(order)
            elif btype in ['mp2', 'rpa', 'rpacorr']:
                threshold = self._get_bath_option('threshold', occtype)
                truncation = self._get_bath_option('truncation', occtype)
                print("££", threshold, truncation)
                bno_threshold = BNO_Threshold(truncation, threshold)
                print("$$", factory, bno_threshold)
                c_bath, c_frozen = factory.get_bath(bno_threshold)
            else:
                raise ValueError
            return c_bath, c_frozen

        c_bath_occ, c_frozen_occ = get_orbitals('occupied')
        c_bath_vir, c_frozen_vir = get_orbitals('virtual')
        c_active_occ = spinalg.hstack_matrices(self._dmet_bath.c_cluster_occ, c_bath_occ)
        c_active_vir = spinalg.hstack_matrices(self._dmet_bath.c_cluster_vir, c_bath_vir)
        # Canonicalize orbitals
        if self._get_bath_option('canonicalize', 'occupied'):
            c_active_occ = self.canonicalize_mo(c_active_occ)[0]
        if self._get_bath_option('canonicalize', 'virtual'):
            c_active_vir = self.canonicalize_mo(c_active_vir)[0]
        cluster = Cluster.from_coeffs(c_active_occ, c_active_vir, c_frozen_occ, c_frozen_vir)

        # Check occupations
        def check_occup(mo_coeff, expected):
            occup = self.get_mo_occupation(mo_coeff)
            # RHF
            atol = self.opts.bath_options['occupation_tolerance']
            if np.ndim(occup[0]) == 0:
                assert np.allclose(occup, 2*expected, rtol=0, atol=2*atol)
            else:
                assert np.allclose(occup[0], expected, rtol=0, atol=atol)
                assert np.allclose(occup[1], expected, rtol=0, atol=atol)
        check_occup(cluster.c_total_occ, 1)
        check_occup(cluster.c_total_vir, 0)

        self.log.info('Orbitals for %s', self)
        self.log.info('-------------%s', len(str(self))*'-')
        self.log.info(cluster.repr_size().replace('%', '%%'))

        self.cluster = cluster
        return cluster

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
    def get_fragment_dmet_energy(self, dm1=None, dm2=None, h1e_eff=None, hamil=None, part_cumulant=True, approx_cumulant=True):
        """Get fragment contribution to whole system DMET energy from cluster DMs.

        After fragment summation, the nuclear-nuclear repulsion must be added to get the total energy!

        Parameters
        ----------
        dm1: array, optional
            Cluster one-electron reduced density-matrix in cluster basis. If `None`, `self.results.dm1` is used. Default: None.
        dm2: array, optional
            Cluster two-electron reduced density-matrix in cluster basis. If `None`, `self.results.dm2` is used. Default: None.
        hamil : ClusterHamiltonian object.
            Object representing cluster hamiltonian, possibly including cached ERIs.
        part_cumulant: bool, optional
            If True, the 2-DM cumulant will be partitioned to calculate the energy. If False,
            the full 2-DM will be partitioned, as it is done in most of the DMET literature.
            True is recommended, unless checking for agreement with literature results. Default: True.
        approx_cumulant: bool, optional
            If True, the approximate cumulant, containing (delta 1-DM)-squared terms, is partitioned,
            instead of the true cumulant, if `part_cumulant=True`. Default: True.

        Returns
        -------
        e_dmet: float
            Electronic fragment DMET energy.
        """
        assert (mpi.rank == self.mpi_rank)
        if dm1 is None: dm1 = self.results.dm1
        if dm1 is None: raise RuntimeError("DM1 not found for %s" % self)
        c_act = self.cluster.c_active
        t0 = timer()
        if hamil is None:
            hamil = self._hamil
        if hamil is None:
            hamil = self.get_frag_hamil()

        eris = hamil.get_eris_bare()

        if dm2 is None:
            dm2 = self.results.wf.make_rdm2(with_dm1=not part_cumulant, approx_cumulant=approx_cumulant)

        # Get effective core potential
        if h1e_eff is None:
            if part_cumulant:
                h1e_eff = dot(c_act.T, self.base.get_hcore_for_energy(), c_act)
            else:
                # Use the original Hcore (without chemical potential modifications), but updated mf-potential!
                h1e_eff = self.base.get_hcore_for_energy() + self.base.get_veff_for_energy(with_exxdiv=False)/2
                h1e_eff = dot(c_act.T, h1e_eff, c_act)
                occ = np.s_[:self.cluster.nocc_active]
                v_act = einsum('iipq->pq', eris[occ,occ,:,:]) - einsum('iqpi->pq', eris[occ,:,:,occ])/2
                h1e_eff -= v_act

        p_frag = self.get_fragment_projector(c_act)
        # Check number of electrons
        ne = einsum('ix,ij,jx->', p_frag, dm1, p_frag)
        self.log.debugv("Number of electrons for DMET energy in fragment %12s: %.8f", self, ne)

        # Evaluate energy
        e1b = einsum('xj,xi,ij->', h1e_eff, p_frag, dm1)
        e2b = einsum('xjkl,xi,ijkl->', eris, p_frag, dm2)/2

        self.log.debugv("E(DMET): E(1)= %s E(2)= %s", energy_string(e1b), energy_string(e2b))
        e_dmet = self.opts.sym_factor*(e1b + e2b)
        self.log.debugv("Fragment E(DMET)= %+16.8f Ha", e_dmet)
        self.log.timingv("Time for DMET energy: %s", time_string(timer()-t0))
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
        if len(self.atoms) != 1:
            raise NotImplementedError
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

    def check_solver(self, solver):
        is_uhf = np.ndim(self.base.mo_coeff[1]) == 2
        if self.opts.screening:
            is_eb = "crpa_full" in self.opts.screening
        else:
            is_eb = False
        check_solver_config(is_uhf, is_eb, solver, self.log)

    def get_solver(self, solver=None):
        if solver is None:
            solver = self.solver
        cl_ham = self.hamil
        solver_cls = get_solver_class(cl_ham, solver)
        solver_opts = self.get_solver_options(solver)
        cluster_solver = solver_cls(cl_ham, **solver_opts)
        if self.opts.screening is not None:
            cluster_solver.hamil.add_screening(self._seris_ov)
        return cluster_solver

    def get_local_rpa_correction(self, hamil=None):
        e_loc_rpa = None
        if self.base.opts.ext_rpa_correction:
            hamil = hamil or self.hamil
            cumulant = self.base.opts.ext_rpa_correction == "cumulant"
            if self.base.opts.ext_rpa_correction not in ["erpa", "cumulant"]:
                raise ValueError("Unknown external rpa correction %s specified.")
            e_loc_rpa = hamil.calc_loc_erpa(*self._seris_ov[1:], cumulant)
        return e_loc_rpa

    def get_frag_hamil(self):
        if self.opts.screening is not None:
            if "crpa_full" in self.opts.screening:
                self.bos_freqs, self.couplings = get_frag_W(self.mf, self, pcoupling=("pcoupled" in self.opts.screening),
                                                            only_ov_screened=("ov" in self.opts.screening),
                                                            log=self.log)
        # This detects based on fragment what kind of Hamiltonian is appropriate (restricted and/or EB).
        return ClusterHamiltonian(self, self.mf, self.log, screening=self.opts.screening,
                                  cache_eris=self.opts.store_eris, match_fock=self.opts.match_cluster_fock)

    def get_solver_options(self, *args, **kwargs):
        raise AbstractMethodError
