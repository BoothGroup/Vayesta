import dataclasses
import itertools
import copy

import numpy as np
import scipy
import scipy.linalg

import pyscf
import pyscf.lib
import pyscf.lo

from vayesta.core.util import *
from vayesta.core import helper, tsymmetry
import vayesta.core.ao2mo
import vayesta.core.ao2mo.helper

class QEmbeddingFragment:


    @dataclasses.dataclass
    class Options(OptionsBase):
        solver_options: dict = NotSet
        coupled_fragments: list = dataclasses.field(default_factory=list)
        # Symmetry
        sym_factor: float = 1.0
        wf_partition: str = NotSet  # ['first-occ', 'first-vir', 'democratic']

    @dataclasses.dataclass
    class Results:
        fid: int = None             # Fragment ID
        converged: bool = None      # True, if solver reached convergence criterion or no convergence required (eg. MP2 solver)
        e_corr: float = None        # Fragment correlation energy contribution
        e_dmet: float = None        # DMET energy contribution
        # --- Wave-function
        c0: float = None            # Reference determinant CI coefficient
        c1: np.ndarray = None       # CI single coefficients
        c2: np.ndarray = None       # CI double coefficients
        t1: np.ndarray = None       # CC single amplitudes
        t2: np.ndarray = None       # CC double amplitudes
        l1: np.ndarray = None       # CC single Lambda-amplitudes
        l2: np.ndarray = None       # CC double Lambda-amplitudes
        # Fragment-projected ("fp") amplitudes:
        t1_fp: np.ndarray = None    # Fragment-projected CC single amplitudes
        t2_fp: np.ndarray = None    # Fragment-projected CC double amplitudes
        l1_fp: np.ndarray = None    # Fragment-projected CC single Lambda-amplitudes
        l2_fp: np.ndarray = None    # Fragment-projected CC double Lambda-amplitudes
        # Cluster density-matrices
        dm1: np.ndarray = None      # One-particle reduced density matrix (dm1[i,j] = <0| i^+ j |0>
        dm2: np.ndarray = None      # Two-particle reduced density matrix (dm2[i,j,k,l] = <0| i^+ k^+ l j |0>)

        def convert_amp_c_to_t(self):
            self.t1 = self.c1/self.c0
            self.t2 = self.c2/self.c0 - einsum('ia,jb->ijab', self.t1, self.t1)
            return self.t1, self.t2

        def get_t1(self):
            if self.t1 is not None:
                return self.t1
            if self.c1 is not None:
                return self.c1 / self.c0
            return None

        def get_t2(self):
            if self.t2 is not None:
                return self.t2
            if self.c2 is not None:
                c1 = self.c1/self.c0
                return self.c2/self.c0 - einsum('ia,jb->ijab', c1, c1)
            return None

    @dataclasses.dataclass
    class Stash(StashBase):
        """Dataclass to stash intermediate results."""
        eris: 'typing.Any' = None

        def clear(self):
            self.eris = None

    class Exit(Exception):
        """Raise for controlled early exit."""
        pass

    @staticmethod
    def stack_mo(*mo_coeff):
        return np.hstack(mo_coeff)

    def __init__(self, base, fid, name, c_frag, c_env, #fragment_type,
            atoms=None, aos=None,
            sym_parent=None, sym_op=None,
            log=None, options=None, **kwargs):
        """Abstract base class for quantum embedding fragments.

        The fragment may keep track of associated atoms or atomic orbitals, using
        the `atoms` and `aos` attributes, respectively.

        Parameters
        ----------
        base : QEmbeddingMethod
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
        base : QEmbeddingMethod
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
        self.log.info("Initializing %s" % self)
        self.log.info("-------------%s" % (len(str(self))*"-"))

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
        #self.fragment_type = fragment_type
        self.sym_factor = self.opts.sym_factor
        self.sym_parent = sym_parent
        self.sym_op = sym_op
        # For some embeddings, it may be necessary to keep track of any associated atoms or basis functions (AOs)
        self.atoms = atoms
        self.aos = aos

        # This set of orbitals is used in the projection to evaluate expectation value contributions
        # of the fragment. By default it is equal to `self.c_frag`.
        self.c_proj = self.c_frag

        # Some output
        fmt = '  > %-24s     '
        #self.log.info(fmt+'%r', "Fragment type:", self.fragment_type)
        self.log.info(fmt+'%r', "Fragment orbitals:", self.n_frag)
        self.log.info(fmt+'%r', "Symmetry factor:", self.sym_factor)
        self.log.info(fmt+'%.10f', "Number of electrons:", self.nelectron)
        if self.atoms is not None:
            self.log.info(fmt+'%r', "Associated atoms:", self.atoms)
        if self.aos is not None:
            self.log.info(fmt+'%r', "Associated AOs:", self.aos)

        # Final cluster active orbitals
        self._c_active_occ = None
        self._c_active_vir = None
        # Final cluster frozen orbitals (Avoid storing these, as they scale as N^2 per cluster)
        self._c_frozen_occ = None
        self._c_frozen_vir = None
        # Final results
        self._results = None

        # Intermediates
        self.stash = self.Stash()


    def __repr__(self):
        keys = ['id', 'name', 'atoms', 'aos']
        fmt = ('%s(' + len(keys)*'%s: %r, ')[:-2] + ')'
        values = [self.__dict__[k] for k in keys]
        return fmt % (self.__class__.__name__, *[x for y in zip(keys, values) for x in y])

    def __str__(self):
        return '%s %d: %s' % (self.__class__.__name__, self.id, self.trimmed_name())

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
    def size(self):
        self.log.warning("fragment.size is deprecated!")
        return self.n_frag

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
        """Use this whenever a unique name is needed (for example to open a seperate file for each fragment)."""
        return "%s-%s" % (self.id, self.trimmed_name())

    @property
    def boundary_cond(self):
        return self.base.boundary_cond

    # --- Active orbitals

    @property
    def c_active(self):
        """Active orbital coefficients."""
        if self.c_active_occ is None:
            return None
        return self.stack_mo(self.c_active_occ, self.c_active_vir)

    @property
    def c_active_occ(self):
        """Active occupied orbital coefficients."""
        if self.sym_parent is None:
            return self._c_active_occ
        else:
            return self.sym_op(self.sym_parent.c_active_occ)

    @property
    def c_active_vir(self):
        """Active virtual orbital coefficients."""
        if self.sym_parent is None:
            return self._c_active_vir
        else:
            return self.sym_op(self.sym_parent.c_active_vir)

    @property
    def n_active(self):
        """Number of active orbitals."""
        return (self.n_active_occ + self.n_active_vir)

    @property
    def n_active_occ(self):
        """Number of active occupied orbitals."""
        return self.c_active_occ.shape[-1]

    @property
    def n_active_vir(self):
        """Number of active virtual orbitals."""
        return self.c_active_vir.shape[-1]

    # --- Frozen orbitals

    @property
    def c_frozen(self):
        """Frozen orbital coefficients."""
        if self.c_frozen_occ is None:
            return None
        return self.stack_mo(self.c_frozen_occ, self.c_frozen_vir)

    @property
    def c_frozen_occ(self):
        """Frozen occupied orbital coefficients."""
        if self.sym_parent is None:
            return self._c_frozen_occ
        else:
            return self.sym_op(self.sym_parent.c_frozen_occ)

    @property
    def c_frozen_vir(self):
        """Frozen virtual orbital coefficients."""
        if self.sym_parent is None:
            return self._c_frozen_vir
        else:
            return self.sym_op(self.sym_parent.c_frozen_vir)

    @property
    def n_frozen(self):
        """Number of frozen orbitals."""
        return (self.n_frozen_occ + self.n_frozen_vir)

    @property
    def n_frozen_occ(self):
        """Number of frozen occupied orbitals."""
        return self.c_frozen_occ.shape[-1]

    @property
    def n_frozen_vir(self):
        """Number of frozen virtual orbitals."""
        return self.c_frozen_vir.shape[-1]

    # --- All orbitals

    @property
    def mo_coeff(self):
        return self.stack_mo(self.c_frozen_occ, self.c_active_occ,
                             self.c_active_vir, self.c_frozen_vir)

    # Rotation matrices

    def get_rot_to_mf(self):
        """Get rotation matrices from occupied/virtual active space to MF orbitals."""
        ovlp = self.base.get_ovlp()
        r_occ = dot(self.c_active_occ.T, ovlp, self.base.mo_coeff_occ)
        r_vir = dot(self.c_active_vir.T, ovlp, self.base.mo_coeff_vir)
        return r_occ, r_vir

    def get_rot_to_fragment(self, fragment):
        """Get rotation matrices between occupied/virtual active space of this and another fragment."""
        ovlp = self.base.get_ovlp()
        r_occ = dot(self.c_active_occ.T, ovlp, fragment.c_active_occ)
        r_vir = dot(self.c_active_vir.T, ovlp, fragment.c_active_vir)
        return r_occ, r_vir

    @property
    def results(self):
        if self.sym_parent is None:
            return self._results
        else:
            return self.sym_parent.results

    def reset(self):
        self.log.debugv("Resetting fragment %s", self)
        self._c_active_occ = None
        self._c_active_vir = None
        self._c_frozen_occ = None
        self._c_frozen_vir = None
        self._results = None

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

    def get_fragment_projector(self, coeff, inverse=False):
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
        r = dot(coeff.T, self.base.get_ovlp(), self.c_proj)
        p = np.dot(r, r.T)
        if inverse:
            p = (np.eye(p.shape[-1]) - p)
        return p

    def get_mo_occupation(self, *mo_coeff):
        """Get mean-field occupation numbers (diagonal of 1-RDM) of orbitals.

        Parameters
        ----------
        mo_coeff : ndarray, shape(N, M)
            Orbital coefficients.

        Returns
        -------
        occ : ndarray, shape(M)
            Occupation numbers of orbitals.
        """
        mo_coeff = np.hstack(mo_coeff)
        sc = np.dot(self.base.get_ovlp(), mo_coeff)
        occ = einsum('ai,ab,bi->i', sc, self.mf.make_rdm1(), sc)
        return occ

    def loop_fragments(self, exclude_self=False):
        """Loop over all fragments of the base quantum embedding method."""
        for frag in self.base.fragments:
            if (exclude_self and frag is self):
                continue
            yield frag

    def canonicalize_mo(self, *mo_coeff, eigvals=False, sign_convention=True):
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
        mo_coeff = self.stack_mo(*mo_coeff)
        fock = dot(mo_coeff.T, self.base.get_fock(), mo_coeff)
        mo_energy, rot = np.linalg.eigh(fock)
        mo_can = np.dot(mo_coeff, rot)
        if sign_convention:
            mo_can, signs = helper.orbital_sign_convention(mo_can)
            rot = rot*signs[np.newaxis]
        assert np.allclose(np.dot(mo_coeff, rot), mo_can)
        assert np.allclose(np.dot(mo_can, rot.T), mo_coeff)
        if eigvals:
            return mo_can, rot, mo_energy
        return mo_can, rot

    def diagonalize_cluster_dm(self, *mo_coeff, dm1=None, tol=1e-4):
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
        c_cluster = np.hstack(mo_coeff)
        sc = np.dot(self.base.get_ovlp(), c_cluster)
        dm = dot(sc.T, dm1, sc)
        e, r = np.linalg.eigh(dm)
        if tol and not np.allclose(np.fmin(abs(e), abs(e-2)), 0, atol=tol, rtol=0):
            raise RuntimeError("Error while diagonalizing cluster DM: eigenvalues not all close to 0 or 2:\n%s" % e)
        e, r = e[::-1], r[:,::-1]
        c_cluster = np.dot(c_cluster, r)
        nocc = np.count_nonzero(e >= 1)
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

    # --- DMET
    # ========

    def make_dmet_bath(self, c_env, dm1=None, c_ref=None, nbath=None, tol=1e-5, verbose=True, reftol=0.8):
        """Calculate DMET bath, occupied environment and virtual environment orbitals.

        If c_ref is not None, complete DMET orbital space using active transformation of reference orbitals.

        TODO:
        * reftol should not be necessary - just determine how many DMET bath orbital N are missing
        from C_ref and take the N largest eigenvalues over the combined occupied and virtual
        eigenvalues.

        Parameters
        ----------
        c_env : (n(AO), n(env)) array
            MO-coefficients of environment orbitals.
        dm1 : (n(AO), n(AO)) array, optional
            Mean-field one-particle reduced density matrix in AO representation. If None, `self.mf.make_rdm1()` is used.
            Default: None.
        c_ref : ndarray, optional
            Reference DMET bath orbitals from previous calculation.
        nbath : int, optional
            Number of DMET bath orbitals. If set, the paramter `tol` is ignored. Default: None.
        tol : float, optional
            Tolerance for DMET orbitals in eigendecomposition of density-matrix. Default: 1e-5.
        reftol : float, optional
            Tolerance for DMET orbitals in projection of reference orbitals.

        Returns
        -------
        c_bath : (n(AO), n(bath)) array
            DMET bath orbitals.
        c_occenv : (n(AO), n(occ. env)) array
            Occupied environment orbitals.
        c_virenv : (n(AO), n(vir. env)) array
            Virtual environment orbitals.
        """

        # No environemnt -> no bath/environment orbitals
        if c_env.shape[-1] == 0:
            nao = c_env.shape[0]
            return np.zeros((nao, 0)), np.zeros((nao, 0)), np.zeros((nao, 0))

        # Divide by 2 to get eigenvalues in [0,1]
        sc = np.dot(self.base.get_ovlp(), c_env)
        if dm1 is None: dm1 = self.mf.make_rdm1()
        dm_env = np.linalg.multi_dot((sc.T, dm1, sc)) / 2
        try:
            eig, r = np.linalg.eigh(dm_env)
        except np.linalg.LinAlgError:
            eig, r = scipy.linalg.eigh(dm_env)
        # Sort: occ. env -> DMET bath -> vir. env
        eig, r = eig[::-1], r[:,::-1]
        if (eig.min() < -1e-9):
            self.log.warning("Min eigenvalue of env. DM = %.6e", eig.min())
        if ((eig.max()-1) > 1e-9):
            self.log.warning("Max eigenvalue of env. DM = %.6e", eig.max())
        c_env = np.dot(c_env, r)

        if nbath is not None:
            # Work out tolerance which leads to nbath bath orbitals. This overwrites `tol`.
            abseig = abs(eig[np.argsort(abs(eig-0.5))])
            low, up = abseig[nbath-1], abseig[nbath]
            if abs(low - up) < 1e-14:
                raise RuntimeError("Degeneracy in env. DM does not allow for clear identification of %d bath orbitals!\nabs(eig)= %r"
                        % (nbath, abseig[:nbath+5]))
            tol = (low + up)/2
            self.log.debugv("Tolerance for %3d bath orbitals= %.8g", nbath, tol)

        mask_bath = np.logical_and(eig >= tol, eig <= 1-tol)
        mask_occenv = (eig > 1-tol)
        mask_virenv = (eig < tol)
        nbath = sum(mask_bath)

        noccenv = sum(mask_occenv)
        nvirenv = sum(mask_virenv)
        self.log.info("DMET bath:  n(Bath)= %4d  n(occ-Env)= %4d  n(vir-Env)= %4d", nbath, noccenv, nvirenv)
        assert (nbath + noccenv + nvirenv == c_env.shape[-1])
        c_bath = c_env[:,mask_bath].copy()
        c_occenv = c_env[:,mask_occenv].copy()
        c_virenv = c_env[:,mask_virenv].copy()

        if verbose:
            # Orbitals in [print_tol, 1-print_tol] will be printed (even if they don't fall in the DMET tol range)
            print_tol = 1e-10
            # DMET bath orbitals with eigenvalue in [strong_tol, 1-strong_tol] are printed as strongly entangled
            strong_tol = 0.1
            limits = [print_tol, tol, strong_tol, 1-strong_tol, 1-tol, 1-print_tol]
            if np.any(np.logical_and(eig > limits[0], eig <= limits[-1])):
                names = [
                        "Unentangled vir. env. orbital",
                        "Weakly-entangled vir. bath orbital",
                        "Strongly-entangled bath orbital",
                        "Weakly-entangled occ. bath orbital",
                        "Unentangled occ. env. orbital",
                        ]
                self.log.info("Non-(0 or 1) eigenvalues (n) of environment DM:")
                for i, e in enumerate(eig):
                    name = None
                    for j, llim in enumerate(limits[:-1]):
                        ulim = limits[j+1]
                        if (llim < e and e <= ulim):
                            name = names[j]
                            break
                    if name:
                        self.log.info("  > %-34s  n= %12.6g  1-n= %12.6g", name, e, 1-e)

            # DMET bath analysis
            self.log.info("DMET bath character:")
            for i in range(c_bath.shape[-1]):
                ovlp = einsum('a,b,ba->a', c_bath[:,i], c_bath[:,i], self.base.get_ovlp())
                sort = np.argsort(-ovlp)
                ovlp = ovlp[sort]
                n = np.amin((len(ovlp), 6))     # Get the six largest overlaps
                labels = np.asarray(self.mol.ao_labels())[sort][:n]
                lines = [('%s= %.5f' % (labels[i].strip(), ovlp[i])) for i in range(n)]
                self.log.info("  > %2d:  %s", i+1, '  '.join(lines))

        # Calculate entanglement entropy
        entropy = np.sum(eig * (1-eig))
        entropy_bath = np.sum(eig[mask_bath] * (1-eig[mask_bath]))
        self.log.info("Entanglement entropy: total= %.6e  bath= %.6e  captured=  %.2f %%",
                entropy, entropy_bath, 100.0*entropy_bath/entropy)

        # Complete DMET orbital space using reference orbitals
        # NOT MAINTAINED!
        if c_ref is not None:
            nref = c_ref.shape[-1]
            self.log.debug("%d reference DMET orbitals given.", nref)
            nmissing = nref - nbath

            # DEBUG
            _, eig = self.project_ref_orbitals(c_ref, c_bath)
            self.log.debug("Eigenvalues of reference orbitals projected into DMET bath:\n%r", eig)

            if nmissing == 0:
                self.log.debug("Number of DMET orbitals equal to reference.")
            elif nmissing > 0:
                # Perform the projection separately for occupied and virtual environment space
                # Otherwise, it is not guaranteed that the additional bath orbitals are
                # fully (or very close to fully) occupied or virtual.
                # --- Occupied
                C_occenv, eig = self.project_ref_orbitals(c_ref, c_occenv)
                mask_occref = eig >= reftol
                mask_occenv = eig < reftol
                self.log.debug("Eigenvalues of projected occupied reference: %s", eig[mask_occref])
                if np.any(mask_occenv):
                    self.log.debug("Largest remaining: %s", max(eig[mask_occenv]))
                # --- Virtual
                c_virenv, eig = self.project_ref_orbitals(c_ref, c_virenv)
                mask_virref = eig >= reftol
                mask_virenv = eig < reftol
                self.log.debug("Eigenvalues of projected virtual reference: %s", eig[mask_virref])
                if np.any(mask_virenv):
                    self.log.debug("Largest remaining: %s", max(eig[mask_virenv]))
                # -- Update coefficient matrices
                c_bath = np.hstack((c_bath, c_occenv[:,mask_occref], c_virenv[:,mask_virref]))
                c_occenv = c_occenv[:,mask_occenv].copy()
                c_virenv = c_virenv[:,mask_virenv].copy()
                nbath = C_bath.shape[-1]
                self.log.debug("New number of occupied environment orbitals: %d", c_occenv.shape[-1])
                self.log.debug("New number of virtual environment orbitals: %d", c_virenv.shape[-1])
                if nbath != nref:
                    err = "Number of DMET bath orbitals=%d not equal to reference=%d" % (nbath, nref)
                    self.log.critical(err)
                    raise RuntimeError(err)
            else:
                err = "More DMET bath orbitals found than in reference!"
                self.log.critical(err)
                raise RuntimeError(err)

        return c_bath, c_occenv, c_virenv


    # Amplitude projection
    # --------------------

    def project_amplitude_to_fragment(self, c, c_occ=None, c_vir=None, partition=None, symmetrize=True):
        """Get fragment contribution of CI coefficients or CC amplitudes.

        Parameters
        ----------
        c: (n(occ), n(vir)) or (n(occ), n(occ), n(vir), n(vir)) array
            CI coefficients or CC amplitudes.
        c_occ: (n(AO), n(MO)) array, optional
            Occupied MO coefficients. If `None`, `self.c_active_occ` is used. Default: `None`.
        c_vir: (n(AO), n(MO)) array, optional
            Virtual MO coefficients. If `None`, `self.c_active_vir` is used. Default: `None`.
        partition: ['first-occ', 'first-vir', 'democractic'], optional
            Partitioning scheme of amplitudes. Default: 'first-occ'.
        symmetrize: bool, optional
            Symmetrize C2/T2 amplitudes such that they obey "(ijab) = (jiba)" symmetry. Default: True.

        Returns
        -------
        pc: array
            Projected CI coefficients or CC amplitudes.
        """

        if partition is None: partition = self.opts.wf_partition

        if np.ndim(c) not in (2, 4):
            raise NotImplementedError("C.shape= %r" % c.shape)
        partition = partition.lower()
        if partition not in ('first-occ', 'occ-2', 'first-vir', 'democratic'):
            raise ValueError("Unknown partitioning of amplitudes: %r" % partition)

        if c_occ is None: c_occ = self.c_active_occ
        if c_vir is None: c_vir = self.c_active_vir

        # Projectors into fragment occupied and virtual space
        if partition in ('first-occ', 'occ-2', 'democratic'):
            fo = self.get_fragment_projector(c_occ)
        if partition in ('first-vir', 'democratic'):
            fv = self.get_fragment_projector(c_vir)
        # Inverse projectors
        if partition == 'democratic':
            ro = np.eye(fo.shape[-1]) - fo
            rv = np.eye(fv.shape[-1]) - fv

        if np.ndim(c) == 2:
            if partition in ('first-occ', 'occ-2'):
                pc = einsum('xi,ia->xa', fo, c)
            elif partition == 'first-vir':
                pc = einsum('ia,xa->ix', c, fv)
            elif partition == 'democratic':
                pc = einsum('xi,ia,ya->xy', fo, c, fv)
                pc += einsum('xi,ia,ya->xy', fo, c, rv) / 2.0
                pc += einsum('xi,ia,ya->xy', ro, c, fv) / 2.0
            return pc

        if partition == 'first-occ':
            pc = einsum('xi,ijab->xjab', fo, c)
        elif partition == 'occ-2':
            pc = einsum('xj,ijab->ixab', fo, c)
        elif partition == 'first-vir':
            pc = einsum('ijab,xa->ijxb', c, fv)
        elif partition == 'democratic':

            def project(p1, p2, p3, p4):
                pc = einsum('xi,yj,ijab,za,wb->xyzw', p1, p2, c, p3, p4)
                return pc

            # Factors of 2 due to ij,ab <-> ji,ba symmetry
            # Denominators 1/N due to element being shared between N clusters

            # Quadruple F
            # ===========
            # This is fully included
            pc = project(fo, fo, fv, fv)
            # Triple F
            # ========
            # This is fully included
            pc += 2*project(fo, fo, fv, rv)
            pc += 2*project(fo, ro, fv, fv)
            # Double F
            # ========
            # P(FFrr) [This wrongly includes: 1x P(FFaa), instead of 0.5x - correction below]
            pc +=   project(fo, fo, rv, rv)
            pc += 2*project(fo, ro, fv, rv)
            pc += 2*project(fo, ro, rv, fv)
            pc +=   project(ro, ro, fv, fv)
            # Single F
            # ========
            # P(Frrr) [This wrongly includes: P(Faar) (where r could be a) - correction below]
            pc += 2*project(fo, ro, rv, rv) / 4.0
            pc += 2*project(ro, ro, fv, rv) / 4.0

            # Corrections
            # ===========
            # Loop over all other clusters x
            for x in self.loop_fragments(exclude_self=True):

                xo = x.get_fragment_projector(c_occ)
                xv = x.get_fragment_projector(c_vir)

                # Double correction
                # -----------------
                # Correct for wrong inclusion of P(FFaa)
                # The case P(FFaa) was included with prefactor of 1 instead of 1/2
                # We thus need to only correct by "-1/2"
                pc -=   project(fo, fo, xv, xv) / 2.0
                pc -= 2*project(fo, xo, fv, xv) / 2.0
                pc -= 2*project(fo, xo, xv, fv) / 2.0
                pc -=   project(xo, xo, fv, fv) / 2.0

                # Single correction
                # -----------------
                # Correct for wrong inclusion of P(Faar)
                # This corrects the case P(Faab) but overcorrects P(Faaa)!
                pc -= 2*project(fo, xo, xv, rv) / 4.0
                pc -= 2*project(fo, xo, rv, xv) / 4.0 # If r == x this is the same as above -> overcorrection
                pc -= 2*project(fo, ro, xv, xv) / 4.0 # overcorrection
                pc -= 2*project(xo, xo, fv, rv) / 4.0
                pc -= 2*project(xo, ro, fv, xv) / 4.0 # overcorrection
                pc -= 2*project(ro, xo, fv, xv) / 4.0 # overcorrection

                # Correct overcorrection
                # The additional factor of 2 comes from how often the term was wrongly included above
                pc += 2*2*project(fo, xo, xv, xv) / 4.0
                pc += 2*2*project(xo, xo, fv, xv) / 4.0

        # Note that the energy should be invariant to symmetrization
        if symmetrize:
            pc = (pc + pc.transpose(1,0,3,2)) / 2

        return pc

    # --- Symmetry
    # ============

    def make_tsymmetric_fragments(self, tvecs, unit='Ang', mf_tol=1e-6):
        """

        Parameters
        ----------
        tvecs: (3,3) float array or (3,) integer array
            Translational symmetry vectors. If an array with shape (3,3) is passed, each row represents
            a translation vector in cartesian coordinates, in units defined by the parameter `unit`.
            If an array with shape (3,) is passed, each element represent the number of
            translation vector corresponding to the a0, a1, and a2 lattice vectors of the cell.
        unit: ['Ang', 'Bohr'], optional
            Units of translation vectors. Only used if a (3, 3) array is passed. Default: 'Ang'.
        mf_tol: float, optional
            Tolerance for the error of the mean-field density matrix between symmetry related fragments.
            If the largest absolute difference in the density-matrix is above `mf_tol`,
            the translated fragment is not considered as symmetry related. Default: 1e-6.

        Returns
        -------
        fragments: list
            List of T-symmetry related fragments. These will be automatically added to base.fragments and
            have the attributes `sym_parent` and `sym_op` set.
        """
        #if self.boundary_cond == 'open': return []

        mesh, tvecs = tsymmetry.get_mesh_tvecs(self.mol, tvecs, unit)
        self.log.debugv("nx= %d ny= %d nz= %d", *mesh)
        self.log.debugv("tvecs=\n%r", tvecs)

        ovlp = self.base.get_ovlp()
        sds = np.linalg.multi_dot((ovlp, self.mf.make_rdm1(), ovlp))
        c_all = np.hstack((self.c_frag, self.c_env))
        dm0 = np.linalg.multi_dot((c_all.T, sds, c_all))

        fragments = []
        # last index is fastest looping - change x first, then y, then z:
        for dz, dy, dx in itertools.product(range(mesh[2]), range(mesh[1]), range(mesh[0])):
            if abs(dx) + abs(dy) + abs(dz) == 0:
                continue
            t = dx*tvecs[0] + dy*tvecs[1] + dz*tvecs[2]
            reorder, inverse, phases = tsymmetry.reorder_aos(self.mol, t, unit='Bohr')
            self.log.debugv("reorder=\n%r", reorder)
            self.log.debugv("inverse=\n%r", inverse)
            self.log.debugv("phases=\n%r", phases)
            if reorder is None:
                self.log.error("No T-symmetric fragment found for translation [%d %d %d] of fragment %s", dx, dy, dz, self.name)
                continue
            name = '%s.t%d.%d.%d' % (self.name, dx, dy, dz)
            c_frag = self.c_frag[reorder]*phases[:,None]
            c_env = self.c_env[reorder]*phases[:,None]
            # Check that translated fragment does not overlap with current fragment:
            ovlp = np.linalg.norm(np.linalg.multi_dot((self.c_frag.T, self.base.get_ovlp(), c_frag)))
            if ovlp > 1e-10:
                self.log.error("Translation [%d %d %d] of fragment %s not orthogonal to original fragment (overlap= %.3e)!",
                            dx, dy, dz, self.name, ovlp)
            # Check that MF solution has lattice periodicity:
            c_all = np.hstack((c_frag, c_env))
            dm = np.linalg.multi_dot((c_all.T, sds, c_all))
            err = abs(dm - dm0).max()
            if err > mf_tol:
                self.log.error("Mean-field not T-symmetric for translation [%d %d %d] of fragment space %s (error= %.3e)!",
                        dx, dy, dz, self.name, err)
                continue
            else:
                self.log.debugv("Mean-field T-symmetry error for translation [%d %d %d]= %.3e", dx, dy, dz, err)

            sym_op = tsymmetry.make_sym_op(reorder, phases)
            # Deprecated:
            if hasattr(self.base, 'add_fragment'):
                frag = self.base.add_fragment(name, c_frag, c_env, #fragment_type=self.fragment_type,
                        options=self.opts,
                        sym_parent=self, sym_op=sym_op)
            else:
                fid = self.base.fragmentation.get_next_fid()
                frag = self.base.Fragment(self.base, fid, name, c_frag, c_env, #self.fragment_type,
                        options=self.opts,
                        sym_parent=self, sym_op=sym_op)
                self.base.fragments.append(frag)
            fragments.append(frag)

        return fragments

    # --- Results
    # ===========

    def get_fragment_mo_energy(self, c_active=None, fock=None):
        """Returns approximate MO energies, using the the diagonal of the Fock matrix.

        Parameters
        ----------
        c_active: array, optional
        fock: array, optional
        """
        if c_active is None: c_active = self.c_active
        if fock is None: fock = self.base.get_fock()
        mo_energy = einsum('ai,ab,bi->i', c_active, fock, c_active)
        return mo_energy

    def get_fragment_dmet_energy(self, dm1=None, dm2=None, h1e_eff=None, eris=None):
        """Get fragment contribution to whole system DMET energy.

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
        if dm1 is None: dm1 = self.results.dm1
        if dm2 is None: dm2 = self.results.dm2
        if dm1 is None: raise RuntimeError("DM1 not found for %s" % self)
        if dm2 is None: raise RuntimeError("DM2 not found for %s" % self)
        c_act = self.c_active
        t0 = timer()
        if eris is None:
            eris = self.base.get_eris_array(c_act)
        elif not isinstance(eris, np.ndarray):
            self.log.debugv("Extracting ERI array from CCSD ERIs object.")
            eris = vayesta.core.ao2mo.helper.get_full_array(eris, c_act)

        # Get effective core potential
        if h1e_eff is None:
            occ = np.s_[:self.n_active_occ]
            # Use the original Hcore (without chemical potential modifications), but updated mf-potential!
            h1e_eff = self.base.get_hcore_orig() + self.base.get_veff(with_exxdiv=False)/2
            h1e_eff = dot(c_act.T, h1e_eff, c_act)
            v_act = einsum('iipq->pq', eris[occ,occ,:,:]) - einsum('iqpi->pq', eris[occ,:,:,occ])/2
            h1e_eff -= v_act

        p_frag = self.get_fragment_projector(c_act)
        # Check number of electrons
        ne = einsum('ix,ij,jx->', p_frag, dm1, p_frag)
        self.log.info("Number of local electrons for DMET energy: %.8f", ne)

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
