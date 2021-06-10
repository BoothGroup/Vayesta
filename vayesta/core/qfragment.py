import numpy as np

import pyscf
import pyscf.lib

from vayesta.core.util import *


class QEmbeddingFragment:

    def __init__(self, base, fid, name, c_frag, c_env, fragment_type, atoms=None, aos=None, sym_factor=1.0, log=None):
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
        """
        self.log = log or base.log
        self.id = fid
        self.name = name
        self.log.info("Initializing %s" % self)
        self.log.info("*************%s" % (len(str(self))*"*"))

        self.base = base
        self.c_frag = c_frag
        self.c_env = c_env
        self.fragment_type = fragment_type
        self.sym_factor = sym_factor
        # For some embeddings, it may be necessary to keep track of any associated atoms or basis functions (AOs)
        self.atoms = atoms
        self.aos = aos

        # Some output
        fmt = '  * %-24s %r'
        self.log.info(fmt, "Fragment type:", self.fragment_type)
        self.log.info(fmt, "Fragment orbitals:", self.size)
        self.log.info(fmt, "Symmetry factor:", self.sym_factor)
        self.log.info(fmt, "Number of electrons:", self.nelectron)
        if self.atoms is not None:
            self.log.info(fmt, "Associated atoms:", self.atoms)
        if self.aos is not None:
            self.log.info(fmt, "Associated AOs:", self.aos)


    def __repr__(self):
        keys = ['id', 'name', 'fragment_type', 'sym_factor', 'atoms', 'aos']
        fmt = ('%s(' + len(keys)*'%s: %r, ')[:-2] + ')'
        values = [self.__dict__[k] for k in keys]
        return fmt % (self.__class__.__name__, *[x for y in zip(keys, values) for x in y])

    def __str__(self):
        return '%s %d: %s' % (self.__class__.__name__, self.id, self.name)

    @property
    def mol(self):
        return self.base.mol

    @property
    def mf(self):
        return self.base.mf

    @property
    def size(self):
        """Number of fragment orbitals."""
        return self.c_frag.shape[-1]

    @property
    def nelectron(self):
        """Number of mean-field electrons."""
        sc = np.dot(self.base.get_ovlp(), self.c_frag)
        ne = np.einsum("ai,ab,bi->", sc, self.mf.make_rdm1(), sc)
        return ne

    def trimmed_name(self, length=10):
        """Fragment name trimmed to a given maximum length."""
        if len(self.name) <= length:
            return self.name
        return self.name[:(length-3)] + "..."

    @property
    def id_name(self):
        """Use this whenever a unique name is needed (for example to open a seperate file for each fragment)."""
        return "%s-%s" % (self.id, self.trimmed_name())

    @property
    def boundary_cond(self):
        return self.base.boundary_cond


    def get_fragment_mf_energy(self):
        """Calculate the part of the mean-field energy associated with the fragment.

        Does not include nuclear-nuclear repulsion!
        """
        h1e = np.linalg.multi_dot((self.base.mo_coeff.T, self.mf.get_hcore(), self.base.mo_coeff))
        h1e += np.diag(self.base.mo_energy)
        p = self.get_fragment_projector(self.base.mo_coeff)
        h1e = np.dot(p, h1e)
        e_mf = np.sum(np.diag(h1e)[self.base.mo_occ>0])
        return e_mf


    def get_fragment_projector(self, coeff, ao_ptype='right', inverse=False):
        """Projector for one index of amplitudes local energy expression.

        Parameters
        ----------
        coeff : ndarray, shape(nAO, n)
            Occupied or virtual orbital coefficients.
        ao_ptype : {'right', 'left', 'symmetric'}, optional
            Defines were the projector is restricted to AO indices. Is only used
            of `self.fragment_type == 'AO'`. Default: 'right'.
        inverse : bool, optional
            Return 1-p instead. Default: False.

        Returns
        -------
        p : (n, n) array
            Projection matrix.
        """

        if self.fragment_type.upper() in ('IAO', 'LOWDIN-AO'):
            r = np.linalg.multi_dot((coeff.T, self.base.get_ovlp(), self.c_frag))
            p = np.dot(r, r.T)
        if self.fragment_type.upper() == 'AO':
            if self.aos is None:
                raise ValueError("Cannot obtain local projector for fragment_type 'AO', if attribute `aos` is not set.")
            if ao_ptype == 'right':
                p = np.linalg.multi_dot((coeff.T, self.base.get_ovlp()[:,self.aos], self.c_frag[self.aos]))
            elif ao_ptype == 'right':
                p = np.linalg.multi_dot((coeff[self.aos].T, self.base.get_ovlp()[self.aos], self.c_frag))
            elif ao_ptype == 'symmetric':
                # Does this even make sense?
                shalf = scipy.linalg.fractional_matrix_power(self.get_ovlp, 0.5)
                assert np.allclose(s.half.imag, 0)
                shalf = shalf.real
                p = np.linalg.multi_dot((C.T, shalf[:,self.aos], s[self.aos], C))
        if inverse:
            p = np.eye(p.shape[-1]) - p
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
        """Loop over all fragments."""
        for frag in self.base.fragments:
            if (exclude_self and frag is self):
                continue
            yield frag


    def canonicalize_mo(self, *mo_coeff, eigvals=False):
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
        mo_coeff = np.hstack(mo_coeff)
        fock = np.linalg.multi_dot((mo_coeff.T, self.base.get_fock(), mo_coeff))
        mo_energy, rot = np.linalg.eigh(fock)
        mo_can = np.dot(mo_coeff, rot)
        if eigvals:
            return mo_can, rot, mo_energy
        return mo_can, rot


    def diagonalize_cluster_dm(self, c_bath, tol=1e-4):
        """Diagonalize cluster (fragment+bath) DM to get fully occupied and virtual orbitals.

        Parameters
        ----------
        c_bath : ndarray
            Bath orbitals.
        tol : float, optional
            If set, check that all eigenvalues of the cluster DM are close
            to 0 or 1, with the tolerance given by tol. Default= 1e-4.

        Returns
        -------
        c_occclt : ndarray
            Occupied cluster orbitals.
        c_virclt : ndarray
            Virtual cluster orbitals.
        """
        c_clt = np.hstack((self.c_frag, c_bath))
        sc = np.dot(self.base.get_ovlp(), c_clt)
        dm = np.linalg.multi_dot((sc.T, self.mf.make_rdm1(), sc)) / 2
        e, v = np.linalg.eigh(dm)
        if tol and not np.allclose(np.fmin(abs(e), abs(e-1)), 0, atol=tol, rtol=0):
            raise RuntimeError("Error while diagonalizing cluster DM: eigenvalues not all close to 0 or 1:\n%s", e)
        e, v = e[::-1], v[:,::-1]
        c_clt = np.dot(c_clt, v)
        nocc = sum(e >= 0.5)
        c_occclt, c_virclt = np.hsplit(c_clt, [nocc])
        return c_occclt, c_virclt


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
