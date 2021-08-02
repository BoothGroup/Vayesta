import logging
from timeit import default_timer as timer
from datetime import datetime
import dataclasses
import copy

import numpy as np
import scipy
import scipy.linalg

import pyscf
import pyscf.gto
import pyscf.mp
import pyscf.lo
import pyscf.pbc
import pyscf.ao2mo
import pyscf.pbc.gto
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
from vayesta.core.fragment import QEmbeddingFragment
from .kao2gmo import gdf_to_pyscf_eris
from vayesta.misc.gdf import GDF

import copy


class QEmbeddingMethod:

    # Shadow these in inherited methods:
    Fragment = QEmbeddingFragment

    @dataclasses.dataclass
    class Options(OptionsBase):
        copy_mf: bool = True        # Create shallow copy of mean-field object on entry
        recalc_veff: bool = True    # TODO: automatic?


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
        if self.opts.copy_mf:
            mf = copy.copy(mf)
        self.log.debug("type(MF)= %r", type(mf))
        if hasattr(mf, 'kpts') and mf.kpts is not None:
            mf = unfold_scf(mf)
        if isinstance(mf, UnfoldedSCF):
            self.kcell, self.kpts, self.kdf = mf.kmf.mol, mf.kmf.kpts, mf.kmf.with_df
        else:
            self.kcell = self.kpts = self.kdf = None
        self.mf = mf
        # Set current mean-field field to shallow copy of original; all attributes will be the same objects, but
        # reassignments won't overwrite the original object.
        # If we store coefficients separately then we'll need to have a different interface around all solvers using
        # the fock matrix, as the fock matrix calculated from the provided mf object is used by default, so we need
        # to have a separate mf object.
        self.curr_mf = copy.copy(mf)
        # Copy MO attributes, so they can be modified later with no side-effects (updating the mean-field)
        self.mo_energy = self.mf.mo_energy.copy()
        self.mo_coeff = self.mf.mo_coeff.copy()
        self.mo_occ = self.mf.mo_occ.copy()
        self._ovlp = self.mf.get_ovlp()
        self._hcore = self.mf.get_hcore()
        if self.opts.recalc_veff:
            self._veff = self.mf.get_veff()
        else:
            cs = np.dot(self.mo_coeff.T, self.get_ovlp())
            fock = np.dot(cs.T*self.mo_energy, cs)
            self._veff = fock - self.get_hcore()

        # Some MF output
        if self.mf.converged:
            self.log.info("E(MF)= %+16.8f Ha", self.e_mf)
        else:
            self.log.warning("E(MF)= %+16.8f Ha (not converged!)", self.e_mf)
        self.log.info("n(AO)= %4d  n(MO)= %4d  n(linear dep.)= %4d", self.nao, self.nmo, self.nao-self.nmo)
        idterr = self.mo_coeff.T.dot(self._ovlp).dot(self.mo_coeff) - np.eye(self.nmo)
        self.log.log(logging.ERROR if np.linalg.norm(idterr) > 1e-5 else logging.DEBUG,
                "Orthogonality error of MF orbitals: L(2)= %.2e  L(inf)= %.2e", np.linalg.norm(idterr), abs(idterr).max())

        # 3) Fragments
        # ------------
        self.default_fragment_type = None
        self.fragments = []

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

    @property
    def mol(self):
        """Mole or Cell object."""
        return self.mf.mol

    @property
    def has_lattice_vectors(self):
        """Flag if self.mol has lattice vectors defined."""
        return (hasattr(self.mol, 'a') and self.mol.a is not None)
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
    def nfrag(self):
        """Number of fragments."""
        return len(self.fragments)

    @property
    def e_mf(self):
        """Total mean-field energy per unit cell (not unfolded supercell!).
        Note that the input unit cell itself can be a supercell, in which case
        `e_mf` refers to this cell.
        """
        return self.curr_mf.e_tot/self.ncells

    def loop(self):
        """Loop over fragments."""
        for frag in self.fragments:
            yield frag


    # --- Integral methods
    # ====================

    def get_ovlp(self):
        """AO overlap matrix."""
        return self._ovlp

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

    def get_hcore(self):
        return self._hcore

    def get_veff(self):
        """Hartree-Fock Coulomb and exchange potential."""
        return self._veff

    def get_fock(self):
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
        eris = gdf_to_pyscf_eris(self.curr_mf, self.kdf, cm, fock=self.get_fock())
        return eris

    # --- Initialization of fragmentations
    # ------------------------------------
    # These need to be called before any fragments of the respective type can be added.
    # Only one fragmentation type may be initialized!

    def init_fragmentation(self, fragment_type, **kwargs):
        """Initialize 'IAO', 'Lowdin-AO', or 'AO', fragmentation.

        Currently supports
        'IAO' :         Intrinsic atomic orbitals
        'Lowdin-AO' :   Symmetrically orthogonalized atomic orbitals
        'AO' :          Non-orthogonal atomic orbitals

        Check `init_iao_fragmentation`, `init_lowdin_fragmentation`, and `init_ao_fragmentation`
        for further information.

        Parameters
        ----------
        fragment_type : ['IAO', 'Lowdin-AO', 'AO', 'Sites']
            Fragmentation type.
        **kwargs :
            Additional keyword arguments will be passed to the corresponding fragmentation
            initialization method.
        """
        fragment_type = fragment_type.upper()
        if fragment_type == 'IAO':
            return self.init_iao_fragmentation(**kwargs)
        if fragment_type == 'LOWDIN-AO':
            return self.init_lowdin_fragmentation(**kwargs)
        if fragment_type == 'AO':
            return self.init_ao_fragmentation(**kwargs)
        if fragment_type == 'SITE':
            return self.init_site_fragmentation(**kwargs)
        raise ValueError("Unknown fragment_type: %s", fragment_type)


    def init_iao_fragmentation(self, minao='minao'):
        """This needs to be called before any IAO fragments can be added.

        The following attributes will be defined:
        default_fragment_type
        iao_minao
        iao_labels
        iao_coeff
        iao_rest_coeff
        iao_occup

        Parameters
        ----------
        minao : str
            Minimal basis set of IAOs.
        """
        self.log.info("Making IAOs for minimal basis set %s.", minao)
        iao_coeff, iao_rest_coeff = self.make_iao_coeffs(minao)
        iao_labels = self.get_iao_labels(minao)
        iao_occup = self.get_fragment_occupancy(iao_coeff, iao_labels)
        # Define fragmentation specific attributes
        self.default_fragment_type = 'IAO'
        self.iao_minao = minao
        self.iao_labels = iao_labels
        self.iao_coeff = iao_coeff
        self.iao_rest_coeff = iao_rest_coeff
        self.iao_occup = iao_occup


    def init_lowdin_fragmentation(self):
        """This needs to be called before any Lowdin-AO fragments can be added.

        TODO: Linear-dependency treatment

        The following attributes will be defined:
        default_fragment_type
        lao_labels
        lao_coeff
        lao_occup
        """
        self.log.info("Making Lowdin-AOs")
        lao_coeff = pyscf.lo.vec_lowdin(np.eye(self.nao), self.get_ovlp())
        lao_labels = self.mol.ao_labels(None)
        lao_occup = self.get_fragment_occupancy(lao_coeff, lao_labels)
        # Define fragmentation specific attributes
        self.default_fragment_type = 'Lowdin-AO'
        self.lao_labels = lao_labels
        self.lao_coeff = lao_coeff
        self.lao_occup = lao_occup


    def init_ao_fragmentation(self):
        """This needs to be called before any AO fragments can be added.

        TODO: Linear-dependency treatment not needed here (unless fragments get very large...)?

        The following attributes will be defined:
        default_fragment_type
        ao_labels
        """
        self.log.info("Initializing AO fragmentation")
        ao_labels = self.mol.ao_labels(None)
        # Define fragmentation specific attributes
        self.default_fragment_type = 'AO'
        self.ao_labels = ao_labels


    def init_site_fragmentation(self):
        """This needs to be called before any site fragments can be added.

        The following attributes will be defined:
        default_fragment_type
        site_labels
        """
        self.log.info("Initializing site fragmentation")
        site_labels = self.mol.ao_labels(None)
        # Define fragmentation specific attributes
        self.default_fragment_type = 'Site'
        self.site_labels = site_labels


    # Fragmentation methods
    # ---------------------

    def make_atom_fragment(self, atoms, aos=None, name=None, fragment_type=None, **kwargs):
        """Create a fragment for one atom or a set of atoms.

        Parameters
        ----------
        atoms : list
            List of atom IDs or labels.
        aos : list, optional
            Additionally restrict fragment orbitals to a specific AO type (e.g. '2p'). Default: None.
        name : str, optional
            Name for fragment.
        fragment_type : [None, 'IAO', 'Lowdin-AO', 'AO', 'Site']
            Fragment orbital type. If `None`, the value of `self.default_fragment_type` is used.
            Default: `None`.
        **kwargs :
            Additional keyword arguments will be passed through to `add_fragment`.

        Returns
        -------
        frag : self.Fragment
            Fragment object of type self.Fragment.
        """
        fragment_type = (fragment_type or self.default_fragment_type).upper()

        if np.ndim(atoms) == 0:
            atoms = [atoms]

        # `atoms` can either store atoms indices or labels.
        # Determine the other list and store in `atom_indices` and `atom_labels`
        if isinstance(atoms[0], (int, np.integer)):
            atom_indices = atoms
            atom_labels = [self.mol.atom_symbol(i) for i in atoms]
        else:
            atom_labels = atoms
            all_atom_labels = [self.mol.atom_symbol(atm) for atm in range(self.mol.natm)]
            for atom_label in atom_labels:
                if atom_label not in all_atom_labels:
                    raise ValueError("Atom with label %s not in molecule." % atom_label)
            atom_indices = np.nonzero(np.isin(all_atom_labels, atom_labels))[0]
        #assert len(atom_indices) == len(atom_labels)

        # Generate cluster name if not given
        if name is None:
            name = "-".join(atom_labels)

        # Fragment type specific implementation
        if fragment_type == 'IAO':
            # Base atom for each IAO
            iao_atoms = [iao[0] for iao in self.iao_labels]
            # Indices of IAOs based at atoms
            frag_iaos = np.nonzero(np.isin(iao_atoms, atom_indices))[0]
            refmol = pyscf.lo.iao.reference_mol(self.mol, minao=self.iao_minao)
            if aos is not None:
                for ao in aos:
                    if len(refmol.search_ao_label(ao)) == 0:
                        raise ValueError("No orbitals matching the label %s in molecule", ao)
                ao_indices = refmol.search_ao_label(aos)
                frag_iaos = [i for i in frag_iaos if (i in ao_indices)]
            self.log.debug("Adding fragment orbitals %r", np.asarray(refmol.ao_labels())[frag_iaos].tolist())
            c_frag = self.iao_coeff[:,frag_iaos].copy()
            # Combine remaining IAOs and rest virtual space (`iao_rest_coeff`)
            #rest_iaos = np.asarray([i for i in np.arange(self.iao_coeff.shape[-1]) if i not in frag_iaos])
            rest_iaos = np.setdiff1d(range(self.iao_coeff.shape[-1]), frag_iaos)
            c_env = np.hstack((self.iao_coeff[:,rest_iaos], self.iao_rest_coeff))
        elif fragment_type == 'LOWDIN-AO':
            # Base atom for each LowdinAO
            lao_atoms = [lao[0] for lao in self.lao_labels]
            # Indices of LowdinAOs based at atoms
            frag_laos = np.nonzero(np.isin(lao_atoms, atom_indices))[0]
            c_frag = self.lao_coeff[:,frag_laos].copy()
            #rest_laos = np.asarray([i for i in np.arange(self.lao_coeff.shape[-1]) if i not in frag_laos])
            rest_laos = np.setdiff1d(range(self.lao_coeff.shape[-1]), frag_laos)
            c_env = self.lao_coeff[:,rest_laos].copy()
        elif fragment_type == 'AO':
            # TODO: linear-dependency treatment
            ao_atoms = [ao[0] for ao in self.ao_labels]
            frag_aos = np.nonzero(np.isin(ao_atoms, atom_indices))[0]
            p_frag = self.get_subset_ao_projector(frag_aos)
            e, c = scipy.linalg.eigh(p_frag, b=self.get_ovlp())
            e, c = e[::-1], c[:,::-1]
            size = len(e[e>1e-5])
            if size != len(frag_aos):
                raise RuntimeError("Error finding fragment atomic orbitals. Eigenvalues: %s" % e)
            assert np.allclose(np.linalg.multi_dot((c.T, self.get_ovlp(), c)) - np.eye(nao), 0)
            c_frag, c_env = np.hsplit(c, [size])
            # In the case of AOs, we can also store them in the fragment
            kwargs['aos'] = frag_aos
        elif fragment_type == 'SITE':
            sites = atom_indices
            c_frag = np.eye(self.mol.nao_nr())[:,sites]
            rest = [i for i in range(self.mol.nao_nr()) if i not in sites]
            c_env = np.eye(self.mol.nao_nr())[:,rest]
        else:
            raise ValueError("Unknown fragment_type: %s" % fragment_type)

        frag = self.add_fragment(name, c_frag, c_env, fragment_type=fragment_type, atoms=atom_indices, **kwargs)
        return frag


    def get_ao_labels(self, ao_indices, fragment_type=None):
        fragment_type = (fragment_type or self.default_fragment_type).upper()
        if fragment_type in ('LOWDIN-AO', 'AO'):
            mol = self.mol
        elif fragment_type == 'IAO':
            mol = pyscf.lo.iao.reference_mol(self.mol, minao=self.iao_minao)
        ao_labels = np.asarray(mol.ao_labels())[ao_indices]
        return ao_labels


    def get_ao_indices(self, ao_labels, fragment_type=None):
        fragment_type = (fragment_type or self.default_fragment_type).upper()
        if fragment_type in ('LOWDIN-AO', 'AO'):
            mol = self.mol
        elif fragment_type == 'IAO':
            mol = pyscf.lo.iao.reference_mol(self.mol, minao=self.iao_minao)
        for ao_label in ao_labels:
            if len(mol.search_ao_label(ao_label)) == 0:
                raise ValueError("No orbitals matching the label %s in molecule", ao_label)
        ao_indices = mol.search_ao_label(ao_labels)
        return ao_indices


    def make_ao_fragment(self, aos, name=None, fragment_type=None, **kwargs):
        """Create a fragment for one atomic orbitals or a set of atomic orbitals.

        'Atomic orbital' refers here either to intrinsic atomic orbitals,
        Lowdin-orthogonalized atomic orbitals, or non-orthogonal atomic orbitals.

        Parameters
        ----------
        aos : list
            List of AO IDs or labels.
        name : str, optional
            Name for fragment.
        fragment_type : [None, 'IAO', "Lowdin-AO', 'AO']
            Fragment orbital type. If `None`, the value of `self.default_fragment_type` is used.
            Default: `None`.
        **kwargs :
            Additional keyword arguments will be passed through to `add_fragment`.

        Returns
        -------
        frag : self.Fragment
            Fragment object of type self.Fragment.
        """
        fragment_type = (fragment_type or self.default_fragment_type).upper()

        if np.ndim(aos) == 0:
            aos = [aos]

        # `aos` can either store AO indices or labels.
        # Determine the other list and store in `ao_indices` and `ao_labels`
        if isinstance(aos[0], (int, np.integer)):
            ao_indices = aos
            #if fragment_type in ('LOWDIN-AO', 'AO'):
            #    ao_labels = np.asarray(self.mol.ao_labels())[ao_indices]
            #elif fragment_type == 'IAO':
            #    refmol = pyscf.lo.iao.reference_mol(self.mol, minao=self.iao_minao)
            #    ao_labels = np.asarray(refmol.ao_labels())[ao_indices]
            ao_labels = self.get_ao_labels(ao_indices, fragment_type)
        else:
            ao_labels = aos
            #if fragment_type in ('LOWDIN-AO', 'AO'):
            #    for ao_label in ao_labels:
            #        if len(mol.search_ao_label(ao_label)) == 0:
            #            raise ValueError("No orbitals matching the label %s in molecule", ao_label)
            #    ao_indices = self.mol.search_ao_label(ao_labels)
            #elif fragment_type == 'IAO':
            #    refmol = pyscf.lo.iao.reference_mol(self.mol, minao=self.iao_minao)
            #    for ao_label in ao_labels:
            #        if len(refmol.search_ao_label(ao_label)) == 0:
            #            raise ValueError("No orbitals matching the label %s in molecule", ao_label)
            #    ao_indices = refmol.search_ao_label(ao_labels)
            ao_indices = self.get_ao_indices(ao_labels, fragment_type)

        if name is None:
            #name = ",".join(["-".join(ao) for ao in aos])
            #name = ";".join([",".join(ao.split()) for ao in ao_labels])
            name = "-".join([ao.rstrip() for ao in ao_labels])

        # Fragment type specific implementation
        if fragment_type == 'IAO':
            c_frag = self.iao_coeff[:,ao_indices].copy()
            rest_iaos = np.setdiff1d(range(self.iao_coeff.shape[-1]), ao_indices)
            # Combine remaining IAOs and rest virtual space (`iao_rest_coeff`)
            c_env = np.hstack((self.iao_coeff[:,rest_iaos], self.iao_rest_coeff))
        elif fragment_type == 'LOWDIN-AO':
            c_frag = self.lao_coeff[:,ao_indices].copy()
            rest_laos = np.setdiff1d(range(self.lao_coeff.shape[-1]), ao_indices)
            c_env = self.lao_coeff[:,rest_laos].copy()
            kwargs['aos'] = ao_indices
        elif fragment_type == 'AO':
            # TODO: linear-dependency treatment
            p_frag = self.get_subset_ao_projector(ao_indices)
            e, c = scipy.linalg.eigh(p_frag, b=self.get_ovlp())
            e, c = e[::-1], c[:,::-1]
            size = len(e[e>1e-5])
            if size != len(ao_indices):
                raise RuntimeError("Error finding fragment atomic orbitals. Eigenvalues: %s" % e)
            assert np.allclose(np.linalg.multi_dot((c.T, self.get_ovlp(), c)) - np.eye(c.shape[-1]), 0)
            c_frag, c_env = np.hsplit(c, [size])
            kwargs['aos'] = ao_indices
        else:
            raise ValueError("Unknown fragment_type: %s" % fragment_type)

        frag = self.add_fragment(name, c_frag, c_env, fragment_type=fragment_type, **kwargs)
        return frag


    def add_fragment(self, name, c_frag, c_env, fragment_type, sym_factor=1.0, **kwargs):
        """Create Fragment object and add to fragment list.

        This may have to be shadowed by an embedding method specific version!

        Parameters
        ----------
        name : str
            Name for fragment.
        c_frag : ndarray
            Local (fragment) orbital coefficients.
        c_env : ndarray
            All environment (non-fragment) orbital coefficients.
        fragment_type : ['IAO', "Lowdin-AO', 'AO']
            Fragment orbital type.
        sym_factor : float, optional
            Symmetry factor. Default: 1.0.

        Returns
        -------
        frag : self.Fragment
            Fragment object of type self.Fragment.
        """
        # Get new fragment ID
        fid = self.nfrag + 1
        frag = self.Fragment(self, fid=fid, name=name, c_frag=c_frag, c_env=c_env,
                fragment_type=fragment_type, sym_factor=sym_factor, **kwargs)
        self.fragments.append(frag)
        return frag


    def make_all_atom_fragments(self, **kwargs):
        """Create a fragment for each atom in the molecule."""
        fragments = []
        for atom in range(self.mol.natm):
            frag = self.make_atom_fragment(atom, **kwargs)
            fragments.append(frag)
        return fragments



    # IAO fragmentation specific
    # --------------------------

    def make_iao_coeffs(self, minao='minao', return_rest=True):
        """Make intrinsic atomic orbitals (IAOs) and remaining virtual orbitals via projection.

        Parameters
        ----------
        minao : str, optional
            Minimal basis set for IAOs. Default: 'minao'.
        return_rest : bool, optional
            Return coefficients of remaining virtual orbitals. Default: `True`.

        Returns
        -------
        c_iao : (nAO, nIAO) array
            IAO coefficients.
        c_rest : (nAO, nRest) array
            Remaining virtual orbital coefficients. `None`, if `make_rest == False`.
        """
        mo_coeff = self.mo_coeff
        ovlp = self.get_ovlp()

        c_occ = self.mo_coeff[:,self.mo_occ>0]
        c_iao = pyscf.lo.iao.iao(self.mol, c_occ, minao=minao)
        niao = c_iao.shape[-1]
        self.log.info("Total number of IAOs= %4d", niao)

        # Orthogonalize IAO using symmetric orthogonalization
        c_iao = pyscf.lo.vec_lowdin(c_iao, ovlp)

        # Check that all electrons are in IAO space
        sc = np.dot(ovlp, c_iao)
        dm_iao = np.linalg.multi_dot((sc.T, self.curr_mf.make_rdm1(), sc))
        nelec_iao = np.trace(dm_iao)
        self.log.debugv('nelec_iao= %.8f', nelec_iao)
        if abs(nelec_iao - self.mol.nelectron) > 1e-5:
            self.log.error("IAOs do not contain the correct number of electrons: %.8f", nelec_iao)

        # Test orthogonality of IAO
        idterr = c_iao.T.dot(ovlp).dot(c_iao) - np.eye(niao)
        self.log.log(logging.ERROR if np.linalg.norm(idterr) > 1e-5 else logging.DEBUG,
                "Orthogonality error of IAO: L(2)= %.2e  L(inf)= %.2e", np.linalg.norm(idterr), abs(idterr).max())

        if not return_rest:
            return c_iao, None

        # Add remaining virtual space, work in MO space, so that we automatically get the
        # correct linear dependency treatment, if nMO < nAO
        c_iao_mo = np.linalg.multi_dot((self.mo_coeff.T, ovlp, c_iao))
        # Get eigenvectors of projector into complement
        p_iao = np.dot(c_iao_mo, c_iao_mo.T)
        p_rest = np.eye(self.nmo) - p_iao
        e, c = np.linalg.eigh(p_rest)

        # Corresponding expression in AO basis (but no linear-dependency treatment):
        # p_rest = ovlp - ovlp.dot(c_iao).dot(c_iao.T).dot(ovlp)
        # e, c = scipy.linalg.eigh(p_rest, ovlp)
        # c_rest = c[:,e>0.5]

        # Ideally, all eigenvalues of P_env should be 0 (IAOs) or 1 (non-IAO)
        # Error if > 1e-3
        mask_iao, mask_rest = (e <= 0.5), (e > 0.5)
        e_iao, e_rest = e[mask_iao], e[mask_rest]
        if np.any(abs(e_iao) > 1e-3):
            self.log.error("CRITICAL: Some IAO eigenvalues of 1-P_IAO are not close to 0:\n%r", e_iao)
        elif np.any(abs(e_iao) > 1e-6):
            self.log.warning("Some IAO eigenvalues e of 1-P_IAO are not close to 0: n= %d max|e|= %.2e",
                    np.count_nonzero(abs(e_iao) > 1e-6), abs(e_iao).max())
        if np.any(abs(1-e_rest) > 1e-3):
            self.log.error("CRITICAL: Some non-IAO eigenvalues of 1-P_IAO are not close to 1:\n%r", e_rest)
        elif np.any(abs(1-e_rest) > 1e-6):
            self.log.warning("Some non-IAO eigenvalues e of 1-P_IAO are not close to 1: n= %d max|1-e|= %.2e",
                    np.count_nonzero(abs(1-e_rest) > 1e-6), abs(1-e_rest).max())

        if not (np.sum(mask_rest) + niao == self.nmo):
            self.log.critical("Error in construction of remaining virtual orbitals! Eigenvalues of projector 1-P_IAO:\n%r", e)
            self.log.critical("Number of eigenvalues above 0.5 = %d", np.sum(mask_rest))
            self.log.critical("Total number of orbitals = %d", self.nmo)
            raise RuntimeError("Incorrect number of remaining virtual orbitals")
        c_rest = np.dot(self.mo_coeff, c[:,mask_rest])        # Transform back to AO basis

        # --- Some checks below:

        # Test orthogonality of IAO + rest
        c_all = np.hstack((c_iao, c_rest))
        idterr = c_all.T.dot(ovlp).dot(c_all) - np.eye(self.nmo)
        self.log.log(logging.ERROR if np.linalg.norm(idterr) > 1e-5 else logging.DEBUG,
                "Orthogonality error of IAO+vir. orbitals: L(2)= %.2e  L(inf)= %.2e", np.linalg.norm(idterr), abs(idterr).max())

        return c_iao, c_rest


    def get_fragment_occupancy(self, coeff, labels=None, verbose=True):
        """Get electron occupancy of all fragment orbitals.

        This can be used for any orthogonal fragment basis (IAO, LowdinAO)

        Parameters
        ----------
        coeff : (nAO, nFO) array
            Fragment orbital coefficients.
        labels : (nFO), array
            Fragment orbital labels. Only needed if verbose==True.
        verbose : bool, optional
            Check lattice symmetry of fragment orbitals and print occupations per atom.
            Default: True.

        Returns
        -------
        occup : (nFO,) array
            Occupation of fragment orbitals.
        """
        sc = np.dot(self.get_ovlp(), coeff)
        occup = einsum('ai,ab,bi->i', sc, self.curr_mf.make_rdm1(), sc)
        if not verbose:
            return occup

        if len(labels) != coeff.shape[-1]:
            raise RuntimeError("Inconsistent number of fragment orbitals and labels.")
        # Occupancy per atom
        occup_atom = []
        atoms = np.asarray([i[0] for i in labels])
        self.log.debugv('atoms= %r', atoms)
        for a in range(self.mol.natm):
            mask = np.where(atoms == a)[0]
            occup_atom.append(occup[mask])
        self.log.debugv("occup_atom: %r", occup_atom)

        # Check lattice symmetry if k-point mf object was used
        tsym = False
        if self.ncells > 1:
            # Fragment orbital occupations per cell
            occup_cell = np.split(np.hstack(occup_atom), self.ncells)
            self.log.debugv("occup_cell: %r", occup_cell)
            # Compare all cells to the primitive cell
            tsym = np.all([np.allclose(occup_cell[i], occup_cell[0]) for i in range(self.ncells)])
        self.log.debugv("Translation symmetry in fragment orbitals: %r", tsym)

        # Print occupations of IAOs
        self.log.info("Fragment Orbital Occupancy per Atom")
        self.log.info("-----------------------------------")
        for a in range(self.mol.natm if not tsym else self.kcell.natm):
            mask = np.where(atoms == a)[0]
            fmt = "  > %3d: %-8s total= %12.8f" + len(occup_atom[a])*"  %s= %10.8f"
            sublabels = [("_".join((x[2], x[3])) if x[3] else x[2]) for x in np.asarray(labels)[mask]]
            vals = [val for pair in zip(sublabels, occup_atom[a]) for val in pair]
            self.log.info(fmt, a, self.mol.atom_symbol(a), np.sum(occup_atom[a]), *vals)

        return occup


    def get_iao_labels(self, minao):
        """Get labels of IAOs

        Parameters
        ----------
        minao : str, optional
            Minimal basis set for IAOs. Default: 'minao'.

        Returns
        -------
        iao_labels : list of length nIAO
            Orbital label (atom-id, atom symbol, nl string, m string) for each IAO.
        """
        refmol = pyscf.lo.iao.reference_mol(self.mol, minao=minao)
        iao_labels_refmol = refmol.ao_labels(None)
        self.log.debugv('iao_labels_refmol: %r', iao_labels_refmol)
        if refmol.natm == self.mol.natm:
            iao_labels = iao_labels_refmol
        # If there are ghost atoms in the system, they will be removed in refmol.
        # For this reason, the atom IDs of mol and refmol will not agree anymore.
        # Here we will correct the atom IDs of refmol to agree with mol
        # (they will no longer be contiguous integers).
        else:
            ref2mol = []
            for refatm in range(refmol.natm):
                ref_coords = refmol.atom_coord(refatm)
                for atm in range(self.mol.natm):
                    coords = self.mol.atom_coord(atm)
                    if np.allclose(coords, ref_coords):
                        self.log.debugv('reference cell atom %r maps to atom %r', refatm, atm)
                        ref2mol.append(atm)
                        break
                else:
                    raise RuntimeError("No atom found with coordinates %r" % ref_coords)
            iao_labels = []
            for iao in iao_labels_refmol:
                iao_labels.append((ref2mol[iao[0]], iao[1], iao[2], iao[3]))
        self.log.debugv('iao_labels: %r', iao_labels)
        assert (len(iao_labels_refmol) == len(iao_labels))
        return iao_labels


    # AO fragmentation specific
    # -------------------------
    # Not tested

    def get_subset_ao_projector(self, aos):
        """Get projector onto AO subspace in the non-orthogonal AO basis.

        Projector from large (1) to small (2) AO basis according to https://doi.org/10.1021/ct400687b
        This is a special case of the more general `get_ao_projector`, which can also
        handle a different AO basis set.

        Parameters
        ----------
        aos : list of AO indices or AO labels or mask
            List of indices/labels or mask of subspace AOs. If a list of labels is given,
            it is converted to AO indices using the PySCF `search_ao_label` function.

        Returns
        -------
        p : (nAO, nAO) array
            Projector onto AO subspace.
        """
        s1 = self.get_ovlp()
        if aos is None:
            aos = np.s_[:]

        if isinstance(aos, slice):
            s2 = s1[aos,aos]
        elif isinstance(aos[0], str):
            self.log.debugv("Searching for AO indices of AOs %r", aos)
            aos_idx = self.mol.search_ao_label(aos)
            self.log.debugv("Found AO indices: %r", aos_idx)
            self.log.debugv("Corresponding to AO labels: %r", np.asarray(self.mol.ao_labels())[aos_idx])
            if len(aos_idx) == 0:
                raise RuntimeError("No AOs with labels %r found" % aos)
            aos = aos_idx
            s2 = s1[np.ix_(aos, aos)]
        else:
            s2 = s1[np.ix_(aos, aos)]
        s21 = s1[aos]
        p21 = scipy.linalg.solve(s2, s21, assume_a="pos")
        p = np.dot(s21.T, p21)
        assert np.allclose(p, p.T)
        return p


    def get_ao_projector(self, aos, basis=None):
        """Get projector onto AO subspace in the non-orthogonal AO basis.

        Projector from large (1) to small (2) AO basis according to https://doi.org/10.1021/ct400687b

        TODO: This is probably not correct (check Ref. above) if basis is not fully contained
        in the span of self.mol.basis. In this case a <1|2> is missing.

        Parameters
        ----------
        aos : list of AO indices or AO labels or mask
            List of indices/labels or mask of subspace AOs. If a list of labels is given,
            it is converted to AO indices using the PySCF `search_ao_label` function.
        basis : str, optional
            Basis set for AO subspace. If `None`, the same basis set as that of `self.mol`
            is used. Default: `None`.

        Returns
        -------
        p : (nAO, nAO) array
            Projector onto AO subspace.
        """

        mol1 = self.mol
        s1 = self.get_ovlp()

        # AOs are given in same basis as mol1
        if basis is None:
            mol2 = mol1
            s2 = s21 = s1
        # AOs of a different basis
        else:
            mol2 = mol1.copy()
            # What was this for? - commented for now
            #if getattr(mol2, 'rcut', None) is not None:
            #    mol2.rcut = None
            mol2.build(False, False, basis=basis2)

            if self.boundary_cond == 'open':
                s2 = mol2.intor_symmetric('int1e_ovlp')
                s12 = pyscf.gto.mole.intor_cross('int1e_ovlp', mol1, mol2)
            else:
                s2 = np.asarray(mol2.pbc_intor('int1e_ovlp', hermi=1, kpts=None))
                s21 = np.asarray(pyscf.pbc.gto.cell.intor_cross('int1e_ovlp', mol1, mol2, kpts=None))
        assert s1.ndim == 2
        assert s2.ndim == 2
        assert s21.ndim == 2

        # All AOs
        if aos is None:
            aos = np.s_[:]

        if isinstance(aos, slice):
            s2 = s1[aos,aos]
        elif isinstance(aos[0], str):
            self.log.debugv("Searching for AO indices of AOs %r", aos)
            aos_idx = mol2.search_ao_label(aos)
            self.log.debugv("Found AO indices: %r", aos_idx)
            self.log.debugv("Corresponding to AO labels: %r", np.asarray(mol2.ao_labels())[aos_idx])
            if len(aos_idx) == 0:
                raise RuntimeError("No AOs with labels %r found" % aos)
            aos = aos_idx
            s2 = s1[np.ix_(aos, aos)]
        else:
            s2 = s1[np.ix_(aos, aos)]

        s21 = s21[aos]

        p21 = scipy.linalg.solve(s2, s21, assume_a="pos")
        p = np.dot(s21.T, p21)
        assert np.allclose(p, p.T)
        return p

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


    def get_symmetry_child_fragments(self):
        """Returns a list of all fragments, which are children to symmetry related parent fragments.

        Returns
        -------
        children: list of lists
            A list with the length of the number of parent fragments in the system, each element
            being another list containing all the children fragments of the given parent fragment.
            Both the outer and inner lists are ordered in the same way that the fragments appear in `self.fragments`.
        """
        parent_ids = [x.id for x in self.get_symmetry_parent_fragments()]
        children = len(parent_ids)*[[]]
        for f in self.fragments:
            if f.sym_parent is None: continue
            pid = f.sym_parent.id
            assert (pid in parent_ids)
            idx = parent_ids.index(pid)
            children[idx].append(f)
        return children


    # Utility
    # -------

    def pop_analysis(self, dm1, mo_coeff=None, kind='lo', c_lo=None, filename=None, filemode='a', verbose=True):
    #def pop_analysis(self, dm1, mo_coeff=None, kind='mulliken', c_lo=None, filename=None, filemode='a', verbose=True):
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

        #shellslices = self.mol.aoslice_by_atom()[:,:2]
        aoslices = self.mol.aoslice_by_atom()[:,2:]
        aolabels = self.mol.ao_labels()

        for atom in range(self.mol.natm):
            write("> Charge of atom %d%-6s= % 11.8f (% 11.8f electrons)", atom, self.mol.atom_symbol(atom), chg[atom], elecs[atom])
            aos = aoslices[atom]
            for ao in range(aos[0], aos[1]):
                label = aolabels[ao]
                write("    %4d %-16s= % 11.8f" % (ao, label, pop[ao]))
            #for sh in range(self.mol.nbas):
            #    # Loop over AOs in shell

        if filename is not None:
            f.close()
        return pop, chg

