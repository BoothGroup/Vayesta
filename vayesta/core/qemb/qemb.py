import logging
from timeit import default_timer as timer
from datetime import datetime
import dataclasses
import copy
import os
import os.path

import numpy as np

import pyscf
import pyscf.gto
import pyscf.mp
import pyscf.ci
import pyscf.cc
import pyscf.lo
import pyscf.pbc
import pyscf.pbc.df
import pyscf.pbc.tools
import pyscf.lib
from pyscf.mp.mp2 import _mo_without_core

import vayesta
from vayesta.core import vlog
from vayesta.core.foldscf import FoldedSCF, fold_scf
from vayesta.core.util import *
from vayesta.core.ao2mo import kao2gmo_cderi
from vayesta.core.ao2mo import postscf_ao2mo
from vayesta.core.ao2mo import postscf_kao2gmo
from vayesta.core.ao2mo.kao2gmo import gdf_to_pyscf_eris
from vayesta import lattmod
from vayesta.core.scmf import PDMET, Brueckner
from vayesta.core.mpi import mpi
from .register import FragmentRegister

# Symmetry
#import vayesta.core.symmetry
#from vayesta.core.symmetry import Symmetry

# Fragmentations
from vayesta.core.fragmentation import make_sao_fragmentation
from vayesta.core.fragmentation import make_iao_fragmentation
from vayesta.core.fragmentation import make_iaopao_fragmentation
from vayesta.core.fragmentation import make_site_fragmentation
from vayesta.core.fragmentation import make_cas_fragmentation

# --- This Package

from .fragment import Fragment
from . import helper
from .rdm import make_rdm1_demo_rhf
from .rdm import make_rdm2_demo_rhf

class Embedding:

    # Shadow this in inherited methods:
    Fragment = Fragment

    @dataclasses.dataclass
    class Options(OptionsBase):
        dmet_threshold: float = 1e-6
        #recalc_vhf: bool = True
        solver_options: dict = dataclasses.field(default_factory=dict)
        wf_partition: str = 'first-occ'     # ['first-occ', 'first-vir', 'democratic']
        store_eris: bool = True             # If True, ERIs will be stored in Fragment._eris
        global_frag_chempot: float = None   # Global fragment chemical potential (e.g. for democratically partitioned DMs)

    def __init__(self, mf, options=None, log=None, overwrite=None, **kwargs):
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
        self.fragments : list
            List of fragments for embedding calculation.
        self.kcell : pyscf.pbc.gto.Cell
            For k-point sampled mean-field calculation, which have been folded to the supercell,
            this will hold the original primitive unit cell.
        self.kpts : (nK, 3) array
            For k-point sampled mean-field calculation, which have been folded to the supercell,
            this will hold the original k-points.
        self.kdf : pyscf.pbc.df.GDF
            For k-point sampled mean-field calculation, which have been folded to the supercell,
            this will hold the original Gaussian density-fitting object.
        """

        # 1) Logging
        # ----------
        self.log = log or logging.getLogger(__name__)
        self.log.info("")
        self.log.info("Initializing %s" % self.__class__.__name__)
        self.log.info("=============%s" % (len(str(self.__class__.__name__))*"="))

        # 2) Options
        # ----------
        if options is None:
            options = self.Options(**kwargs)
        else:
            options = options.replace(kwargs)
        self.opts = options

        # 3) Overwrite methods/attributes
        # -------------------------------
        if overwrite is not None:
            for name, attr in overwrite.items():
                if callable(attr):
                    self.log.info("Overwriting method %s of %s", name, self.__class__.__name__)
                    setattr(self, name, attr.__get__(self))
                else:
                    self.log.info("Overwriting attribute %s of %s", name, self.__class__.__name__)
                    setattr(self, name, attr)

        # 4) Mean-field
        # -------------
        self.mf = None
        self.kcell = None
        self.kpts = None
        self.kdf = None
        self.madelung = None
        with log_time(self.log.timing, "Time for mean-field setup: %s"):
            self.init_mf(mf)
        #with log_time(self.log.timing, "Time for symmetry setup: %s"):
        #    self.symmetry = Symmetry(self.mf)

        # 5) Fragments
        # ------------
        self.register = FragmentRegister()
        self.fragmentation = None
        self.fragments = []

        # 6) Other
        # --------
        self.with_scmf = None   # Self-consistent mean-field


    def _mpi_bcast_mf(self, mf):
        """Use mo_energy and mo_coeff from master MPI rank only."""
        # If vayesta.misc.scf_with_mpi was used, we do not need broadcast
        # as the MO coefficients will already be the same
        if getattr(mf, 'with_mpi', False):
            return
        with log_time(self.log.timing, "Time to broadcast mean-field to all MPI ranks: %s"):
            # Check if all MPI ranks have the same mean-field MOs
            #mo_energy = mpi.world.gather(mf.mo_energy)
            #if mpi.is_master:
            #    moerr = np.max([abs(mo_energy[i] - mo_energy[0]).max() for i in range(len(mpi))])
            #    if moerr > 1e-6:
            #        self.log.warning("Large difference of MO energies between MPI ranks= %.2e !", moerr)
            #    else:
            #        self.log.debugv("Largest difference of MO energies between MPI ranks= %.2e", moerr)
            # Use MOs of master process
            mf.mo_energy = mpi.world.bcast(mf.mo_energy, root=0)
            mf.mo_coeff = mpi.world.bcast(mf.mo_coeff, root=0)

    def init_mf(self, mf):
        self._mf_orig = mf      # Keep track of original mean-field object - be careful not to modify in any way, to avoid side effects!

        # Create shallow copy of mean-field object; this way it can be updated without side effects outside the quantum
        # embedding method if attributes are replaced in their entirety
        # (eg. `mf.mo_coeff = mo_new` instead of `mf.mo_coeff[:] = mo_new`).
        mf = copy.copy(mf)
        self.log.debugv("type(mf)= %r", type(mf))
        # If the mean-field has k-points, automatically fold to the supercell:
        if hasattr(mf, 'kpts') and mf.kpts is not None:
            with log_time(self.log.timing, "Time for k->G folding of MOs: %s"):
                mf = fold_scf(mf)
        if isinstance(mf, FoldedSCF):
            self.kcell, self.kpts, self.kdf = mf.kmf.mol, mf.kmf.kpts, mf.kmf.with_df
        # Make sure that all MPI ranks use the same MOs`:
        if mpi:
            self._mpi_bcast_mf(mf)
        self.mf = mf
        if not (self.is_rhf or self.is_uhf):
            raise ValueError("Cannot deduce RHF or UHF!")

        # Evaluating the Madelung constant is expensive - cache result
        if self.has_exxdiv:
            self.madelung = pyscf.pbc.tools.madelung(self.mol, self.mf.kpt)

        # Original mean-field integrals - do not change these!
        self._ovlp_orig = self.mf.get_ovlp()
        self._hcore_orig = self.mf.get_hcore()
        self._veff_orig = self.mf.get_veff()
        # Cached integrals - these can be changed!
        self._ovlp = self._ovlp_orig
        self._hcore = self._hcore_orig
        self._veff = self._veff_orig

        # Hartree-Fock energy - this can be different from mf.e_tot, when the mean-field
        # is not a (converged) HF calculations
        e_mf = (mf.e_tot / self.ncells)
        e_hf = self.e_mf
        de = (e_mf - e_hf)
        rde = (de / e_mf)
        if not self.mf.converged:
            self.log.warning("Mean-field not converged!")
        self.log.info("Initial E(mean-field)= %s", energy_string(e_mf))
        self.log.info("Calculated E(HF)=      %s", energy_string(e_hf))
        self.log.info("Difference dE=         %s ( %.1f%%)", energy_string(de), rde)
        if (abs(de) > 1e-3) or (abs(rde) > 1e-6):
            self.log.warning("Large difference between initial E(mean-field) and calculated E(HF)!")

        #FIXME (no RHF/UHF dependent code here)
        if self.is_rhf:
            self.log.info("n(AO)= %4d  n(MO)= %4d  n(linear dep.)= %4d", self.nao, self.nmo, self.nao-self.nmo)
        else:
            self.log.info("n(AO)= %4d  n(alpha/beta-MO)= (%4d, %4d)  n(linear dep.)= (%4d, %4d)",
                    self.nao, *self.nmo, self.nao-self.nmo[0], self.nao-self.nmo[1])

        self.check_orthonormal(self.mo_coeff, mo_name='MO')

        if self.mo_energy is not None:
            if self.is_rhf:
                self.log.debugv("MO energies (occ):\n%r", self.mo_energy[self.mo_occ > 0])
                self.log.debugv("MO energies (vir):\n%r", self.mo_energy[self.mo_occ == 0])
            else:
                self.log.debugv("alpha-MO energies (occ):\n%r", self.mo_energy[0][self.mo_occ[0] > 0])
                self.log.debugv("beta-MO energies (occ):\n%r", self.mo_energy[1][self.mo_occ[1] > 0])
                self.log.debugv("alpha-MO energies (vir):\n%r", self.mo_energy[0][self.mo_occ[0] == 0])
                self.log.debugv("beta-MO energies (vir):\n%r", self.mo_energy[1][self.mo_occ[1] == 0])


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
    def has_exxdiv(self):
        """Correction for divergent exact-exchange potential."""
        return (hasattr(self.mf, 'exxdiv') and self.mf.exxdiv is not None)

    def get_exxdiv(self):
        """Get divergent exact-exchange (exxdiv) energy correction and potential.

        Returns
        -------
        e_exxdiv: float
            Divergent exact-exchange energy correction per unit cell.
        v_exxdiv: array
            Divergent exact-exchange potential correction in AO basis.
        """
        if not self.has_exxdiv: return 0, None
        sc = np.dot(self.get_ovlp(), self.mo_coeff[:,:self.nocc])
        e_exxdiv = -self.madelung * self.nocc/self.ncells
        v_exxdiv = -self.madelung * np.dot(sc, sc.T)
        self.log.debugv("Divergent exact-exchange (exxdiv) correction= %+16.8f Ha", e_exxdiv)
        return e_exxdiv, v_exxdiv

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
        if self.kpts is None: return 1
        return len(self.kpts)

    @property
    def is_rhf(self):
        return (np.ndim(self.mo_coeff) == 2)

    @property
    def is_uhf(self):
        return (self.mo_coeff[0].ndim == 2)

    @property
    def has_df(self):
        return (self.df is not None) or (self.kdf is not None)

    @property
    def df(self):
        #if self.kdf is not None:
        #    return self.kdf
        if hasattr(self.mf, 'with_df') and self.mf.with_df is not None:
            return self.mf.with_df
        return None

    # Mean-field properties

    #def init_vhf_ehf(self):
    #    """Get Hartree-Fock potential and energy."""
    #    if self.opts.recalc_vhf:
    #        self.log.debug("Calculating HF potential from mean-field object.")
    #        vhf = self.mf.get_veff()
    #    else:
    #        self.log.debug("Calculating HF potential from MOs.")
    #        cs = np.dot(self.mo_coeff.T, self.get_ovlp())
    #        fock = np.dot(cs.T*self.mo_energy, cs)
    #        vhf = (fock - self.get_hcore())
    #    h1e = self.get_hcore_for_energy()
    #    ehf = self.mf.energy_tot(h1e=h1e, vhf=vhf)
    #    return vhf, ehf

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

    #@mo_energy.setter
    #def mo_energy(self, mo_energy):
    #    """Updating the MOs resets the effective potential cache `_veff`."""
    #    self.log.debugv("MF attribute 'mo_energy' is updated; deleting cached _veff.")
    #    #self._veff = None
    #    self.mf.mo_energy = mo_energy

    #@mo_coeff.setter
    #def mo_coeff(self, mo_coeff):
    #    """Updating the MOs resets the effective potential cache `_veff`."""
    #    self.log.debugv("MF attribute 'mo_coeff' is updated; deleting chached _veff.")
    #    #self._veff = None
    #    self.mf.mo_coeff = mo_coeff

    #@mo_occ.setter
    #def mo_occ(self, mo_occ):
    #    """Updating the MOs resets the effective potential cache `_veff`."""
    #    self.log.debugv("MF attribute 'mo_occ' is updated; deleting chached _veff.")
    #    #self._veff = None
    #    self.mf.mo_occ = mo_occ

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
    def mo_energy_occ(self):
        """Occupied MO energies."""
        return self.mo_energy[:self.nocc]

    @property
    def mo_energy_vir(self):
        """Virtual MO coefficients."""
        return self.mo_energy[self.nocc:]

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
        """Total mean-field energy per unit cell (not folded supercell).
        Note that the input unit cell itself can be a supercell, in which case
        `e_mf` refers to this cell.
        """
        h1e = self.get_hcore_for_energy()
        vhf = self.get_veff_for_energy()
        e_mf = self.mf.energy_tot(h1e=h1e, vhf=vhf)
        return e_mf/self.ncells

    @property
    def e_nuc(self):
        """Nuclear-repulsion energy per unit cell (not folded supercell)."""
        return self.mol.energy_nuc()/self.ncells

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

    # Integrals of the original mean-field object - these cannot be changed:

    def get_ovlp_orig(self):
        return self._ovlp_orig

    def get_hcore_orig(self):
        return self._hcore_orig

    def get_veff_orig(self, with_exxdiv=True):
        if not with_exxdiv and self.has_exxdiv:
            v_exxdiv = self.get_exxdiv()[1]
            return self.get_veff_orig() - v_exxdiv
        return self._veff_orig

    def get_fock_orig(self, with_exxdiv=True):
        return (self.get_hcore_orig() + self.get_veff_orig(with_exxdiv=with_exxdiv))

    # Integrals which change with mean-field updates or chemical potential shifts:

    def get_ovlp(self):
        """AO-overlap matrix."""
        return self._ovlp

    def get_hcore(self):
        """Core Hamiltonian (kinetic energy plus nuclear-electron attraction)."""
        return self._hcore

    def get_veff(self, with_exxdiv=True):
        """Hartree-Fock Coulomb and exchange potential in AO basis."""
        if not with_exxdiv and self.has_exxdiv:
            v_exxdiv = self.get_exxdiv()[1]
            return self.get_veff() - v_exxdiv
        return self._veff

    def get_fock(self, with_exxdiv=True):
        """Fock matrix in AO basis."""
        return self.get_hcore() + self.get_veff(with_exxdiv=with_exxdiv)

    def set_ovlp(self, value):
        self.log.debug("Changing ovlp matrix.")
        self._ovlp = value

    def set_hcore(self, value):
        self.log.debug("Changing hcore matrix.")
        self._hcore = value

    def set_veff(self, value):
        self.log.debug("Changing veff matrix.")
        self._veff = value

    # Integrals for energy evaluation
    # Overwriting these allows using different integrals for the energy evaluation

    def get_hcore_for_energy(self):
        """Core Hamiltonian used for energy evaluation."""
        return self.get_hcore()

    def get_veff_for_energy(self, with_exxdiv=True):
        """Hartree-Fock potential used for energy evaluation."""
        return self.get_veff(with_exxdiv=with_exxdiv)

    def get_fock_for_energy(self, with_exxdiv=True):
        """Fock matrix used for energy evaluation."""
        return (self.get_hcore_for_energy() + self.get_veff_for_energy(with_exxdiv=with_exxdiv))

    def get_fock_for_bath(self, with_exxdiv=True):
        """Fock matrix used for bath orbitals."""
        return self.get_fock(with_exxdiv=with_exxdiv)

    # Other integral methods:

    def get_ovlp_power(self, power):
        """get power of AO overlap matrix.

        For folded calculations, this uses the k-point sampled overlap, for better performance and accuracy.

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

    def get_cderi(self, mo_coeff, compact=False, blksize=None):
        """Get density-fitted three-center integrals in MO basis."""
        if compact:
            raise NotImplementedError()
        if self.kdf is not None:
            return kao2gmo_cderi(self.kdf, mo_coeff)

        if np.ndim(mo_coeff[0]) == 1:
            mo_coeff = (mo_coeff, mo_coeff)

        nao = self.mol.nao
        naux = (self.df.auxcell.nao if hasattr(self.df, 'auxcell') else self.df.auxmol.nao)
        cderi = np.zeros((naux, mo_coeff[0].shape[-1], mo_coeff[1].shape[-1]))
        cderi_neg = None
        if blksize is None:
            blksize = int(1e9 / naux*nao*nao * 8)
        # PBC:
        if hasattr(self.df, 'sr_loop'):
            blk0 = 0
            for labr, labi, sign in self.df.sr_loop(compact=False, blksize=blksize):
                assert np.allclose(labi, 0)
                assert (cderi_neg is None)  # There should be only one block with sign -1
                labr = labr.reshape(-1, nao, nao)
                if (sign == 1):
                    blk1 = (blk0 + labr.shape[0])
                    blk = np.s_[blk0:blk1]
                    blk0 = blk1
                    cderi[blk] = einsum('Lab,ai,bj->Lij', labr, mo_coeff[0], mo_coeff[1])
                elif (sign == -1):
                    cderi_neg = einsum('Lab,ai,bj->Lij', labr, mo_coeff[0], mo_coeff[1])
            return cderi, cderi_neg
        # No PBC:
        blk0 = 0
        for lab  in self.df.loop(blksize=blksize):
            blk1 = (blk0 + lab.shape[0])
            blk = np.s_[blk0:blk1]
            blk0 = blk1
            lab = pyscf.lib.unpack_tril(lab)
            cderi[blk] = einsum('Lab,ai,bj->Lij', lab, mo_coeff[0], mo_coeff[1])
        return cderi, None

    def get_eris_array(self, mo_coeff, compact=False):
        """Get electron-repulsion integrals in MO basis as a NumPy array.

        Parameters
        ----------
        mo_coeff: [list(4) of] (n(AO), n(MO)) array
            MO coefficients.

        Returns
        -------
        eris: (n(MO), n(MO), n(MO), n(MO)) array
            Electron-repulsion integrals in MO basis.
        """
        # PBC with k-points:
        if self.kdf is not None:
            if np.ndim(mo_coeff[0]) == 1:
                mo_coeff = 4*[mo_coeff]
            cderi1, cderi1_neg = kao2gmo_cderi(self.kdf, mo_coeff[:2])
            if (mo_coeff[0] is mo_coeff[2]) and (mo_coeff[1] is mo_coeff[3]):
                cderi2, cderi2_neg = cderi1, cderi1_neg
            else:
                cderi2, cderi2_neg = kao2gmo_cderi(self.kdf, mo_coeff[2:])
            eris = einsum('Lij,Lkl->ijkl', cderi1.conj(), cderi2)
            if cderi1_neg is not None:
                eris -= einsum('Lij,Lkl->ijkl', cderi1_neg.conj(), cderi2_neg)
            return eris
        # Molecules and Gamma-point PBC:
        if hasattr(self.mf, 'with_df') and self.mf.with_df is not None:
            eris = self.mf.with_df.ao2mo(mo_coeff, compact=compact)
        elif self.mf._eri is not None:
            eris = pyscf.ao2mo.kernel(self.mf._eri, mo_coeff, compact=compact)
        else:
            eris = self.mol.ao2mo(mo_coeff, compact=compact)
        if not compact:
            if isinstance(mo_coeff, np.ndarray) and mo_coeff.ndim == 2:
                shape = 4*[mo_coeff.shape[-1]]
            else:
                shape = [mo.shape[-1] for mo in mo_coeff]
            eris = eris.reshape(shape)
        return eris

    def get_eris_object(self, postscf, fock=None):
        """Get ERIs for post-SCF methods.

        For folded PBC calculations, this folds the MO back into k-space
        and contracts with the k-space three-center integrals..

        Parameters
        ----------
        postscf: one of the following PySCF methods: MP2, CCSD, RCCSD, DFCCSD
            Post-SCF method with attribute mo_coeff set.

        Returns
        -------
        eris: _ChemistsERIs
            ERIs which can be used for the respective post-scf method.
        """
        if fock is None:
            if isinstance(postscf, pyscf.mp.mp2.MP2):
                fock = self.get_fock()
            elif isinstance(postscf, (pyscf.ci.cisd.CISD, pyscf.cc.ccsd.CCSD)):
                fock = self.get_fock(with_exxdiv=False)
            else:
                raise ValueError("Unknown post-SCF method: %r", type(postscf))
        # For MO energies, always use get_fock():
        mo_act = _mo_without_core(postscf, postscf.mo_coeff)
        mo_energy = einsum('ai,ab,bi->i', mo_act, self.get_fock(), mo_act)
        e_hf = self.mf.e_tot

        # Fold MOs into k-point sampled primitive cell, to perform efficient AO->MO transformation:
        if self.kdf is not None:
            eris = postscf_kao2gmo(postscf, self.kdf, fock=fock, mo_energy=mo_energy, e_hf=e_hf)
            ## COMPARISON:
            ## OLD:
            #with log_time(self.log.timing, "Time OLD: %s"):
            #    eris_old = gdf_to_pyscf_eris(self.mf, self.kdf, postscf, fock=fock, mo_energy=mo_energy, e_hf=e_hf)
            ## NEW:
            #with log_time(self.log.timing, "Time NEW: %s"):
            #    eris = postscf_kao2gmo(postscf, self.kdf, fock=fock, mo_energy=mo_energy, e_hf=e_hf)

            #for key in ['oooo', 'ovoo', 'ovvo', 'oovv', 'ovov', 'ovvv', 'vvvv', 'vvL']:
            #    if not hasattr(eris, key) or (getattr(eris, key) is None):
            #        continue
            #    old = getattr(eris_old, key)
            #    new = getattr(eris, key)
            #    close = np.allclose(old, new)
            #    assert close
            #    assert old.shape == new.shape
            return eris
        # Regular AO->MO transformation
        eris = postscf_ao2mo(postscf, fock=fock, mo_energy=mo_energy, e_hf=e_hf)
        return eris

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

    def get_fragments(self, fragment_list=None, **filters):
        """Return all fragments which obey the specified conditions.

        Arguments
        ---------
        **kwargs:
            List of returned fragmens will be filtered according to specified
            keyword arguments.

        Returns
        -------
        fragments: list
            List of fragments.

        Examples
        --------

        Only returns fragments with mpi_rank 0, 1, or 2:

        >>> self.get_fragments(mpi_rank=[0,1,2])

        Only returns fragments with no symmetry parent:

        >>> self.get_fragments(sym_parent=None)
        """
        if fragment_list is None:
            fragment_list = self.fragments
        if not filters:
            return fragment_list
        filters = {key : np.atleast_1d(filters[key]) for key in filters}
        fragments = []
        for frag in fragment_list:
            skip = False
            for key, filtr in filters.items():
                val = getattr(frag, key)
                if val not in filtr:
                    skip = True
                    break
            if skip:
                self.log.debugv("Skipping %s: attribute %s= %r, filter= %r", frag, key, val, filtr)
                continue
            self.log.debugv("Returning %s: attribute %s= %r, filter= %r", frag, key, val, filtr)
            fragments.append(frag)
        return fragments

    # Results
    # -------

    make_rdm1_demo = make_rdm1_demo_rhf
    make_rdm2_demo = make_rdm2_demo_rhf

    def check_fragment_nelectron(self):
        nelec_frags = sum([f.sym_factor*f.nelectron for f in self.loop()])
        self.log.info("Total number of mean-field electrons over all fragments= %.8f", nelec_frags)
        if abs(nelec_frags - np.rint(nelec_frags)) > 1e-4:
            self.log.warning("Number of electrons not integer!")
        return nelec_frags

    @mpi.with_allreduce()
    def get_dmet_energy_elec(self):
        """Calculate electronic DMET energy via democratically partitioned density-matrices.

        Returns
        -------
        e_dmet: float
            Electronic DMET energy.
        """
        e_dmet = 0.0
        for f in self.get_fragments(mpi_rank=mpi.rank):
            e_dmet += f.get_fragment_dmet_energy()
        self.log.debugv("E_elec(DMET)= %s", energy_string(e_dmet))
        return e_dmet

    def get_dmet_energy(self, with_nuc=True, with_exxdiv=True):
        """Calculate DMET energy via democratically partitioned density-matrices.

        Parameters
        ----------
        with_nuc: bool, optional
            Include nuclear-repulsion energy. Default: True.
        with_exxdiv: bool, optional
            Include divergent exact-exchange correction. Default: True.

        Returns
        -------
        e_dmet: float
            DMET energy.
        """
        e_dmet = self.get_dmet_energy_elec()
        if with_nuc:
            e_dmet += self.e_nuc
        if with_exxdiv and self.has_exxdiv:
            e_dmet += self.get_exxdiv()[0]
        return e_dmet

    # Utility
    # -------

    def check_orthonormal(self, *mo_coeff, mo_name='', crit_tol=1e-2, err_tol=1e-7):
        """Check orthonormality of mo_coeff."""
        mo_coeff = hstack(*mo_coeff)
        err = dot(mo_coeff.T, self.get_ovlp(), mo_coeff) - np.eye(mo_coeff.shape[-1])
        l2 = np.linalg.norm(err)
        linf = abs(err).max()
        if mo_name:
            mo_name = (' of %ss' % mo_name)
        if max(l2, linf) > crit_tol:
            self.log.critical("Orthonormality error%s: L(2)= %.2e  L(inf)= %.2e !", mo_name, l2, linf)
            raise OrthonormalityError("Orbitals not orhonormal!")
        elif max(l2, linf) > err_tol:
            self.log.error("Orthonormality error%s: L(2)= %.2e  L(inf)= %.2e !", mo_name, l2, linf)
        else:
            self.log.debugv("Orthonormality error%s: L(2)= %.2e  L(inf)= %.2e", mo_name, l2, linf)
        return l2, linf

    # --- Population analysis
    # -----------------------

    def get_lo_coeff(self, local_orbitals='lowdin', minao='auto'):
        if local_orbitals.lower() == 'lowdin':
            # Avoid pre_orth_ao step!
            #self.c_lo = c_lo = pyscf.lo.orth_ao(self.mol, 'lowdin')
            #self.c_lo = c_lo = pyscf.lo.orth_ao(self.mol, 'meta-lowdin', pre_orth_ao=None)
            return self.get_ovlp_power(power=-0.5)
        elif local_orbitals.lower() == 'iao+pao':
            return make_iaopao_fragmentation(self.mf, log=self.log, minao=minao).get_coeff()
        raise ValueError("Unknown local orbitals: %r" % local_orbitals)

    def pop_analysis(self, dm1, mo_coeff=None, local_orbitals='lowdin', minao='auto', write=True, filename=None, filemode='a',
            full=False, mpi_rank=0):
        """
        Parameters
        ----------
        dm1 : (N, N) array
            If `mo_coeff` is None, AO representation is assumed.
        local_orbitals : {'lowdin', 'mulliken', 'iao+pao'} or array
            Kind of population analysis. Default: 'lowdin'.

        Returns
        -------
        pop : (N) array
            Population of atomic orbitals.
        """
        if mo_coeff is not None:
            dm1 = einsum('ai,ij,bj->ab', mo_coeff, dm1, mo_coeff)

        ovlp = self.get_ovlp()
        if isinstance(local_orbitals, str):
            lo = local_orbitals.lower()
            if lo == 'mulliken':
                c_lo = None
            else:
                c_lo = self.get_lo_coeff(lo, minao=minao)
        else:
            c_lo = local_orbitals

        if c_lo is None:
            pop = einsum('ab,ba->a', dm1, ovlp)
        else:
            cs = np.dot(c_lo.T, ovlp)
            pop = einsum('ia,ab,ib->i', cs, dm1, cs)

        if write and (mpi.rank == mpi_rank):
            self.write_population(pop, filename=filename, filemode=filemode, full=full)
        return pop

    def get_atomic_charges(self, pop):
        charges = np.zeros(self.mol.natm)
        spins = np.zeros(self.mol.natm)
        if len(pop) != self.mol.nao:
            raise ValueError("n(AO)= %d n(Pop)= %d" % (self.mol.nao, len(pop)))

        for i, label in enumerate(self.mol.ao_labels(None)):
            charges[label[0]] -= pop[i]
        charges += self.mol.atom_charges()
        return charges, spins

    def write_population(self, pop, filename=None, filemode='a', full=False):
        charges, spins = self.get_atomic_charges(pop)
        if full:
            aoslices = self.mol.aoslice_by_atom()[:,2:]
            aolabels = self.mol.ao_labels()

        if filename is None:
            write = lambda *args : self.log.info(*args)
            write("Population analysis")
            write("-------------------")
        else:
            dirname = os.path.dirname(filename)
            if dirname: os.makedirs(dirname, exist_ok=True)
            f = open(filename, filemode)
            write = lambda fmt, *args : f.write((fmt+'\n') % args)
            tstamp = datetime.now()
            self.log.info("Writing population analysis to file \"%s\". Time-stamp: %s", filename, tstamp)
            write("# Time-stamp: %s", tstamp)
            write("# Population analysis")
            write("# -------------------")

        for atom, charge in enumerate(charges):
            write("%3d %-7s  q= % 11.8f  s= % 11.8f", atom, self.mol.atom_symbol(atom) + ':', charge, spins[atom])
            if full:
                aos = aoslices[atom]
                for ao in range(aos[0], aos[1]):
                    label = aolabels[ao]
                    if np.ndim(pop[0]) == 1:
                        write("    %4d %-16s= % 11.8f  % 11.8f" % (ao, label, pop[0][ao], pop[1][ao]))
                    else:
                        write("    %4d %-16s= % 11.8f" % (ao, label, pop[ao]))
        if filename is not None:
            f.close()

    def check_fragment_nelectron(self):
        nelec_frags = sum([f.sym_factor*f.nelectron for f in self.fragments])
        #nelec = (self.kcell if self.kcell is not None else self.mol).nelectron
        nelec = self.mol.nelectron
        self.log.info("Number of electrons over all fragments= %.8f , system= %.8f", nelec_frags, nelec)
        if abs(nelec_frags - nelec) > 1e-6:
            self.log.warning("Number of electrons over all fragments not equal to the system's number of electrons.")
        return nelec_frags

    # --- Fragmentation methods

    def sao_fragmentation(self):
        """Initialize the quantum embedding method for the use of SAO (Lowdin-AO) fragments."""
        self.fragmentation = make_sao_fragmentation(self.mf, log=self.log)
        self.fragmentation.kernel()

    def site_fragmentation(self):
        """Initialize the quantum embedding method for the use of site fragments."""
        self.fragmentation = make_site_fragmentation(self.mf, log=self.log)
        self.fragmentation.kernel()

    def iao_fragmentation(self, minao='auto'):
        """Initialize the quantum embedding method for the use of IAO fragments.

        Parameters
        ----------
        minao: str, optional
            IAO reference basis set. Default: 'auto'
        """
        self.fragmentation = make_iao_fragmentation(self.mf, log=self.log, minao=minao)
        self.fragmentation.kernel()

    def iaopao_fragmentation(self, minao='auto'):
        """Initialize the quantum embedding method for the use of IAO+PAO fragments.

        Parameters
        ----------
        minao: str, optional
            IAO reference basis set. Default: 'auto'
        """
        self.fragmentation = make_iaopao_fragmentation(self.mf, log=self.log, minao=minao)
        self.fragmentation.kernel()

    def cas_fragmentation(self):
        """Initialize the quantum embedding method for the use of CAS fragments.

        Parameters
        ----------
        minao: str, optional
            IAO reference basis set. Default: 'auto'
        """
        self.fragmentation = make_cas_fragmentation(self.mf, log=self.log)
        self.fragmentation.kernel()

    def add_atomic_fragment(self, atoms, orbital_filter=None, name=None, add_symmetric=True, **kwargs):
        """Create a fragment of one or multiple atoms, which will be solved by the embedding method.

        Parameters
        ----------
        atoms: int, str, list[int], or list[str]
            Atom indices or symbols which should be included in the fragment.
        name: str, optional
            Name for the fragment. If None, a name is automatically generated from the chosen atoms. Default: None.
        add_symmetric: bool, optional
            Add symmetry equivalent fragments. Default: True.
        **kwargs:
            Additional keyword arguments are passed through to the fragment constructor.

        Returns
        -------
        Fragment:
            Fragment object.
        """
        if self.fragmentation is None:
            raise RuntimeError("No fragmentation defined. Call method x_fragmentation() where x=[iao, iaopao, sao, site].")
        atom_indices, atom_symbols = self.fragmentation.get_atom_indices_symbols(atoms)
        name, indices = self.fragmentation.get_atomic_fragment_indices(atoms, orbital_filter=orbital_filter, name=name)
        return self._add_fragment(indices, name, add_symmetric=add_symmetric, atoms=atom_indices, **kwargs)

    def add_orbital_fragment(self, orbitals, atom_filter=None, name=None, **kwargs):
        """Create a fragment of one or multiple orbitals, which will be solved by the embedding method.

        Parameters
        ----------
        orbitals: int, str, list[int], or list[str]
            Orbital indices or labels which should be included in the fragment.
        name: str, optional
            Name for the fragment. If None, a name is automatically generated from the chosen orbitals. Default: None.
        **kwargs:
            Additional keyword arguments are passed through to the fragment constructor.

        Returns
        -------
        Fragment:
            Fragment object.
        """
        if self.fragmentation is None:
            raise RuntimeError("No fragmentation defined. Call method x_fragmentation() where x=[iao, iaopao, sao, site].")
        name, indices = self.fragmentation.get_orbital_fragment_indices(orbitals, atom_filter=atom_filter, name=name)
        return self._add_fragment(indices, name, **kwargs)

    def _add_fragment(self, indices, name, add_symmetric=False, **kwargs):
        c_frag = self.fragmentation.get_frag_coeff(indices)
        c_env = self.fragmentation.get_env_coeff(indices)
        fid, mpirank = self.register.get_next()
        frag = self.Fragment(self, fid, name, c_frag, c_env, mpi_rank=mpirank, **kwargs)
        self.fragments.append(frag)
        # Log fragment orbitals:
        self.log.debugv("Fragment %ss:\n%r", self.fragmentation.name, indices)
        self.log.debug("Fragment %ss of fragment %s:", self.fragmentation.name, name)
        labels = np.asarray(self.fragmentation.labels)[indices]
        helper.log_orbitals(self.log.debug, labels)

        if add_symmetric:
            # Translational symmetry
            #subcellmesh = self.symmetry.nsubcells
            #if subcellmesh is not None and np.any(np.asarray(subcellmesh) > 1):
            subcellmesh = getattr(self.mf, 'subcellmesh', None)
            if subcellmesh is not None and np.any(np.asarray(subcellmesh) > 1):
                self.log.debugv("mean-field has attribute 'subcellmesh'; adding T-symmetric fragments")
                frag.add_tsymmetric_fragments(subcellmesh)

        return frag

    def add_all_atomic_fragments(self, **kwargs):
        """Create a single fragment for each atom in the system.

        Parameters
        ----------
        **kwargs:
            Additional keyword arguments are passed through to each fragment constructor.
        """
        fragments = []
        #for atom in self.symmetry.get_unique_atoms():
        natom = self.kcell.natm if self.kcell is not None else self.mol.natm
        for atom in range(natom):
            frag = self.add_atomic_fragment(atom, **kwargs)
            fragments.append(frag)
        return fragments

    # --- Mean-field updates

    @deprecated
    def reset_fragments(self, *args, **kwargs):
        self.reset()

    def reset(self, *args, **kwargs):
        for x in self.fragments:
            x.reset(*args, **kwargs)

    def update_mf(self, mo_coeff, mo_energy=None, veff=None):
        """Update underlying mean-field object."""
        # Chech orthonormal MOs
        if not np.allclose(dot(mo_coeff.T, self.get_ovlp(), mo_coeff) - np.eye(mo_coeff.shape[-1]), 0):
            raise ValueError("MO coefficients not orthonormal!")
        self.mf.mo_coeff = mo_coeff
        dm = self.mf.make_rdm1(mo_coeff=mo_coeff)
        if veff is None:
            veff = self.mf.get_veff(dm=dm)
        self.set_veff(veff)
        if mo_energy is None:
            # Use diagonal of Fock matrix as MO energies
            mo_energy = einsum('ai,ab,bi->i', mo_coeff, self.get_fock(), mo_coeff)
        self.mf.mo_energy = mo_energy
        self.mf.e_tot = self.mf.energy_tot(dm=dm, h1e=self.get_hcore(), vhf=veff)

    def check_fragment_symmetry(self, dm1, charge_tol=1e-6, spin_tol=1e-6):
        frags = self.get_symmetry_child_fragments(include_parents=True)
        for group in frags:
            parent, children = group[0], group[1:]
            for child in children:
                charge_err = parent.get_tsymmetry_error(child, dm1=dm1)
                if (charge_err > charge_tol):
                    raise RuntimeError("%s and %s not symmetric: charge error= %.3e !"
                            % (parent.name, child.name, charge_err))
                self.log.debugv("Symmetry between %s and %s: charge error= %.3e", parent.name, child.name, charge_err)

    def pdmet_scmf(self, *args, **kwargs):
        """Decorator for p-DMET."""
        self.with_scmf = PDMET(self, *args, **kwargs)
        self.kernel = self.with_scmf.kernel.__get__(self)

    def brueckner_scmf(self, *args, **kwargs):
        """Decorator for Brueckner-DMET."""
        self.with_scmf = Brueckner(self, *args, **kwargs)
        self.kernel = self.with_scmf.kernel.__get__(self)

    # --- Backwards compatibility:

    def get_eris(self, mo_or_cm, *args, **kwargs):  # pragma: no cover
        """For backwards compatibility only!"""
        self.log.warning("get_eris is deprecated!")
        if isinstance(mo_or_cm, np.ndarray):
            return self.get_eris_array(mo_or_cm, *args, **kwargs)
        return self.get_eris_object(mo_or_cm, *args, **kwargs)

    def make_atom_fragment(self, *args, aos=None, add_symmetric=True, **kwargs):    # pragma: no cover
        """Deprecated. Do not use."""
        self.log.warning("make_atom_fragment is deprecated. Use add_atomic_fragment.")
        return self.add_atomic_fragment(*args, orbital_filter=aos, add_symmetric=add_symmetric, **kwargs)

    def make_all_atom_fragments(self, *args, **kwargs):  # pragma: no cover
        """Deprecated. Do not use."""
        self.log.warning("make_all_atom_fragments is deprecated. Use add_all_atomic_fragments.")
        return self.add_all_atomic_fragments(*args, **kwargs)

    def make_ao_fragment(self, *args, **kwargs):  # pragma: no cover
        """Deprecated. Do not use."""
        self.log.warning("make_ao_fragment is deprecated. Use add_orbital_fragment.")
        return self.add_orbital_fragment(*args, **kwargs)

    def init_fragmentation(self, ftype, **kwargs):  # pragma: no cover
        """Deprecated. Do not use."""
        self.log.warning("init_fragmentation is deprecated. Use X_fragmentation(), where X=[iao, iaopao, sao, site].")
        ftype = ftype.lower()
        if ftype == 'iao':
            return self.iao_fragmentation(**kwargs)
        if ftype == 'lowdin-ao':
            return self.sao_fragmentation(**kwargs)
        if ftype == 'site':
            return self.site_fragmentation(**kwargs)
        if ftype == 'ao':
            raise ValueError("AO fragmentation is no longer supported")
        raise ValueError("Unknown fragment type: %r", ftype)
