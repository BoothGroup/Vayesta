import logging
from timeit import default_timer as timer
from datetime import datetime
import dataclasses
import copy
import itertools
import os
import os.path
from typing import Optional

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
from vayesta.core import spinalg
from vayesta.core.ao2mo import kao2gmo_cderi
from vayesta.core.ao2mo import postscf_ao2mo
from vayesta.core.ao2mo import postscf_kao2gmo
from vayesta import lattmod
from vayesta.core.scmf import PDMET, Brueckner
from vayesta.core.qemb.scrcoulomb import build_screened_eris
from vayesta.mpi import mpi
from .register import FragmentRegister

# Symmetry
from vayesta.core.symmetry import SymmetryGroup
from vayesta.core.symmetry import SymmetryRotation
from vayesta.core.symmetry import SymmetryTranslation

# Fragmentations
from vayesta.core.fragmentation import SAO_Fragmentation
from vayesta.core.fragmentation import IAO_Fragmentation
from vayesta.core.fragmentation import IAOPAO_Fragmentation
from vayesta.core.fragmentation import Site_Fragmentation
from vayesta.core.fragmentation import CAS_Fragmentation

from vayesta.misc.cptbisect import ChempotBisection

# Expectation values
from vayesta.core.qemb.corrfunc import get_corrfunc
from vayesta.core.qemb.corrfunc import get_corrfunc_mf

# --- This Package

from .fragment import Fragment
#from . import helper
from .rdm import make_rdm1_demo_rhf
from .rdm import make_rdm2_demo_rhf


@dataclasses.dataclass
class Options(OptionsBase):
    store_eris: bool = True             # If True, ERIs will be stored in Fragment._eris
    global_frag_chempot: float = None   # Global fragment chemical potential (e.g. for democratically partitioned DMs)
    dm_with_frozen: bool = False        # Add frozen parts to cluster DMs
    # --- Bath options
    bath_options: dict = OptionsBase.dict_with_defaults(
        # DMET bath
        bathtype='dmet', dmet_threshold=1e-6,
        # R2 bath
        rcut=None, unit='Ang',
        # MP2 bath
        threshold=None, truncation='occupation', project_t2=False, addbuffer=False,
        # General
        canonicalize=True,
        )
    # --- Solver options
    solver_options: dict = OptionsBase.dict_with_defaults(
            # General
            conv_tol=None,
            # CCSD
            solve_lambda=False, conv_tol_normt=None, t_as_lambda=False,
            # FCI
            threads=1, max_cycle=300, fix_spin=0.0, lindep=None,
            # EBFCI/EBCCSD
            max_boson_occ=2,
            # Dump
            dumpfile='clusters.h5')
    # --- Other
    screening: Optional[str] = None

class Embedding:
    """Base class for quantum embedding methods.

    Parameters
    ----------
    mf : pyscf.scf.SCF
        PySCF mean-field object.
    solver : str, optional
        Default solver for cluster problems. The available solvers depend on the embedding class.
        Default: 'CCSD'.
    log : logging.Logger, optional
        Vayesta logger object. Default: None
    bath_options : dict, optional
        Bath specific options. The bath type is determined by the key `bathtype` (default: 'DMET').
        The following bath specific options can be specified.

        All bath types:

            dmet_threshold : float, optional
                Threshold for DMET bath orbitals. Orbitals with eigenvalues larger than `dmet_threshold`
                or smaller than 1-`dmet_threshold` will be added as bath orbitals. Default: 1e-6.

        MP2 bath (`bathtype = 'MP2'`):

            threshold : float
                Threshold for MP2 natural orbital truncation. Orbitals with eigenvalues larger than
                `threshold` will be added as bath orbitals.

        R2 bath (`bathtype = 'R2'`):

            rcut : float
                Range cutoff for R2 bath. Orbitals with eigenvalues smaller than `rcut` will be added
                as bath orbitals.
            unit : {'Ang', 'Bohr'}, optional
                Unit of `rcut`. Default: 'Ang'.

    solver_options : dict, optional
        Solver specific options. The following solver specific options can be specified.


            conv_tol : float
                Energy convergence tolerance [valid for 'CISD', 'CCSD', 'TCCSD', 'FCI']
            conv_tol_normt : float
                Amplitude convergence tolerance [valid for 'CCSD', 'TCCSD']
            fix_spin : float
                Target specified spin state [valid for 'FCI']
            t_as_lambda : bool
                Use T-amplitudes as Lambda-amplitudes [valid for 'CCSD', 'TCCSD']
            solve_lambda : bool
                Solve Lambda-equations [valid for 'CCSD', 'TCCSD']
            dumpfile : str
                Dump cluster orbitals and integrals to file [valid for 'Dump']

    Attributes
    ----------
    mol
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



    # Shadow these in inherited methods:
    Fragment = Fragment
    Options = Options
    valid_solvers = ['HF', 'MP2', 'CISD', 'CCSD', 'TCCSD', 'FCI', 'FCI-SPIN0', 'FCI-SPIN1', 'Dump']

    # Deprecated:
    is_rhf = True
    is_uhf = False
    # Use instead:
    spinsym = 'restricted'

    def __init__(self, mf, solver='CCSD', log=None, overwrite=None, **kwargs):
        # 1) Logging
        # ----------
        self.log = log or logging.getLogger(__name__)
        self.log.info("")
        self.log.info("INITIALIZING %s" % self.__class__.__name__.upper())
        self.log.info("=============%s" % (len(str(self.__class__.__name__))*"="))
        with self.log.indent():

            # 2) Options
            # ----------
            self.opts = self.Options().replace(**kwargs)

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

            # 5) Other
            # --------
            if solver not in self.valid_solvers:
                raise ValueError("Unknown solver: %s" % solver)
            self.solver = solver
            self.symmetry = SymmetryGroup(self.mol)
            nimages = getattr(self.mf, 'subcellmesh', None)
            if nimages:
                self.symmetry.set_translations(nimages)
            # Rotations need to be added manually!

            self.register = FragmentRegister()
            self.fragments = []
            self.with_scmf = None   # Self-consistent mean-field
            # Initialize results
            self._reset()


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
        if getattr(mf, 'kpts', None) is not None:
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

        self._check_orthonormal(self.mo_coeff, mo_name='MO')

        if self.mo_energy is not None:
            if self.is_rhf:
                self.log.debugv("MO energies (occ):\n%r", self.mo_energy[self.mo_occ > 0])
                self.log.debugv("MO energies (vir):\n%r", self.mo_energy[self.mo_occ == 0])
            else:
                self.log.debugv("alpha-MO energies (occ):\n%r", self.mo_energy[0][self.mo_occ[0] > 0])
                self.log.debugv("beta-MO energies (occ):\n%r", self.mo_energy[1][self.mo_occ[1] > 0])
                self.log.debugv("alpha-MO energies (vir):\n%r", self.mo_energy[0][self.mo_occ[0] == 0])
                self.log.debugv("beta-MO energies (vir):\n%r", self.mo_energy[1][self.mo_occ[1] == 0])

    def change_options(self, **kwargs):
        self.opts.replace(**kwargs)
        for fx in self.fragments:
            fkwds = {key : kwargs[key] for key in [key for key in kwargs if hasattr(fx.opts, key)]}
            fx.change_options(**fkwds)

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
    def pbc_dimension(self):
        return getattr(self.mol, 'dimension', 0)

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
    def has_df(self):
        return (self.df is not None) or (self.kdf is not None)

    @property
    def df(self):
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

    def _get_ovlp_orig(self):
        return self._ovlp_orig

    def _get_hcore_orig(self):
        return self._hcore_orig

    def _get_veff_orig(self, with_exxdiv=True):
        if not with_exxdiv and self.has_exxdiv:
            v_exxdiv = self.get_exxdiv()[1]
            return self._get_veff_orig() - v_exxdiv
        return self._veff_orig

    def _get_fock_orig(self, with_exxdiv=True):
        return (self._get_hcore_orig() + self._get_veff_orig(with_exxdiv=with_exxdiv))

    # Integrals which change with mean-field updates or chemical potential shifts:

    def get_ovlp(self):
        """AO-overlap matrix."""
        return self._ovlp

    def get_hcore(self):
        """Core Hamiltonian (kinetic energy plus nuclear-electron attraction)."""
        return self._hcore

    def get_veff(self, dm1=None, with_exxdiv=True):
        """Hartree-Fock Coulomb and exchange potential in AO basis."""
        if not with_exxdiv and self.has_exxdiv:
            v_exxdiv = self.get_exxdiv()[1]
            return self.get_veff(dm1=dm1) - v_exxdiv
        if dm1 is None:
            return self._veff
        return self.mf.get_veff(dm=dm1)

    def get_fock(self, dm1=None, with_exxdiv=True):
        """Fock matrix in AO basis."""
        return self.get_hcore() + self.get_veff(dm1=dm1, with_exxdiv=with_exxdiv)

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

    def get_veff_for_energy(self, dm1=None, with_exxdiv=True):
        """Hartree-Fock potential used for energy evaluation."""
        return self.get_veff(dm1=dm1, with_exxdiv=with_exxdiv)

    def get_fock_for_energy(self, dm1=None, with_exxdiv=True):
        """Fock matrix used for energy evaluation."""
        return (self.get_hcore_for_energy() + self.get_veff_for_energy(dm1=dm1, with_exxdiv=with_exxdiv))

    def get_fock_for_bath(self, dm1=None, with_exxdiv=True):
        """Fock matrix used for bath orbitals."""
        return self.get_fock(dm1=dm1, with_exxdiv=with_exxdiv)

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

    @log_method()
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

    @log_method()
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
            return postscf_kao2gmo(postscf, self.kdf, fock=fock, mo_energy=mo_energy, e_hf=e_hf)
        # Regular AO->MO transformation
        eris = postscf_ao2mo(postscf, fock=fock, mo_energy=mo_energy, e_hf=e_hf)
        return eris

    @log_method()
    @with_doc(build_screened_eris)
    def build_screened_eris(self, *args, **kwargs):
        seris_ov, delta_e = build_screened_eris(self, *args, **kwargs)
        return seris_ov, delta_e

    # Symmetry between fragments
    # --------------------------

    def add_symmetric_fragments(self, symmetry, symbol=None, symtol=1e-6, check_mf=True):
        """Add rotationally or translationally symmetric fragments.

        Parameters
        ----------
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
        symtype = symmetry['type']
        if symtype == 'rotation':
            order = symmetry['order']
            axis = np.asarray(symmetry['axis'], dtype=float)
            center = np.asarray(symmetry['center'], dtype=float)
            unit = symmetry['unit'].lower()
            if unit == 'ang':
                BOHR = 0.529177210903
                center = center/BOHR # To Bohr
            elif unit == 'latvec':
                kcell = self.kcell if self.kcell is not None else self.mol
                ak = kcell.lattice_vectors()
                center = np.dot(center, ak)
                # units of lattice vectors also change the axis:
                axis = np.dot(axis, ak)
            if self.kcell is not None:
                # center needs to be moved to the supercell center!
                ak = self.kcell.lattice_vectors()
                bk = np.linalg.inv(ak)
                a = self.mol.lattice_vectors()
                shift = (np.diag(a)/np.diag(ak) - 1)/2
                # Shift in internal coordinates
                self.log.debugv("Primitive cell rotation center= %r" % center)
                center = np.dot(np.dot(center, bk) + shift, ak)
                self.log.debugv("Supercell rotation center= %r" % center)
            symlist = range(1, order)
            symbol = symbol or 'R'
        elif symtype == 'translation':
            translation = np.asarray(symmetry['translation'])
            symlist = list(itertools.product(range(translation[0]), range(translation[1]), range(translation[2])))[1:]
            symbol = symbol or 'T'
        else:
            raise ValueError

        ovlp = self.get_ovlp()
        if check_mf:
            dm1 = self.mf.make_rdm1()

        ftree = [[fx] for fx in self.get_fragments()]
        for i, sym in enumerate(symlist):

            if symtype == 'rotation':
                rotvec = 2*np.pi * (sym/order) * axis/np.linalg.norm(axis)
                sym_op = SymmetryRotation(self.symmetry, rotvec, center=center)
            elif symtype == 'translation':
                transvec = np.asarray(sym)/translation
                sym_op = SymmetryTranslation(self.symmetry, transvec)

            for flist in ftree:
                parent = flist[0]
                # Name for symmetry related fragment
                if symtype == 'rotation':
                    name = '%s_%s(%d)' % (parent.name, symbol, sym)
                elif symtype == 'translation':
                    name = '%s_%s(%d,%d,%d)' % (parent.name, symbol, *sym)
                # Translated coefficients
                c_frag_t = sym_op(parent.c_frag)
                c_env_t = None  # Avoid expensive symmetry operation on environment orbitals
                # Check that translated fragment does not overlap with current fragment:
                fragovlp = parent._csc_dot(parent.c_frag, c_frag_t, ovlp=ovlp)
                if self.spinsym == 'restricted':
                    fragovlp = abs(fragovlp).max()
                elif self.spinsym == 'unrestricted':
                    fragovlp = max(abs(fragovlp[0]).max(), abs(fragovlp[1]).max())
                if (fragovlp > 1e-8):
                    self.log.critical("%s of fragment %s not orthogonal to original fragment (overlap= %.3e)!",
                                sym_op, parent.name, fragovlp)
                    raise RuntimeError("Overlapping fragment spaces.")

                # Add fragment
                frag_id = self.register.get_next_id()
                frag = self.Fragment(self, frag_id, name, c_frag_t, c_env_t, sym_parent=parent, sym_op=sym_op,
                        mpi_rank=parent.mpi_rank, **parent.opts.asdict())
                # Check symmetry
                # (only for the first rotation or primitive translations (1,0,0), (0,1,0), and (0,0,1)
                # to reduce number of sym_op(c_env) calls)
                if check_mf and (abs(np.asarray(sym)).sum() == 1):
                    charge_err, spin_err = parent.get_symmetry_error(frag, dm1=dm1)
                    if max(charge_err, spin_err) > symtol:
                        self.log.critical("Mean-field DM1 not symmetric for %s of %s (errors: charge= %.3e, spin= %.3e)!",
                            sym_op, parent.name, charge_err, spin_err)
                        raise RuntimeError("MF not symmetric under %s" % sym_op)
                    else:
                        self.log.debugv("Mean-field DM symmetry error for %s of %s: charge= %.3e, spin= %.3e",
                            sym_op, parent.name, charge_err, spin_err)

                # Insert after parent fragment
                flist.append(frag)
        # Update fragment list
        self.fragments = [fx for flist in ftree for fx in flist]
        self.log.info("Added %d %s-symmetry related fragments for each fragment.", len(symlist), symtype)

    def add_rotsym_fragments(self, order, axis, center, unit='Ang', **kwargs):
        """Add rotationally symmetric fragments.

        Parameters
        ----------
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
        symmetry = dict(type='rotation', order=order, axis=axis, center=center, unit=unit)
        return self.add_symmetric_fragments(symmetry, **kwargs)

    def add_transsym_fragments(self, translation, **kwargs):
        """Add translationally symmetric fragments.

        Parameters
        ----------
        translation: array(3) of integers
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
        symmetry = dict(type='translation', translation=translation)
        return self.add_symmetric_fragments(symmetry, **kwargs)

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

    def get_fragments(self, fragments=None, **filters):
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
        if fragments is None:
            fragments = self.fragments
        if not filters:
            return fragments
        filters = {k: (v if callable(v) else np.atleast_1d(v)) for k, v in filters.items()}
        filtered_fragments = []
        for frag in fragments:
            skip = False
            for key, filtr in filters.items():
                attr = getattr(frag, key)
                if callable(filtr):
                    if not filtr(attr):
                        skip = True
                        break
                elif attr not in filtr:
                    skip = True
                    break
            if skip:
                continue
            filtered_fragments.append(frag)
        return filtered_fragments

    def _absorb_fragments(self, tol=1e-10):
        """TODO"""
        for fx in self.get_fragments(active=True):
            for fy in self.get_fragments(active=True):
                if (fx.id == fy.id):
                    continue
                if not (fx.active and fy.active):
                    continue

                def svd(cx, cy):
                    rxy = np.dot(cx.T, cy)
                    u, s, v = np.linalg.svd(rxy, full_matrices=False)
                    if s.min() >= (1-tol):
                        nx = cx.shape[-1]
                        ny = cy.shape[-1]
                        swap = False if (nx >= ny) else True
                        return swap
                    return None

                cx_occ = fx.get_overlap('mo[occ]|cluster[occ]')
                cy_occ = fy.get_overlap('mo[occ]|cluster[occ]')
                swocc = svd(cx_occ, cy_occ)
                if swocc is None:
                    continue

                cx_vir = fx.get_overlap('mo[vir]|cluster[vir]')
                cy_vir = fy.get_overlap('mo[vir]|cluster[vir]')
                swvir = svd(cx_vir, cy_vir)
                if swocc != swvir:
                    continue

                # Absorb smaller
                if swocc:
                    fx, fy = fy, fx
                c_frag = hstack(fx.c_frag, fy.c_frag)
                fx.c_frag = c_frag
                name = '/'.join((fx.name, fy.name))
                fy.active = False
                self.log.info("Subspace found: adding %s to %s (new name= %s)!", fy, fx, name)
                # Update fx
                fx.name = name
                fx.c_env = None
                fx._dmet_bath = None
                fx._occ_bath_factory = None
                fx._vir_bath_factory = None

    # Results
    # -------

    def communicate_clusters(self):
        """Communicate cluster orbitals between MPI ranks."""
        if not mpi:
            return
        with log_time(self.log.timing, "Time to communicate clusters: %s"):
            for x in self.get_fragments(sym_parent=None):
                source = x.mpi_rank
                if (mpi.rank == source):
                    x.cluster.mf = None
                cluster = mpi.world.bcast(x.cluster, root=source)
                if (mpi.rank != source):
                    x.cluster = cluster
                x.cluster.mf = self.mf

    @log_method()
    @with_doc(make_rdm1_demo_rhf)
    def make_rdm1_demo(self, *args, **kwargs):
        self.require_complete_fragmentation("Democratically partitioned DMs will not be accurate.")
        return make_rdm1_demo_rhf(self, *args, **kwargs)

    @log_method()
    @with_doc(make_rdm2_demo_rhf)
    def make_rdm2_demo(self, *args, **kwargs):
        self.require_complete_fragmentation("Democratically partitioned DMs will not be accurate.")
        return make_rdm2_demo_rhf(self, *args, **kwargs)

    def get_dmet_elec_energy(self, version=0, approx_cumulant=True):
        """Calculate electronic DMET energy via democratically partitioned density-matrices.

        Returns
        -------
        e_dmet: float
            Electronic DMET energy.
        """
        self.require_complete_fragmentation("DMET energy will not be accurate.")
        e_dmet = 0.0
        for x in self.get_fragments(active=True, mpi_rank=mpi.rank, sym_parent=None):
            wx = x.symmetry_factor
            e_dmet += wx*x.get_fragment_dmet_energy(version=version, approx_cumulant=approx_cumulant)
        if mpi:
            mpi.world.allreduce(e_dmet)
        version = (version or 1)
        if (version == 2):
            dm1 = self.make_rdm1_demo(ao_basis=True)
            if not approx_cumulant:
                vhf = self.get_veff_for_energy(dm1=dm1, with_exxdiv=False)
            elif (int(approx_cumulant) == 1):
                dm1 = 2*np.asarray(dm1) - self.mf.make_rdm1()
                vhf = self.get_veff_for_energy(with_exxdiv=False)
            else:
                raise ValueError
            e_dmet += np.sum(np.asarray(vhf) * dm1)/2

        self.log.debugv("E_elec(DMET)= %s", energy_string(e_dmet))
        return e_dmet / self.ncells

    def get_dmet_energy(self, with_nuc=True, with_exxdiv=True, version=0, approx_cumulant=True):
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
        e_dmet = self.get_dmet_elec_energy(version=version, approx_cumulant=approx_cumulant)
        if with_nuc:
            e_dmet += self.e_nuc
        if with_exxdiv and self.has_exxdiv:
            e_dmet += self.get_exxdiv()[0]
        return e_dmet

    get_corrfunc_mf = log_method()(get_corrfunc_mf)
    get_corrfunc = log_method()(get_corrfunc)

    # Utility
    # -------

    def _check_orthonormal(self, *mo_coeff, mo_name='', crit_tol=1e-2, err_tol=1e-7):
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

    def get_average_cluster_size(self):
        return np.mean([x.cluster.norb_active for x in self.fragments])

    # --- Population analysis
    # -----------------------

    def _get_atom_projectors(self, atoms=None, projection='sao'):
        if atoms is None:
            atoms2 = list(range(self.mol.natm))
            # For supercell systems, we do not want all supercell-atom pairs,
            # but only primitive-cell -- supercell pairs:
            atoms1 = atoms2 if (self.kcell is None) else list(range(self.kcell.natm))
        elif isinstance(atoms[0], (int, np.integer)):
            atoms1 = atoms2 = atoms
        else:
            atoms1, atoms2 = atoms

        # Get atomic projectors:
        projection = projection.lower()
        if projection == 'sao':
            frag = SAO_Fragmentation(self)
        elif projection.replace('+', '').replace('/', '') == 'iaopao':
            frag = IAOPAO_Fragmentation(self)
        else:
            raise ValueError("Invalid projection: %s" % projection)
        frag.kernel()
        ovlp = self.get_ovlp()
        projectors = {}
        cs = spinalg.dot(spinalg.transpose(self.mo_coeff), ovlp)
        for atom in sorted(set(atoms1).union(atoms2)):
            name, indices = frag.get_atomic_fragment_indices(atom)
            c_atom = frag.get_frag_coeff(indices)
            r = spinalg.dot(cs, c_atom)
            projectors[atom] = spinalg.dot(r, spinalg.transpose(r))

        return atoms1, atoms2, projectors

    def get_lo_coeff(self, local_orbitals='lowdin', minao='auto'):
        if local_orbitals.lower() == 'lowdin':
            # Avoid pre_orth_ao step!
            #self.c_lo = c_lo = pyscf.lo.orth_ao(self.mol, 'lowdin')
            #self.c_lo = c_lo = pyscf.lo.orth_ao(self.mol, 'meta-lowdin', pre_orth_ao=None)
            return self.get_ovlp_power(power=-0.5)
        elif local_orbitals.lower() == 'iao+pao':
            return IAOPAO_Fragmentation(self, minao=minao).get_coeff()
        raise ValueError("Unknown local orbitals: %r" % local_orbitals)

    def pop_analysis(self, dm1, mo_coeff=None, local_orbitals='lowdin', minao='auto', write=True, filename=None, filemode='a',
            orbital_resolved=False, mpi_rank=0):
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
            self.write_population(pop, filename=filename, filemode=filemode, orbital_resolved=orbital_resolved)
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

    def write_population(self, pop, filename=None, filemode='a', orbital_resolved=False):
        charges, spins = self.get_atomic_charges(pop)
        if orbital_resolved:
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
            if orbital_resolved:
                aos = aoslices[atom]
                for ao in range(aos[0], aos[1]):
                    label = aolabels[ao]
                    if np.ndim(pop[0]) == 1:
                        write("    %4d %-16s= % 11.8f  % 11.8f" % (ao, label, pop[0][ao], pop[1][ao]))
                    else:
                        write("    %4d %-16s= % 11.8f" % (ao, label, pop[ao]))
        if filename is not None:
            f.close()

    def _check_fragment_nelectron(self):
        nelec_frags = sum([f.sym_factor*f.nelectron for f in self.fragments])
        nelec = self.mol.nelectron
        self.log.info("Number of electrons over all fragments= %.8f , system= %.8f", nelec_frags, nelec)
        if abs(nelec_frags - nelec) > 1e-6:
            self.log.warning("Number of electrons over all fragments not equal to the system's number of electrons.")
        return nelec_frags

    # --- Fragmentation methods

    def sao_fragmentation(self, **kwargs):
        """Initialize the quantum embedding method for the use of SAO (Lowdin-AO) fragments."""
        return SAO_Fragmentation(self, **kwargs)

    def site_fragmentation(self, **kwargs):
        """Initialize the quantum embedding method for the use of site fragments."""
        return Site_Fragmentation(self, **kwargs)

    def iao_fragmentation(self, minao='auto', **kwargs):
        """Initialize the quantum embedding method for the use of IAO fragments.

        Parameters
        ----------
        minao: str, optional
            IAO reference basis set. Default: 'auto'
        """
        return IAO_Fragmentation(self, minao=minao, **kwargs)

    def iaopao_fragmentation(self, minao='auto', **kwargs):
        """Initialize the quantum embedding method for the use of IAO+PAO fragments.

        Parameters
        ----------
        minao: str, optional
            IAO reference basis set. Default: 'auto'
        """
        return IAOPAO_Fragmentation(self, minao=minao, **kwargs)

    def cas_fragmentation(self, **kwargs):
        """Initialize the quantum embedding method for the use of site fragments."""
        return CAS_Fragmentation(self, **kwargs)

    def _check_fragmentation(self, complete_occupied=True, complete_virtual=True, tol=1e-7):
        """Check if union of fragment spaces is orthonormal and complete."""
        if self.spinsym == 'restricted':
            nspin = 1
            tspin = lambda x, s : x
            nelec = self.mol.nelectron
        elif self.spinsym == 'unrestricted':
            nspin = 2
            tspin = lambda x, s : x[s]
            nelec = self.mol.nelec
        ovlp = self.get_ovlp()
        dm1 = self.mf.make_rdm1()
        for s in range(nspin):
            nmo_s = tspin(self.nmo, s)
            nelec_s = tspin(nelec, s)
            c_frags = np.hstack([tspin(x.c_frag, s) for x in self.fragments])
            nfrags = c_frags.shape[-1]
            csc = dot(c_frags.T, ovlp, c_frags)
            if not np.allclose(csc, np.eye(nfrags), rtol=0, atol=tol):
                self.log.debug("Non-orthogonal error= %.3e", abs(csc - np.eye(nfrags)).max())
                return False
            if complete_occupied and complete_virtual:
                if (nfrags != nmo_s):
                    return False
            elif complete_occupied or complete_virtual:
                cs = np.dot(c_frags.T, ovlp)
                ne = einsum('ia,ab,ib->', cs, tspin(dm1, s), cs)
                if complete_occupied and (abs(ne - nelec_s) > tol):
                    return False
                if complete_virtual and (abs((nfrags-ne) - (nmo_s-nelec_s)) > tol):
                    return False
        return True

    def has_orthonormal_fragmentation(self, **kwargs):
        """Check if union of fragment spaces is orthonormal."""
        return self._check_fragmentation(complete_occupied=False, complete_virtual=False, **kwargs)

    def has_complete_fragmentation(self, **kwargs):
        """Check if union of fragment spaces is orthonormal and complete."""
        return self._check_fragmentation(complete_occupied=True, complete_virtual=True, **kwargs)

    def has_complete_occupied_fragmentation(self, **kwargs):
        """Check if union of fragment spaces is orthonormal and complete in the occupied space."""
        return self._check_fragmentation(complete_occupied=True, complete_virtual=False, **kwargs)

    def has_complete_virtual_fragmentation(self, **kwargs):
        """Check if union of fragment spaces is orthonormal and complete in the virtual space."""
        return self._check_fragmentation(complete_occupied=False, complete_virtual=True, **kwargs)

    def require_complete_fragmentation(self, message=None, incl_virtual=True, **kwargs):
        if incl_virtual:
            complete = self.has_complete_fragmentation(**kwargs)
        else:
            complete = self.has_complete_occupied_fragmentation(**kwargs)
        if complete:
            return
        if message:
            message = ' %s' % message
        else:
            message = ''
        self.log.error("Fragmentation is not orthogonal and complete.%s", message)

    # --- Reset

    def _reset_fragments(self, *args, **kwargs):
        for fx in self.fragments:
            fx.reset(*args, **kwargs)

    def _reset(self):
        self.e_corr = None
        self.converged = False

    def reset(self, *args, **kwargs):
        self._reset()
        self._reset_fragments(*args, **kwargs)

    # --- Mean-field updates

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

    # --- Decorators
    # These replace the qemb.kernel method!

    def optimize_chempot(self, cpt_init=0.0, dm1func=None, dm1kwds=None, robust=False):

        if dm1func is None:
            dm1func = self.make_rdm1_demo
        if dm1kwds is None:
            dm1kwds = {}

        kernel_orig = self.kernel
        iters = []
        result = None

        def func(cpt, *args, **kwargs):
            nonlocal iters, result
            self.opts.global_frag_chempot = cpt
            result = kernel_orig(*args, **kwargs)
            dm1 = dm1func(**dm1kwds)
            if self.is_rhf:
                ne = np.trace(dm1)
            else:
                ne = np.trace(dm1[0]) + np.trace(dm1[1])
            err = (ne - self.mol.nelectron)
            iters.append((cpt, err, self.converged, self.e_tot))
            return err

        bisect = ChempotBisection(func, cpt_init=cpt_init, robust=robust, log=self.log)

        def kernel(self, *args, **kwargs):
            nonlocal iters, result
            cpt = bisect.kernel(*args, **kwargs)
            # Print info:
            self.log.info("Chemical potential optimization")
            self.log.info("-------------------------------")
            self.log.info("  Iteration   Chemical potential   N(elec) error  Converged         Total Energy")
            for i, (cpt, err, conv, etot) in enumerate(iters):
                self.log.info("  %9d  %19s  %+14.8f  %9r  %19s",
                        i+1, energy_string(cpt), err, conv, energy_string(etot))
            if not bisect.converged:
                self.log.error('Chemical potential not found!')
            return result
        self.kernel = kernel.__get__(self)

    def pdmet_scmf(self, *args, **kwargs):
        """Decorator for p-DMET."""
        self.with_scmf = PDMET(self, *args, **kwargs)
        self.kernel = self.with_scmf.kernel.__get__(self)

    def brueckner_scmf(self, *args, **kwargs):
        """Decorator for Brueckner-DMET."""
        self.with_scmf = Brueckner(self, *args, **kwargs)
        self.kernel = self.with_scmf.kernel.__get__(self)
