# --- Standard
import dataclasses
import functools
from typing import Optional, Union
# --- External
import numpy as np
# --- Internal
import vayesta
from vayesta.core.util import *
from vayesta.core.qemb import Embedding
from vayesta.core.fragmentation import SAO_Fragmentation
from vayesta.core.fragmentation import IAOPAO_Fragmentation
from vayesta.mpi import mpi
# --- Package
from . import helper
from .fragment import Fragment
from .amplitudes import get_global_t1_rhf
from .amplitudes import get_global_t2_rhf
from .rdm import make_rdm1_ccsd
from .rdm import make_rdm1_ccsd_global_wf
from .rdm import make_rdm2_ccsd_global_wf
from .rdm import make_rdm1_ccsd_proj_lambda
from .rdm import make_rdm2_ccsd_proj_lambda
from .icmp2 import get_intercluster_mp2_energy_rhf


@dataclasses.dataclass
class Options(Embedding.Options):
    """Options for EWF calculations."""
    # --- Fragment settings
    iao_minao : str = 'auto'            # Minimal basis for IAOs
    # --- Bath settings
    bath_options: dict = Embedding.Options.change_dict_defaults('bath_options',
            bathtype='mp2', threshold=1e-8)
    #ewdmet_max_order: int = 1
    # If multiple bno thresholds are to be calculated, we can project integrals and amplitudes from a previous larger cluster:
    project_eris: bool = False          # Project ERIs from a pervious larger cluster (corresponding to larger eta), can result in a loss of accuracy especially for large basis sets!
    project_init_guess: bool = True     # Project converted T1,T2 amplitudes from a previous larger cluster
    energy_functional: str = 'wf'
    # Calculation modes
    calc_e_wf_corr: bool = True
    calc_e_dm_corr: bool = False
    # --- Solver settings
    t_as_lambda: bool = None            # If True, use T-amplitudes inplace of Lambda-amplitudes
    store_wf_type: str = None           # If set, fragment WFs will be converted to the respective type, before storing them
    # Counterpoise correction of BSSE
    bsse_correction: bool = True
    bsse_rmax: float = 5.0              # In Angstrom
    nelectron_target: int = None
    # --- Couple embedding problems (currently only CCSD)
    sc_mode: int = 0
    coupled_iterations: bool = False
    # --- Debugging
    _debug_wf: str = None


class EWF(Embedding):

    Fragment = Fragment
    Options = Options

    def __init__(self, mf, solver='CCSD', log=None, **kwargs):
        t0 = timer()
        super().__init__(mf, solver=solver, log=log, **kwargs)

        # Logging
        with self.log.indent():
            # Options
            self.log.info("Parameters of %s:", self.__class__.__name__)
            self.log.info(break_into_lines(str(self.opts), newline='\n    '))
            self.log.info("Time for %s setup: %s", self.__class__.__name__, time_string(timer()-t0))

    def __repr__(self):
        keys = ['mf', 'solver']
        fmt = ('%s(' + len(keys)*'%s: %r, ')[:-2] + ')'
        values = [self.__dict__[k] for k in keys]
        return fmt % (self.__class__.__name__, *[x for y in zip(keys, values) for x in y])

    def _reset(self, **kwargs):
        super()._reset(**kwargs)
        # TODO: Redo self-consistencies
        self.iteration = 0
        self._make_rdm1_ccsd_global_wf_cached.cache_clear()

    # Default fragmentation
    def fragmentation(self, *args, **kwargs):
        return self.iao_fragmentation(*args, **kwargs)

    def tailor_all_fragments(self):
        for x in self.fragments:
            for y in self.fragments:
                if (x == y):
                    continue
                x.add_tailor_fragment(y)

    def kernel(self):
        """Run EWF."""
        t_start = timer()

        # Automatic fragmentation
        if len(self.fragments) == 0:
            self.log.debug("No fragments found. Adding all atomic IAO fragments.")
            with self.fragmentation() as frag:
                frag.add_all_atomic_fragments()
        self._check_fragment_nelectron()

        # Debug: calculate exact WF
        if self.opts._debug_wf is not None:
            self._debug_get_wf(self.opts._debug_wf)

        # --- Create bath and clusters
        if mpi:
            mpi.world.Barrier()
        self.log.info("")
        self.log.info("MAKING BATH AND CLUSTERS")
        self.log.info("========================")
        fragments = self.get_fragments(active=True, sym_parent=None, mpi_rank=mpi.rank)
        fragdict = {f.id: f for f in fragments}
        with log_time(self.log.timing, "Total time for bath and clusters: %s"):
            for x in fragments:
                if x._results is not None:
                    self.log.debug("Resetting %s" % x)
                    x.reset()
                msg = "Making bath and clusters for %s%s" % (x, (" on MPI process %d" % mpi.rank) if mpi else "")
                self.log.info(msg)
                self.log.info(len(msg)*"-")
                with self.log.indent():
                    if x._dmet_bath is None:
                        # Make own bath:
                        if x.flags.bath_parent_fragment_id is None:
                            x.make_bath()
                        # Copy bath (DMET, occupied, virtual) from other fragment:
                        else:
                            bath_parent = fragdict[x.flags.bath_parent_fragment_id]
                            for attr in ('_dmet_bath', '_bath_factory_occ', '_bath_factory_vir'):
                                setattr(x, attr, getattr(bath_parent, attr))
                    if x._cluster is None:
                        x.make_cluster()
            if mpi:
                mpi.world.Barrier()

        if mpi:
            with log_time(self.log.timing, "Time for MPI communication of clusters: %s"):
                self.communicate_clusters()

        # --- Screened Coulomb interaction
        if any(x.opts.screening is not None for x in fragments):
            self.log.info("")
            self.log.info("SCREENING INTERACTIONS")
            self.log.info("======================")
            with log_time(self.log.timing, "Time for screened interations: %s"):
                self.build_screened_eris()

        # --- Loop over fragments with no symmetry parent and with own MPI rank
        self.log.info("")
        self.log.info("RUNNING SOLVERS")
        self.log.info("===============")
        with log_time(self.log.timing, "Total time for solvers: %s"):
            # Split fragments in auxiliary and regular, solve auxiliary fragments first
            fragments_aux = [x for x in fragments if x.opts.auxiliary]
            fragments_reg = [x for x in fragments if not x.opts.auxiliary]
            for frags in [fragments_aux, fragments_reg]:
                for x in frags:
                    msg = "Solving %s%s" % (x, (" on MPI process %d" % mpi.rank) if mpi else "")
                    self.log.info(msg)
                    self.log.info(len(msg)*"-")
                    with self.log.indent():
                        x.kernel()
                if mpi:
                    mpi.world.Barrier()

        if self.solver.lower() == 'dump':
            self.log.output("Clusters dumped to file '%s'", self.opts.solver_options['dumpfile'])
            return

        # --- Check convergence of fragments
        conv = self._all_converged(fragments)
        if not conv:
            self.log.error("Some fragments did not converge!")
        self.converged = conv

        # --- Evaluate correlation energy and log information
        self.e_corr = self.get_e_corr()
        self.log.output('E(MF)=   %s', energy_string(self.e_mf))
        self.log.output('E(corr)= %s', energy_string(self.e_corr))
        self.log.output('E(tot)=  %s', energy_string(self.e_tot))
        self.log.info("Total wall time:  %s", time_string(timer()-t_start))
        return self.e_tot

    def _all_converged(self, fragments):
        conv = True
        for fx in fragments:
            conv = (conv and fx.results.converged)
        if mpi:
            conv = mpi.world.allreduce(conv, op=mpi.MPI.LAND)
        return conv

    # --- CC Amplitudes
    # -----------------

    # T-amplitudes
    get_global_t1 = get_global_t1_rhf
    get_global_t2 = get_global_t2_rhf

    # Lambda-amplitudes
    def get_global_l1(self, *args, t_as_lambda=None, **kwargs):
        get_lambda = True if not t_as_lambda else False
        return self.get_global_t1(*args, get_lambda=get_lambda, **kwargs)
    def get_global_l2(self, *args, t_as_lambda=None, **kwargs):
        get_lambda = True if not t_as_lambda else False
        return self.get_global_t2(*args, get_lambda=get_lambda, **kwargs)

    def t1_diagnostic(self, warntol=0.02):
        # Per cluster
        for fx in self.get_fragments(active=True, mpi_rank=mpi.rank):
            wfx = fx.results.wf.as_ccsd()
            t1 = wfx.t1
            nelec = 2*t1.shape[0]
            t1diag = np.linalg.norm(t1) / np.sqrt(nelec)
            if t1diag >= warntol:
                self.log.warning("T1 diagnostic for %-20s %.5f", str(f)+':', t1diag)
            else:
                self.log.info("T1 diagnostic for %-20s %.5f", str(f)+':', t1diag)
        # Global
        t1 = self.get_global_t1(mpi_target=0)
        if mpi.is_master:
            nelec = 2*t1.shape[0]
            t1diag = np.linalg.norm(t1) / np.sqrt(nelec)
            if t1diag >= warntol:
                self.log.warning("Global T1 diagnostic: %.5f", t1diag)
            else:
                self.log.info("Global T1 diagnostic: %.5f", t1diag)

    # --- Density-matrices
    # --------------------

    # Defaults

    def make_rdm1(self, *args, **kwargs):
        if self.solver.lower() == 'ccsd':
            return self._make_rdm1_ccsd_global_wf(*args, **kwargs)
        if self.solver.lower() == 'mp2':
            return self._make_rdm1_mp2_global_wf(*args, **kwargs)
        if self.solver.lower() == 'fci':
            return self.make_rdm1_demo(*args, **kwargs)
        raise NotImplementedError("make_rdm1 for solver '%s'" % self.solver)

    def make_rdm2(self, *args, **kwargs):
        if self.solver.lower() == 'ccsd':
            return self._make_rdm2_ccsd_proj_lambda(*args, **kwargs)
            #return self._make_rdm2_ccsd(*args, **kwargs)
        if self.solver.lower() == 'mp2':
            return self._make_rdm2_ccsd_proj_lambda(*args, t_as_lambda=True, **kwargs)
        raise NotImplementedError("make_rdm2 for solver '%s'" % self.solver)

    # DM1
    @log_method()
    def _make_rdm1_mp2(self, *args, **kwargs):
        return make_rdm1_ccsd(self, *args, mp2=True, **kwargs)

    @log_method()
    def _make_rdm1_ccsd(self, *args, **kwargs):
        return make_rdm1_ccsd(self, *args, mp2=False, **kwargs)

    @log_method()
    def _make_rdm1_ccsd_global_wf(self, *args, ao_basis=False, with_mf=True, **kwargs):
        dm1 = self._make_rdm1_ccsd_global_wf_cached(*args, **kwargs)
        if with_mf:
            dm1[np.diag_indices(self.nocc)] += 2
        if ao_basis:
            dm1 = dot(self.mo_coeff, dm1, self.mo_coeff.T)
        return dm1

    @cache(copy=True)
    def _make_rdm1_ccsd_global_wf_cached(self, *args, **kwargs):
        return make_rdm1_ccsd_global_wf(self, *args, **kwargs)

    def _make_rdm1_mp2_global_wf(self, *args, **kwargs):
        return self._make_rdm1_ccsd_global_wf(*args, t_as_lambda=True, with_t1=False, **kwargs)

    @log_method()
    def _make_rdm1_ccsd_proj_lambda(self, *args, **kwargs):
        return make_rdm1_ccsd_proj_lambda(self, *args, **kwargs)

    # DM2

    @log_method()
    def _make_rdm2_ccsd_global_wf(self, *args, **kwargs):
        return make_rdm2_ccsd_global_wf(self, *args, **kwargs)

    @log_method()
    def _make_rdm2_ccsd_proj_lambda(self, *args, **kwargs):
        return make_rdm2_ccsd_proj_lambda(self, *args, **kwargs)

    # --- Energy
    # ----------

    # Correlation

    def get_e_corr(self, functional=None, **kwargs):
        functional = (functional or self.opts.energy_functional)
        if functional == 'projected':
            # TODO: print deprecation message
            functional = 'wf'
        if functional == 'wf':
            return self.get_wf_corr_energy(**kwargs)
        if functional == 'dm-t2only':
            return self.get_dm_corr_energy(t_as_lambda=True, **kwargs)
        if functional == 'dm':
            return self.get_dm_corr_energy(**kwargs)
        raise ValueError("Unknown energy functional: '%s'" % functional)

    @mpi.with_allreduce()
    def get_wf_corr_energy(self):
        e_corr = 0.0
        # Only loop over fragments of own MPI rank
        for x in self.get_fragments(contributes=True, sym_parent=None, mpi_rank=mpi.rank):
            if x.results.e_corr is not None:
                ex = x.results.e_corr
            else:
                wf = x.results.wf.as_cisd(c0=1.0)
                px = x.get_overlap('frag|cluster-occ')
                wf = wf.project(px)
                es, ed, ex = x.get_fragment_energy(wf.c1, wf.c2)
                self.log.debug("%20s:  E(S)= %s  E(D)= %s  E(tot)= %s", x, energy_string(es), energy_string(ed), energy_string(ex))
            e_corr += x.symmetry_factor * ex
        return e_corr/self.ncells

    def get_proj_corr_energy(self):
        """TODO: deprecate in favor of get_wf_corr_energy."""
        return self.get_wf_corr_energy()

    def get_dm_corr_energy(self, dm1='global-wf', dm2='projected-lambda', t_as_lambda=None, with_exxdiv=None):
        e1 = self.get_dm_corr_energy_e1(dm1=dm1, t_as_lambda=None, with_exxdiv=None)
        e2 = self.get_dm_corr_energy_e2(dm2=dm2, t_as_lambda=t_as_lambda)
        e_corr = (e1 + e2)
        self.log.debug("Ecorr(1)= %s  Ecorr(2)= %s  Ecorr= %s", *map(energy_string, (e1, e2, e_corr)))
        return e_corr

    def get_dm_corr_energy_e1(self, dm1='global-wf', t_as_lambda=None, with_exxdiv=None):
        # Correlation energy due to changes in 1-DM and non-cumulant 2-DM:
        if dm1 == 'global-wf':
            dm1 = self._make_rdm1_ccsd_global_wf(with_mf=False, t_as_lambda=t_as_lambda, ao_basis=True)
        elif dm1 == '2p1l':
            dm1 = self._make_rdm1_ccsd(with_mf=False, t_as_lambda=t_as_lambda, ao_basis=True)
        elif dm1 == '1p1l':
            dm1 = self._make_rdm1_ccsd_1p1l(with_mf=False, t_as_lambda=t_as_lambda, ao_basis=True)
        else:
            raise ValueError

        if with_exxdiv is None:
            if self.has_exxdiv:
                with_exxdiv = np.all([x.solver == 'MP2' for x in self.fragments])
                any_mp2 = np.any([x.solver == 'MP2' for x in self.fragments])
                any_not_mp2 = np.any([x.solver != 'MP2' for x in self.fragments])
                if (any_mp2 and any_not_mp2):
                    self.log.warning("Both MP2 and not MP2 solvers detected - unclear usage of exxdiv!")
            else:
                with_exxdiv = False

        fock = self.get_fock_for_energy(with_exxdiv=with_exxdiv)
        e1 = np.sum(fock*dm1)
        return e1/self.ncells

    @mpi.with_allreduce()
    def get_dm_corr_energy_e2(self, dm2='projected-lambda', t_as_lambda=None):
        """Correlation energy due to cumulant"""
        if t_as_lambda is None:
            t_as_lambda = self.opts.t_as_lambda
        if dm2 == 'global-wf':
            dm2 = self._make_rdm2_ccsd_global_wf(t_as_lambda=t_as_lambda, with_dm1=False)
            # TODO: AO basis, late DF contraction
            if self.spinsym == 'restricted':
                g = self.get_eris_array(self.mo_coeff)
                e2 = einsum('pqrs,pqrs', g, dm2)/2
            else:
                dm2aa, dm2ab, dm2bb = dm2
                gaa, gab, gbb = self.get_eris_array_uhf(self.mo_coeff)
                e2 = (einsum('pqrs,pqrs', gaa, dm2aa)/2
                    + einsum('pqrs,pqrs', gbb, dm2bb)/2
                    + einsum('pqrs,pqrs', gab, dm2ab))
        elif dm2 == 'projected-lambda':
            e2 = 0.0
            for x in self.get_fragments(contributes=True, sym_parent=None, mpi_rank=mpi.rank):
                ex = x.results.e_corr_dm2cumulant
                if ex is None or (t_as_lambda is not None and (t_as_lambda != x.opts.t_as_lambda)):
                    ex = x.make_fragment_dm2cumulant_energy(t_as_lambda=t_as_lambda)
                e2 += x.symmetry_factor * x.sym_factor * ex
        else:
            raise ValueError("Unknown value for dm2: '%s'" % dm2)
        return e2/self.ncells

    def get_ccsd_corr_energy(self, full_wf=False):
        """Get projected correlation energy from partitioned CCSD WF.

        This is the projected (T1, T2) energy expression, instead of the
        projected (C1, C2) expression used in PRX (the differences are very small).

        For testing only, UHF and MPI not implemented"""
        t0 = timer()
        t1 = self.get_global_t1()

        # E(singles)
        fock = self.get_fock_for_energy(with_exxdiv=False)
        fov =  dot(self.mo_coeff_occ.T, fock, self.mo_coeff_vir)
        e_singles = 2*np.sum(fov*t1)

        # E(doubles)
        if full_wf:
            c2 = (self.get_global_t2() + einsum('ia,jb->ijab', t1, t1))
            mos = (self.mo_coeff_occ, self.mo_coeff_vir, self.mo_coeff_vir, self.mo_coeff_occ)
            eris = self.get_eris_array(mos)
            e_doubles = (2*einsum('ijab,iabj', c2, eris)
                         - einsum('ijab,ibaj', c2, eris))
        else:
            e_doubles = 0.0
            for x in self.get_fragments(contributes=True, sym_parent=None, mpi_rank=mpi.rank):
                pwf = x.results.pwf.as_ccsd()
                ro = x.get_overlap('mo-occ|cluster-occ')
                rv = x.get_overlap('mo-vir|cluster-vir')

                t1x = dot(ro.T, t1, rv) # N(frag) * N^2
                c2x = pwf.t2 + einsum('ia,jb->ijab', pwf.t1, t1x)

                noccx = x.cluster.nocc_active
                nvirx = x.cluster.nvir_active

                eris = x.hamil.get_eris_bare("ovvo")

                px = x.get_overlap('frag|cluster-occ')
                eris = einsum('xi,iabj->xabj', px, eris)
                wx = x.symmetry_factor * x.sym_factor
                e_doubles += wx*(2*einsum('ijab,iabj', c2x, eris)
                                 - einsum('ijab,ibaj', c2x, eris))
            if mpi:
                e_doubles = mpi.world.allreduce(e_doubles)

        self.log.timing("Time for E(CCSD)= %s", time_string(timer()-t0))
        e_corr = (e_singles + e_doubles)
        return e_corr / self.ncells

    # Total energy

    @property
    def e_tot(self):
        """Total energy."""
        return self.e_mf + self.e_corr

    def get_wf_energy(self, *args, **kwargs):
        e_corr = self.get_wf_corr_energy(*args, **kwargs)
        return self.e_mf + e_corr

    def get_proj_energy(self, *args, **kwargs):
        """TODO: deprecate in favor of get_wf_energy."""
        return self.get_wf_energy(*args, **kwargs)

    def get_dm_energy(self, *args, **kwargs):
        e_corr = self.get_dm_corr_energy(*args, **kwargs)
        return self.e_mf + e_corr

    def get_ccsd_energy(self, full_wf=False):
        return self.e_mf + self.get_ccsd_corr_energy(full_wf=full_wf)

    # --- Energy corrections

    @mpi.with_allreduce()
    @log_method()
    def get_fbc_energy(self, occupied=True, virtual=True):
        """Get finite-bath correction (FBC) energy.

        This correction consists of two independent contributions, one due to the finite occupied,
        and one due to the finite virtual space.

        The virtual correction for a given fragment x is calculated as
        "E(MP2)[occ=D,vir=F] - E(MP2)[occ=D,vir=C]", where D is the DMET cluster space,
        F is the full space, and C is the full cluster space. For the occupied correction,
        occ and vir spaces are swapped. Fragments which do not have a BNO bath are skipped.

        Parameters
        ----------
        occupied: bool, optional
            If True, the FBC energy from the occupied space is included. Default: True.
        virtual: bool, optional
            If True, the FBC energy from the virtual space is included. Default: True.

        Returns
        -------
        e_fbc: float
            Finite bath correction (FBC) energy.
        """
        if not (occupied or virtual):
            raise ValueError

        e_fbc = 0.0
        # Only loop over fragments of own MPI rank
        for fx in self.get_fragments(contributes=True, sym_parent=None, flags=dict(is_envelop=True), mpi_rank=mpi.rank):
            ex = 0
            if occupied:
                get_fbc = getattr(fx._bath_factory_occ, 'get_finite_bath_correction', False)
                if get_fbc:
                    ex += get_fbc(fx.cluster.c_active_occ, fx.cluster.c_frozen_occ)
                else:
                    self.log.warning("%s does not have occupied BNOs - skipping fragment for FBC energy.", fx)
            if virtual:
                get_fbc = getattr(fx._bath_factory_vir, 'get_finite_bath_correction', False)
                if get_fbc:
                    ex += get_fbc(fx.cluster.c_active_vir, fx.cluster.c_frozen_vir)
                else:
                    self.log.warning("%s does not have virtual BNOs - skipping fragment for FBC energy.", fx)
            self.log.debug("FBC from %-30s  dE= %s", fx, energy_string(ex))
            e_fbc += fx.symmetry_factor * ex
        e_fbc /= self.ncells
        self.log.debug("E(FBC)= %s", energy_string(e_fbc))
        return e_fbc

    @log_method()
    def get_intercluster_mp2_energy(self, *args, **kwargs):
        return get_intercluster_mp2_energy_rhf(self, *args, **kwargs)

    # --- Deprecated

    @deprecated(replacement='_get_atom_projectors')
    def _get_atomic_coeffs(self, atoms=None, projection='sao'):
        if atoms is None:
            atoms = list(range(self.mol.natm))
        natom = len(atoms)
        projection = projection.lower()
        if projection == 'sao':
            frag = SAO_Fragmentation(self)
        elif projection.replace('+', '').replace('/', '') == 'iaopao':
            frag = IAOPAO_Fragmentation(self)
        else:
            raise ValueError("Invalid projection: %s" % projection)
        frag.kernel()
        c_atom = []
        for atom in atoms:
            name, indices = frag.get_atomic_fragment_indices(atom)
            c_atom.append(frag.get_frag_coeff(indices))
        return c_atom

    @deprecated(replacement='get_corrfunc_mf')
    def get_atomic_ssz_mf(self, dm1=None, atoms=None, projection='sao'):
        return self.get_corrfunc_mf('Sz,Sz', dm1=dm1, atoms=atoms, projection=projection)

    @deprecated(replacement='get_corrfunc')
    def get_atomic_ssz(self, dm1=None, dm2=None, atoms=None, projection='sao', dm2_with_dm1=None):
        return self.get_corrfunc('Sz,Sz', dm1=dm1, dm2=dm2, atoms=atoms, projection=projection, dm2_with_dm1=dm2_with_dm1)

    def _get_dm_energy_old(self, global_dm1=True, global_dm2=False):
        """Calculate total energy from reduced density-matrices.

        RHF ONLY!

        Parameters
        ----------
        global_dm1 : bool
            Use 1DM calculated from global amplitutes if True, otherwise use in cluster approximation. Default: True
        global_dm2 : bool
            Use 2DM calculated from global amplitutes if True, otherwise use in cluster approximation. Default: False

        Returns
        -------
        e_tot : float
        """
        return self.e_mf + self._get_dm_corr_energy_old(global_dm1=global_dm1, global_dm2=global_dm2)

    def _get_dm_corr_energy_old(self, global_dm1=True, global_dm2=False):
        """Calculate correlation energy from reduced density-matrices.

        RHF ONLY!

        Parameters
        ----------
        global_dm1 : bool
            Use 1DM calculated from global amplitutes if True, otherwise use in cluster approximation. Default: True
        global_dm2 : bool
            Use 2DM calculated from global amplitutes if True, otherwise use in cluster approximation. Default: False

        Returns
        -------
        e_corr : float
        """
        t_as_lambda = self.opts.t_as_lambda

        if global_dm1:
            dm1 = self._make_rdm1_ccsd_global_wf(t_as_lambda=t_as_lambda, ao_basis=True, with_mf=False)
        else:
            dm1 = self._make_rdm1_ccsd(t_as_lambda=t_as_lambda, ao_basis=True, with_mf=False)

        # Core Hamiltonian + Non Cumulant 2DM contribution
        e1 = np.sum(self.get_fock_for_energy(with_exxdiv=False) * dm1)

        # Cumulant 2DM contribution
        if global_dm2:
            # Calculate global 2RDM and contract with ERIs
            rdm2 = self._make_rdm2_ccsd_global_wf(t_as_lambda=t_as_lambda, with_dm1=False)
            eris = self.get_eris_array(self.mo_coeff)
            e2 = einsum('pqrs,pqrs', eris, rdm2)/2
        else:
            # Fragment Local 2DM cumulant contribution
            e2 = self.get_dm_corr_energy_e2(t_as_lambda=t_as_lambda) * self.ncells
        e_corr = (e1 + e2) / self.ncells
        return e_corr

    # --- Debugging

    def _debug_get_wf(self, kind):
        if kind == 'random':
            return
        if kind == 'exact':
            if self.solver == 'CCSD':
                import pyscf
                import pyscf.cc
                from vayesta.core.types import WaveFunction
                cc = pyscf.cc.CCSD(self.mf)
                cc.kernel()
                if self.opts.solver_options['solve_lambda']:
                    cc.solve_lambda()
                wf = WaveFunction.from_pyscf(cc)
            else:
                raise NotImplementedError
        else:
            wf = self.opts._debug_wf
        self._debug_wf = wf


REWF = EWF
