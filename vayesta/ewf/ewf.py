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
    energy_functional: str = 'projected'
    # Calculation modes
    calc_e_wf_corr: bool = True
    calc_e_dm_corr: bool = False
    # --- Solver settings
    t_as_lambda: bool = None           # If True, use T-amplitudes inplace of Lambda-amplitudes
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

    def __init__(self, mf, solver='CCSD', bno_threshold=None, bath_type=None, solve_lambda=None, log=None, **kwargs):
        t0 = timer()
        super().__init__(mf, solver=solver, log=log, **kwargs)

        # Backwards support
        if bno_threshold is not None:
            self.log.deprecated("keyword argument bno_threshold is deprecated!")
            self.opts.bath_options = {**self.opts.bath_options, **dict(threshold=bno_threshold)}
        if bath_type is not None:
            self.log.deprecated("keyword argument bath_type is deprecated!")
            self.opts.bath_options = {**self.opts.bath_options, **dict(bathtype=bath_type)}
        if solve_lambda is not None:
            self.log.deprecated("keyword argument solve_lambda is deprecated!")
            self.opts.solver_options = {**self.opts.solver_options, **dict(solve_lambda=solve_lambda)}

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
        self._make_rdm1_ccsd_global_wf.cache_clear()

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
        self.log.info("MAKING CLUSTERS")
        self.log.info("===============")
        with log_time(self.log.timing, "Total time for bath and clusters: %s"):
            for x in self.get_fragments(active=True, sym_parent=None, mpi_rank=mpi.rank):
                if x._results is not None:
                    self.log.debug("Resetting %s" % x)
                    x.reset()
                msg = "Making bath for %s%s" % (x, (" on MPI process %d" % mpi.rank) if mpi else "")
                self.log.info(msg)
                self.log.info(len(msg)*"-")
                with self.log.indent():
                    if x._dmet_bath is None:
                        x.make_bath()
                    if x._cluster is None:
                        x.make_cluster()
            if mpi:
                mpi.world.Barrier()
        if mpi:
            with log_time(self.log.timing, "Time for MPI communication of clusters: %s"):
                self.communicate_clusters()

        # --- Screened Coulomb interaction
        if self.opts.screening is None:
            pass
        elif self.opts.screening == 'rpa':
            self.log.info("")
            self.log.info("SCREENING INTERACTIONS")
            self.log.info("======================")
            with log_time(self.log.timing, "Time for screened interations: %s"):
                self.build_screened_eris()
        else:
            raise ValueError

        # --- Loop over fragments with no symmetry parent and with own MPI rank
        self.log.info("")
        self.log.info("RUNNING SOLVERS")
        self.log.info("===============")
        with log_time(self.log.timing, "Total time for solvers: %s"):
            for x in self.get_fragments(active=True, sym_parent=None, mpi_rank=mpi.rank):
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
        conv = self._all_converged()
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

    def _all_converged(self):
        conv = True
        for fx in self.get_fragments(active=True, sym_parent=None, mpi_rank=mpi.rank):
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

    # --- Bardwards compatibility:
    @deprecated("get_t1 is deprecated - use get_global_t1 instead.")
    def get_t1(self, *args, **kwargs):
        return self.get_global_t1(*args, **kwargs)
    @deprecated("get_t2 is deprecated - use get_global_t2 instead.")
    def get_t2(self, *args, **kwargs):
        return self.get_global_t2(*args, **kwargs)
    @deprecated("get_l1 is deprecated - use get_global_l1 instead.")
    def get_l1(self, *args, **kwargs):
        return self.get_global_l1(*args, **kwargs)
    @deprecated("get_l2 is deprecated - use get_global_l2 instead.")
    def get_l2(self, *args, **kwargs):
        return self.get_global_l2(*args, **kwargs)

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
        self.require_complete_fragmentation("Density-matrices will not be accurate.", incl_virtual=False)
        return make_rdm1_ccsd(self, *args, mp2=True, **kwargs)

    @log_method()
    def _make_rdm1_ccsd(self, *args, **kwargs):
        self.require_complete_fragmentation("Density-matrices will not be accurate.", incl_virtual=False)
        return make_rdm1_ccsd(self, *args, mp2=False, **kwargs)

    @log_method()
    @cache(copy=True)
    def _make_rdm1_ccsd_global_wf(self, *args, **kwargs):
        self.require_complete_fragmentation("Density-matrices will not be accurate.", incl_virtual=False)
        return make_rdm1_ccsd_global_wf(self, *args, **kwargs)

    def _make_rdm1_mp2_global_wf(self, *args, **kwargs):
        self.require_complete_fragmentation("Density-matrices will not be accurate.", incl_virtual=False)
        return self._make_rdm1_ccsd_global_wf(*args, t_as_lambda=True, with_t1=False, **kwargs)

    @log_method()
    def _make_rdm1_ccsd_proj_lambda(self, *args, **kwargs):
        self.require_complete_fragmentation("Density-matrices will not be accurate.", incl_virtual=False)
        return make_rdm1_ccsd_proj_lambda(self, *args, **kwargs)

    # DM2

    @log_method()
    def _make_rdm2_ccsd_global_wf(self, *args, **kwargs):
        self.require_complete_fragmentation("Density-matrices will not be accurate.", incl_virtual=False)
        return make_rdm2_ccsd_global_wf(self, *args, **kwargs)

    @log_method()
    def _make_rdm2_ccsd_proj_lambda(self, *args, **kwargs):
        self.require_complete_fragmentation("Density-matrices will not be accurate.", incl_virtual=False)
        return make_rdm2_ccsd_proj_lambda(self, *args, **kwargs)

    # --- Energy
    # ----------

    # Correlation

    def get_e_corr(self, **kwargs):
        if self.opts.energy_functional == 'projected':
            return self.get_proj_corr_energy(**kwargs)
        if self.opts.energy_functional == 'dm-t2only':
            return self.get_dm_corr_energy(t_as_lambda=True, **kwargs)
        if self.opts.energy_functional == 'dm':
            return self.get_dm_corr_energy(**kwargs)
        raise ValueError("Unknown energy functional: '%s'" % self.opts.energy_functional)

    @mpi.with_allreduce()
    def get_proj_corr_energy(self):
        self.require_complete_fragmentation("Energy will not be accurate.", incl_virtual=False)
        e_corr = 0.0
        # Only loop over fragments of own MPI rank
        for x in self.get_fragments(active=True, sym_parent=None, mpi_rank=mpi.rank):
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

    def get_dm_corr_energy(self, dm1='global-wf', t_as_lambda=None, with_exxdiv=None):
        self.require_complete_fragmentation("Energy will not be accurate.", incl_virtual=False)
        e1 = self.get_dm_corr_energy_e1(dm1=dm1, t_as_lambda=None, with_exxdiv=None)
        e2 = self.get_dm_corr_energy_e2(t_as_lambda=t_as_lambda)
        e_corr = (e1 + e2)
        self.log.debug("Ecorr(1)= %s  Ecorr(2)= %s  Ecorr= %s", *map(energy_string, (e1, e2, e_corr)))
        return e_corr

    def get_dm_corr_energy_e1(self, dm1=None, t_as_lambda=None, with_exxdiv=None):
        # Correlation energy due to changes in 1-DM and non-cumulant 2-DM:
        if dm1 is None or dm1 == 'global-wf':
            dm1 = self._make_rdm1_ccsd_global_wf(with_mf=False, t_as_lambda=t_as_lambda, ao_basis=True)
        elif dm1 == '2p1l':
            dm1 = self._make_rdm1_ccsd(with_mf=False, t_as_lambda=t_as_lambda, ao_basis=True)
        elif dm1 == '1p1l':
            dm1 = self._make_rdm1_ccsd_1p1l(with_mf=False, t_as_lambda=t_as_lambda, ao_basis=True)

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
    def get_dm_corr_energy_e2(self, t_as_lambda=None):
        """Correlation energy due to cumulant"""
        if t_as_lambda is None:
            t_as_lambda = self.opts.t_as_lambda
        e2 = 0.0
        for x in self.get_fragments(active=True, sym_parent=None, mpi_rank=mpi.rank):
            ex = x.results.e_corr_dm2cumulant
            if ex is None or (t_as_lambda is not None and (t_as_lambda != x.opts.t_as_lambda)):
                ex = x.make_fragment_dm2cumulant_energy(t_as_lambda=t_as_lambda)
            e2 += x.symmetry_factor * x.sym_factor * ex
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
            for x in self.get_fragments(active=True, sym_parent=None, mpi_rank=mpi.rank):
                pwf = x.results.pwf.as_ccsd()
                ro = x.get_overlap('mo-occ|cluster-occ')
                rv = x.get_overlap('mo-vir|cluster-vir')

                t1x = dot(ro.T, t1, rv) # N(frag) * N^2
                c2x = pwf.t2 + einsum('ia,jb->ijab', pwf.t1, t1x)

                noccx = x.cluster.nocc_active
                nvirx = x.cluster.nvir_active
                eris = x._eris
                if eris is None:
                    raise NotCalculatedError
                if hasattr(eris, 'ovvo'):
                    eris = eris.ovvo[:]
                elif hasattr(eris, 'ovov'):
                    # MP2 only has eris.ovov - for real integrals we transpose
                    eris = eris.ovov[:].reshape(noccx,nvirx,noccx,nvirx).transpose(0,1,3,2).conj()
                elif eris.shape == (noccx, nvirx, noccx, nvirx):
                    eris = eris.transpose(0,1,3,2)
                else:
                    occ = np.s_[:noccx]
                    vir = np.s_[noccx:]
                    eris = eris[occ,vir,vir,occ]
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

    def get_proj_energy(self):
        e_corr = self.get_proj_corr_energy()
        return self.e_mf + e_corr

    def get_dm_energy(self, dm1='global-wf', t_as_lambda=None, with_exxdiv=None):
        e_corr = self.get_dm_corr_energy(dm1=dm1, t_as_lambda=t_as_lambda, with_exxdiv=with_exxdiv)
        return self.e_mf + e_corr

    def get_ccsd_energy(self, full_wf=False):
        return self.e_mf + self.get_ccsd_corr_energy(full_wf=full_wf)

    # --- Energy corrections

    get_intercluster_mp2_energy = get_intercluster_mp2_energy_rhf

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
        mf = self.mf
        nmo = mf.mo_coeff.shape[1]
        nocc = (mf.mo_occ > 0).sum()

        if global_dm1:
            rdm1 = self._make_rdm1_ccsd_global_wf(t_as_lambda=t_as_lambda)
        else:
            rdm1 = self._make_rdm1_ccsd(t_as_lambda=t_as_lambda)
        rdm1[np.diag_indices(nocc)] -= 2

        # Core Hamiltonian + Non Cumulant 2DM contribution
        e1 = einsum('pi,pq,qj,ij->', self.mo_coeff, self.get_fock_for_energy(with_exxdiv=False), self.mo_coeff, rdm1)

        # Cumulant 2DM contribution
        if global_dm2:
            # Calculate global 2RDM and contract with ERIs
            eri = self.get_eris_array(self.mo_coeff)
            rdm2 = self._make_rdm2_ccsd_global_wf(t_as_lambda=t_as_lambda, with_dm1=False)
            e2 = einsum('pqrs,pqrs', eri, rdm2) * 0.5
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

    # -------------------------------------------------------------------------------------------- #


    # TODO: Reimplement PMO
    #def make_atom_fragment(self, atoms, name=None, check_atoms=True, **kwargs):
    #    """
    #    Parameters
    #    ---------
    #    atoms : list of int/str or int/str
    #        Atom labels of atoms in fragment.
    #    name : str
    #        Name of fragment.
    #    """
    #    # Atoms may be a single atom index/label
    #    #if not isinstance(atoms, (tuple, list, np.ndarray)):
    #    if np.isscalar(atoms):
    #        atoms = [atoms]

    #    # Check if atoms are valid labels of molecule
    #    atom_labels_mol = [self.mol.atom_symbol(atomid) for atomid in range(self.mol.natm)]
    #    if isinstance(atoms[0], str) and check_atoms:
    #        for atom in atoms:
    #            if atom not in atom_labels_mol:
    #                raise ValueError("Atom with label %s not in molecule." % atom)

    #    # Get atom indices/labels
    #    if isinstance(atoms[0], (int, np.integer)):
    #        atom_indices = atoms
    #        atom_labels = [self.mol.atom_symbol(i) for i in atoms]
    #    else:
    #        atom_indices = np.nonzero(np.isin(atom_labels_mol, atoms))[0]
    #        atom_labels = atoms
    #    assert len(atom_indices) == len(atom_labels)

    #    # Generate cluster name if not given
    #    if name is None:
    #        name = ",".join(atom_labels)

    #    # Indices refers to AOs or IAOs, respectively

    #    # Non-orthogonal AOs
    #    if self.opts.fragment_type == "AO":
    #        # Base atom for each AO
    #        ao_atoms = np.asarray([ao[1] for ao in self.mol.ao_labels(None)])
    #        indices = np.nonzero(np.isin(ao_atoms, atoms))[0]
    #        C_local, C_env = self.make_local_ao_orbitals(indices)

    #    # Lowdin orthonalized AOs
    #    elif self.opts.fragment_type == "LAO":
    #        lao_atoms = [lao[1] for lao in self.lao_labels]
    #        indices = np.nonzero(np.isin(lao_atoms, atom_labels))[0]
    #        C_local, C_env = self.make_local_lao_orbitals(indices)

    #    # Orthogonal intrinsic AOs
    #    elif self.opts.fragment_type == "IAO":
    #        iao_atoms = [iao[0] for iao in self.iao_labels]
    #        iao_indices = np.nonzero(np.isin(iao_atoms, atom_indices))[0]
    #        C_local, C_env = self.make_local_iao_orbitals(iao_indices)

    #    # Non-orthogonal intrinsic AOs
    #    elif self.opts.fragment_type == "NonOrth-IAO":
    #        ao_atoms = np.asarray([ao[1] for ao in self.mol.ao_labels(None)])
    #        indices = np.nonzero(np.isin(ao_atoms, atom_labels))[0]
    #        C_local, C_env = self.make_local_nonorth_iao_orbitals(indices, minao=self.opts.iao_minao)

    #    # Projected molecular orbitals
    #    # (AVAS paper)
    #    elif self.opts.fragment_type == "PMO":
    #        #ao_atoms = np.asarray([ao[1] for ao in self.mol.ao_labels(None)])
    #        #indices = np.nonzero(np.isin(ao_atoms, atoms))[0]

    #        # Use atom labels as AO labels
    #        self.log.debug("Making occupied projector.")
    #        Po = self.get_ao_projector(atom_labels, basis=kwargs.pop("basis_proj_occ", None))
    #        self.log.debug("Making virtual projector.")
    #        Pv = self.get_ao_projector(atom_labels, basis=kwargs.pop("basis_proj_vir", None))
    #        self.log.debug("Done.")

    #        o = (self.mo_occ > 0)
    #        v = (self.mo_occ == 0)
    #        C = self.mo_coeff
    #        So = np.linalg.multi_dot((C[:,o].T, Po, C[:,o]))
    #        Sv = np.linalg.multi_dot((C[:,v].T, Pv, C[:,v]))
    #        eo, Vo = np.linalg.eigh(So)
    #        ev, Vv = np.linalg.eigh(Sv)
    #        rev = np.s_[::-1]
    #        eo, Vo = eo[rev], Vo[:,rev]
    #        ev, Vv = ev[rev], Vv[:,rev]
    #        self.log.debug("Non-zero occupied eigenvalues:\n%r", eo[eo>1e-10])
    #        self.log.debug("Non-zero virtual eigenvalues:\n%r", ev[ev>1e-10])
    #        #tol = 1e-8
    #        tol = 0.1
    #        lo = eo > tol
    #        lv = ev > tol
    #        Co = np.dot(C[:,o], Vo)
    #        Cv = np.dot(C[:,v], Vv)
    #        C_local = np.hstack((Co[:,lo], Cv[:,lv]))
    #        C_env = np.hstack((Co[:,~lo], Cv[:,~lv]))
    #        self.log.debug("Number of local orbitals: %d", C_local.shape[-1])
    #        self.log.debug("Number of environment orbitals: %d", C_env.shape[-1])

    #    frag = self.make_fragment(name, C_local, C_env, atoms=atom_indices, **kwargs)

    #    # TEMP
    #    #ao_indices = get_ao_indices_at_atoms(self.mol, atomids)
    #    ao_indices = helper.atom_labels_to_ao_indices(self.mol, atom_labels)
    #    frag.ao_indices = ao_indices

    #    return frag

    #def collect_results(self, *attributes):
    #    """Use MPI to collect results from all fragments."""

    #    #self.log.debug("Collecting attributes %r from all clusters", (attributes,))
    #    fragments = self.fragments

    #    if mpi:
    #        def reduce_fragment(attr, op=mpi.MPI.SUM, root=0):
    #            res = mpi.world.reduce(np.asarray([getattr(f, attr) for f in fragments]), op=op, root=root)
    #            return res
    #    else:
    #        def reduce_fragment(attr):
    #            res = np.asarray([getattr(f, attr) for f in fragments])
    #            return res

    #    results = {}
    #    for attr in attributes:
    #        results[attr] = reduce_fragment(attr)

    #    return results


REWF = EWF
