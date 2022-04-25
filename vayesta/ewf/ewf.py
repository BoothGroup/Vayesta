# --- Standard
import dataclasses
from typing import Union
# --- External
import numpy as np
# --- Internal
import vayesta
from vayesta.core.util import *
from vayesta.core import Embedding
from vayesta.core.mpi import mpi
from vayesta.core.fragmentation import SAO_Fragmentation
from vayesta.core.fragmentation import IAOPAO_Fragmentation
# --- Package
from . import helper
from .fragment import EWFFragment as Fragment
from .amplitudes import get_global_t1_rhf
from .amplitudes import get_global_t2_rhf
from .rdm import make_rdm1_ccsd
from .rdm import make_rdm1_ccsd_old
from .rdm import make_rdm1_ccsd_proj_lambda
from .rdm import make_rdm1_ccsd_2p2l
from .rdm import make_rdm2_ccsd
from .rdm import make_rdm2_ccsd_proj_lambda
from .icmp2 import get_intercluster_mp2_energy_rhf

timer = mpi.timer


@dataclasses.dataclass
class EWFResults:
    bno_threshold: float = None
    cluster_sizes: np.ndarray = None
    e_corr: float = None


VALID_SOLVERS = [None, '', 'MP2', 'CISD', 'CCSD', 'TCCSD', 'CCSD(T)', 'FCI', 'FCI-spin0', 'FCI-spin1']

class EWF(Embedding):

    Fragment = Fragment

    @dataclasses.dataclass
    class Options(Embedding.Options):
        """Options for EWF calculations."""
        # --- Fragment settings
        #fragment_type: str = 'IAO'
        localize_fragment: bool = False     # Perform numerical localization on fragment orbitals
        iao_minao : str = 'auto'            # Minimal basis for IAOs
        # --- Bath settings
        bath_type: str = 'MP2-BNO'
        bno_truncation: str = 'occupation'  # Type of BNO truncation ["occupation", "number", "excited-percent", "electron-percent"]
        bno_threshold: float = 1e-8
        bno_threshold_occ: float = None
        bno_threshold_vir: float = None
        bno_project_t2: bool = False
        ewdmet_max_order: int = 1
        # If multiple bno thresholds are to be calculated, we can project integrals and amplitudes from a previous larger cluster:
        project_eris: bool = False          # Project ERIs from a pervious larger cluster (corresponding to larger eta), can result in a loss of accuracy especially for large basis sets!
        project_init_guess: bool = True     # Project converted T1,T2 amplitudes from a previous larger cluster
        orthogonal_mo_tol: float = False
        # --- Solver settings
        solve_lambda: bool = False          # If True, solve for the Lambda-amplitudes if a CCSD solver is used
        t_as_lambda: bool = False           # If True, use T-amplitudes inplace of Lambda-amplitudes
        dm_with_frozen: bool = False        # Add frozen parts to cluster DMs
        # Energy calculation
        calc_cluster_rdm_energy: bool = False
        # Counterpoise correction of BSSE
        bsse_correction: bool = True
        bsse_rmax: float = 5.0              # In Angstrom
        # -- Self-consistency
        sc_maxiter: int = 30
        sc_energy_tol: float = 1e-6
        sc_mode: int = 0
        nelectron_target: int = None
        # --- Intercluster MP2 energy
        icmp2: bool = True
        icmp2_bno_threshold: float = 1e-8
        # --- Couple embedding problems (currently only CCSD)
        coupled_iterations: bool = False
        strict: bool = False                # Stop if cluster not converged

    def __init__(self, mf, solver='CCSD', options=None, log=None, **kwargs):
        """Embedded wave function (EWF) calculation object.

        Parameters
        ----------
        mf : pyscf.scf object
            Converged mean-field object.
        solver : str, optional
            Solver for embedding problem. Default: 'CCSD'.
        **kwargs :
            See class `Options` for additional options.
        """
        t0 = timer()
        super().__init__(mf, options=options, log=log, **kwargs)

        # Options
        self.log.info("Parameters of %s:", self.__class__.__name__)
        self.log.info(break_into_lines(str(self.opts), newline='\n    '))

        # --- Check input
        if not mf.converged:
            if self.opts.strict:
                raise RuntimeError("Mean-field calculation not converged.")
            else:
                self.log.error("Mean-field calculation not converged.")
        if solver not in VALID_SOLVERS:
            raise ValueError("Unknown solver: %s" % solver)
        self.solver = solver

        self.e_corr = None
        self.log.info("Time for %s setup: %s", self.__class__.__name__, time_string(timer()-t0))

        # TODO: Redo self-consistencies
        self.iteration = 0

    def __repr__(self):
        keys = ['mf', 'solver']
        fmt = ('%s(' + len(keys)*'%s: %r, ')[:-2] + ')'
        values = [self.__dict__[k] for k in keys]
        return fmt % (self.__class__.__name__, *[x for y in zip(keys, values) for x in y])


    #    if self.opts.fragment_type.upper() == "IAO":
    #        #self.C_ao, self.C_env, self.iao_labels = self.make_iao_coeffs(minao=self.opts.iao_minao)
    #        minao=self.opts.iao_minao
    #        self.C_ao, self.C_env = self.make_iao_coeffs(minao=minao)
    #        self.iao_labels = self.get_iao_labels(minao=minao)
    #        # Only for printing:
    #        self.get_iao_occupancy(self.C_ao, minao=minao)
    #        self.ao_labels = self.iao_labels
    #    elif self.opts.fragment_type.upper() == "LOWDIN-AO":
    #        self.C_ao, self.lao_labels = self.make_lowdin_ao()
    #        self.ao_labels = self.lao_labels

    #    locmethod = self.opts.localize_fragment
    #    if locmethod:
    #        self.log.debug("Localize fragment orbitals with %s method", locmethod)

    #        #orbs = {self.ao_labels[i] : self.C_ao[:,i:i+1] for i in range(self.C_ao.shape[-1])}
    #        #orbs = {"A" : self.C_ao}
    #        #create_orbital_file(self.mol, "%s.molden" % self.local_orbital_type, orbs)
    #        coeffs = self.C_ao
    #        names = [("%d-%s-%s-%s" % l).rstrip("-") for l in self.ao_labels]
    #        #create_orbital_file(self.mol, self.local_orbital_type, coeffs, names, directory="fragment")
    #        create_orbital_file(self.mol, self.opts.fragment_type, coeffs, names, directory="fragment", filetype="cube")

    #        t0 = timer()
    #        if locmethod in ("BF", "ER", "PM"):
    #            localizer = getattr(pyscf.lo, locmethod)(self.mol)
    #            localizer.init_guess = None
    #            #localizer.pop_method = "lowdin"
    #            C_loc = localizer.kernel(self.C_ao, verbose=4)
    #        elif locmethod == "LAO":
    #            #centers = [l[0] for l in self.mol.ao_labels(None)]
    #            centers = [l[0] for l in self.ao_labels]
    #            self.log.debug("Atom centers: %r", centers)
    #            C_loc = localize_ao(self.mol, self.C_ao, centers)

    #        #C_loc = locfunc(self.mol).kernel(self.C_ao, verbose=4)
    #        self.log.timing("Time for orbital localization: %s", time_string(timer()-t0))
    #        assert C_loc.shape == self.C_ao.shape
    #        # Check that all orbitals kept their fundamental character
    #        chi = np.einsum("ai,ab,bi->i", self.C_ao, self.get_ovlp(), C_loc)
    #        self.log.info("Diagonal of AO-Loc(AO) overlap: %r", chi)
    #        self.log.info("Smallest value: %.3g" % np.amin(chi))
    #        #assert np.all(chi > 0.5)
    #        self.C_ao = C_loc

    #        #orbs = {"A" : self.C_ao}
    #        #orbs = {self.ao_labels[i] : self.C_ao[:,i:i+1] for i in range(self.C_ao.shape[-1])}
    #        #create_orbital_file(self.mol, "%s-local.molden" % self.local_orbital_type, orbs)
    #        #raise SystemExit()

    #        coeffs = self.C_ao
    #        names = [("%d-%s-%s-%s" % l).rstrip("-") for l in self.ao_labels]
    #        #create_orbital_file(self.mol, self.local_orbital_type, coeffs, names, directory="fragment-localized")
    #        create_orbital_file(self.mol, self.opts.fragment_type, coeffs, names, directory="fragment-localized", filetype="cube")

    def get_init_mo_coeff(self, mo_coeff=None):
        """Orthogonalize insufficiently orthogonal MOs.

        (For example as a result of k2gamma conversion with low cell.precision)
        """
        if mo_coeff is None: mo_coeff = self.mo_coeff
        c = mo_coeff.copy()
        ovlp = self.get_ovlp()
        assert np.all(c.imag == 0), "max|Im(C)|= %.2e" % abs(c.imag).max()
        err = abs(dot(c.T, ovlp, c) - np.eye(c.shape[-1])).max()
        if err > 1e-5:
            self.log.error("Orthogonality error of MOs= %.2e !!!", err)
        else:
            self.log.debug("Orthogonality error of MOs= %.2e", err)
        if self.opts.orthogonal_mo_tol and err > self.opts.orthogonal_mo_tol:
            t0 = timer()
            self.log.info("Orthogonalizing orbitals...")
            c_orth = helper.orthogonalize_mo(c, ovlp)
            change = abs(einsum('ai,ab,bi->i', c_orth, ovlp, c)-1)
            self.log.info("Max. orbital change= %.2e%s", change.max(), " (!!!)" if change.max() > 1e-4 else "")
            self.log.timing("Time for orbital orthogonalization: %s", time_string(timer()-t0))
            c = c_orth
        return c

    def tailor_all_fragments(self):
        for frag in self.fragments:
            for frag2 in frag.loop_fragments(exclude_self=True):
                frag.add_tailor_fragment(frag2)

    def kernel(self, bno_threshold=None):
        """Run EWF.

        Parameters
        ----------
        bno_threshold : float or iterable, optional
            Bath natural orbital threshold. If `None`, self.opts.bno_threshold is used. Default: None.
        """
        # Reset previous results
        self.reset()

        # Automatic fragmentation
        if self.fragmentation is None:
            self.log.info("No fragmentation found. Using IAO fragmentation.")
            self.iao_fragmentation()
        if len(self.fragments) == 0:
            self.log.info("No fragments found. Using all atomic fragments.")
            self.add_all_atomic_fragments()

        self.check_fragment_nelectron()
        if np.ndim(bno_threshold) == 0:
            return self._kernel_single_threshold(bno_threshold=bno_threshold)
        return self._kernel_multiple_thresholds(bno_thresholds=bno_threshold)

    def _kernel_multiple_thresholds(self, bno_thresholds):
        results = []
        for i, bno in enumerate(bno_thresholds):
            self.log.info("Now running BNO threshold= %.2e", bno)
            self.log.info("===================================")

            # Project ERIs for next calculation:
            # TODO
            if i > 0:
                #self.log.debugv("Projecting ERIs onto subspace")
                for x in self.fragments:
                    x._eris = None
                    #x._eris = ao2mo.helper.project_ccsd_eris(x._eris, x.cluster.c_active, x.cluster.nocc_active, ovlp=self.get_ovlp())

            # Store ERIs so they can be reused in the next iteration
            # (only if this is not the last calculation and the next calculation uses a larger or equal threshold)
            store_eris = (i+1 < len(bno_thresholds)) and (bno <= bno_thresholds[i+1])
            # Note that if opts.store_eris has been set to True elsewhere, we do not want to overwrite this,
            # even if store_eris was evaluated as False. For this reason we add `or self.opts.store_eris`.
            with replace_attr(self.opts, store_eris=(store_eris or self.opts.store_eris)):
                res = self._kernel_single_threshold(bno_threshold=bno)
            results.append(res)

        # Output
        for i, bno in enumerate(bno_thresholds):
            self.log.info("BNO threshold= %.2e  E(tot)= %s", bno, energy_string(results[i]))

        return results

    def _kernel_single_threshold(self, bno_threshold=None):
        """Run EWF.

        Parameters
        ----------
        bno_threshold : float, optional
            Bath natural orbital threshold.
        """

        if self.nfrag == 0:
            raise ValueError("No fragments defined for calculation.")
        assert (np.ndim(bno_threshold) == 0)

        if mpi: mpi.world.Barrier()
        t_start = timer()

        # Loop over fragments with no symmetry parent and with own MPI rank
        for frag in self.get_fragments(sym_parent=None, mpi_rank=mpi.rank):

            msg = "Now running %s" % frag
            if mpi:
                msg += " on MPI process %d" % mpi.rank
            self.log.info(msg)
            self.log.info(len(msg)*"-")
            self.log.changeIndentLevel(1)
            frag.kernel(bno_threshold=bno_threshold)
            #res = frag.kernel(bno_threshold=bno_threshold)
            #if not res.converged:
            #    self.log.error("%s is not converged!", frag)
            #else:
            #    self.log.info("%s is done.", frag)
            self.log.changeIndentLevel(-1)

        # Evaluate correlation energy
        self.e_corr = self.get_e_corr()

        self.log.output('E(nuc)=  %s', energy_string(self.mol.energy_nuc()))
        self.log.output('E(MF)=   %s', energy_string(self.e_mf))
        self.log.output('E(corr)= %s', energy_string(self.e_corr))
        self.log.output('E(tot)=  %s', energy_string(self.e_tot))

        self.log.info("Total wall time:  %s", time_string(timer()-t_start))
        return self.e_tot

    # --- CC Amplitudes
    # -----------------

    # T-amplitudes
    get_global_t1 = get_global_t1_rhf
    get_global_t2 = get_global_t2_rhf

    # Lambda-amplitudes
    def get_global_l1(self, *args, **kwargs):
        return self.get_global_t1(*args, get_lambda=True, **kwargs)
    def get_global_l2(self, *args, **kwargs):
        return self.get_global_t2(*args, get_lambda=True, **kwargs)

    def t1_diagnostic(self, warn_tol=0.02):
        # Per cluster
        for f in self.get_fragments(mpi_rank=mpi.rank):
            t1 = f.results.t1
            if t1 is None:
                self.log.error("No T1 amplitudes found for %s.", f)
                continue
            nelec = 2*t1.shape[0]
            t1diag = np.linalg.norm(t1) / np.sqrt(nelec)
            if t1diag > warn_tol:
                self.log.warning("T1 diagnostic for %-20s %.5f", str(f)+':', t1diag)
            else:
                self.log.info("T1 diagnostic for %-20s %.5f", str(f)+':', t1diag)
        # Global
        t1 = self.get_global_t1(mpi_target=0)
        if mpi.is_master:
            nelec = 2*t1.shape[0]
            t1diag = np.linalg.norm(t1) / np.sqrt(nelec)
            if t1diag > warn_tol:
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

    def make_rdm1(self, *args, **kwargs):
        if self.solver.lower() == 'ccsd':
            return self.make_rdm1_ccsd_2p2l(*args, **kwargs)
        if self.solver.lower() == 'mp2':
            return self.make_rdm1_mp2(*args, **kwargs)
        if self.solver.lower() == 'fci':
            return self.make_rdm1_demo(*args, **kwargs)
        raise NotImplementedError("make_rdm1 for solver '%s'" % self.solver)

    def make_rdm1_mp2(self, *args, **kwargs):
        return make_rdm1_ccsd(self, *args, mp2=True, **kwargs)

    def make_rdm1_ccsd(self, *args, **kwargs):
        return make_rdm1_ccsd(self, *args, mp2=False, **kwargs)

    def make_rdm2_ccsd(self, *args, **kwargs):
        return make_rdm2_ccsd(self, *args, **kwargs)

    def make_rdm1_ccsd_old(self, *args, **kwargs):
        return make_rdm1_ccsd_old(self, *args, **kwargs)

    def make_rdm1_ccsd_2p2l(self, *args, **kwargs):
        return make_rdm1_ccsd_2p2l(self, *args, **kwargs)

    def make_rdm1_ccsd_proj_lambda(self, *args, **kwargs):
        return make_rdm1_ccsd_proj_lambda(self, *args, **kwargs)

    def make_rdm2_ccsd_proj_lambda(self, *args, **kwargs):
        return make_rdm2_ccsd_proj_lambda(self, *args, **kwargs)

    # --- Energy
    # ----------

    # Correlation

    @mpi.with_allreduce()
    def get_e_corr(self, fragments=None):
        e_corr = 0.0
        # Only loop over fragments of own MPI rank
        for f in self.get_fragments(fragment_list=fragments, mpi_rank=mpi.rank):
            if f.results.e_corr is None:
                self.log.critical("No fragment E(corr) found for %s! Returning total E(corr)=NaN", f)
                return np.nan
            e_corr += f.results.e_corr
        return e_corr / self.ncells

    def get_dm_corr_energy_2(self, t_as_lambda=None, sym_t2=True):
        e1, e2 = self.get_dm_corr_energy_parts(t_as_lambda=t_as_lambda, sym_t2=sym_t2)
        e_corr = (e1 + e2)
        self.log.debug("Ecorr(1)= %s  Ecorr(2)= %s  Ecorr= %s", *map(energy_string, (e1, e2, e_corr)))
        return e_corr

    def get_dm_corr_energy_parts(self, t_as_lambda=None, sym_t2=True):
        if t_as_lambda is None:
            t_as_lambda = self.opts.t_as_lambda
        # Correlation energy due to changes in 1-DM and non-cumulant 2-DM:
        times = [timer()]
        #dm1 = self.make_rdm1_ccsd(with_mf=False, t_as_lambda=t_as_lambda, ao_basis=True)
        #dm1 = self.make_rdm1_ccsd_old(with_mf=False, t_as_lambda=t_as_lambda, ao_basis=True)
        dm1 = self.make_rdm1_ccsd_2p2l(with_mf=False, t_as_lambda=t_as_lambda, ao_basis=True)
        fock = self.get_fock_for_energy(with_exxdiv=False)
        e1 = np.sum(fock*dm1) / self.ncells
        times.append(timer())
        # Correlation energy due to cumulant:
        e2 = 0.0
        for x in self.get_fragments(sym_parent=None, mpi_rank=mpi.rank):
            wx = x.symmetry_factor * x.sym_factor / self.ncells
            e2 += wx * x.make_fragment_cumulant_energy(t_as_lambda=t_as_lambda, sym_t2=sym_t2)
        if mpi:
            e2 = mpi.world.allreduce(e2)
        times.append(timer())
        self.log.timing("Time for DM energy: T(DM1)= %s  T(DM2)= %s  T(tot)= %s",
                *map(time_string, (times[1]-times[0], times[2]-times[1], times[2]-times[0])))
        return e1, e2

    def get_e_corr_ccsd(self, full_wf=False):
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
            for x in self.get_fragments(sym_parent=None, mpi_rank=mpi.rank):
                pwf = x.results.pwf.as_ccsd()
                ro = x.get_mo2co_occ()
                rv = x.get_mo2co_vir()
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
                eris = einsum('xi,iabj->xabj', x.get_fo2co_occ(), eris)
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

    def get_e_tot_ccsd(self, full_wf=False):
        return self.e_mf + self.get_e_corr_ccsd(full_wf=full_wf)

    def get_dm_energy_2(self, t_as_lambda=None, sym_t2=True):
        e_corr = self.get_dm_corr_energy_2(t_as_lambda=t_as_lambda, sym_t2=sym_t2)
        return self.e_mf + e_corr

    # --- Energy corrections

    get_intercluster_mp2_energy = get_intercluster_mp2_energy_rhf

    # --- Other expectation values

    def get_atomic_ssz(self, atoms=None, dm1=None, dm2=None, projection='sao', full_dm2=False):
        """TODO: MPI"""
        t0 = timer()
        if atoms is None:
            atoms = list(range(self.mol.natm))
        natom = len(atoms)

        if projection == 'sao':
            frag = SAO_Fragmentation(self.mf, self.log)
        elif projection.replace('+', '').replace('/', '') == 'iaopao':
            frag = IAOPAO_Fragmentation(self.mf, self.log)
        else:
            raise NotImplementedError("Projection '%s' not implemented" % projection)
        frag.kernel()

        ovlp = self.get_ovlp()

        c_atom = []
        for atom in atoms:
            name, indices = frag.get_atomic_fragment_indices(atom)
            c_atom.append(frag.get_frag_coeff(indices))

        proj = []
        for i, atom in enumerate(atoms):
            rx = np.dot(ovlp, c_atom[i])
            rx = np.dot(self.mo_coeff.T, rx)
            px = np.dot(rx, rx.T)
            proj.append(px)

        # Fragment dependent projection operator:
        if not full_dm2:
            proj_x = []
            for fx in self.get_fragments():
                tmp = np.dot(fx.cluster.c_active.T, ovlp)
                proj_x.append([])
                for a, atom in enumerate(atoms):
                    rx = np.dot(tmp, c_atom[a])
                    px = np.dot(rx, rx.T)
                    proj_x[-1].append(px)

        ssz = np.zeros((natom, natom))

        # 1-DM contribution:
        if dm1 is None:
            #dm1 = self.make_rdm1_ccsd_proj_lambda()
            dm1 = self.make_rdm1_ccsd_old()
            #dm1 = self.make_rdm1_ccsd()
        for a, atom1 in enumerate(atoms):
            tmp = np.dot(proj[a], dm1)
            for b, atom2 in enumerate(atoms):
                ssz[a,b] = np.sum(tmp*proj[b])/4

        occ = np.s_[:self.nocc]
        occdiag = np.diag_indices(self.nocc)

        # Non-cumulant DM2 contribution:
        # (for full_dm2=True this is included in the 2-DM contribution below)
        if not full_dm2:

            # TEST (FULL DM2):
            #dm1 = dm1.copy()
            #dm1[occdiag] -= 1
            #dm1 /= 2
            #ddm2 = np.zeros(4*[self.nmo])
            #for i in range(self.nocc):
            #    ddm2[:,i,i,:] -= dm1
            #    ddm2[i,:,:,i] -= dm1.T
            #for a, atom1 in enumerate(atoms):
            #    pa = proj[a]
            #    tmp = np.dot(pa, dm1)
            #    for b, atom2 in enumerate(atoms):
            #        pb = proj[b]
            #        ssz[a,b] += einsum('ij,ijkl,kl->', pa, ddm2, pb)/2

            dm1 = dm1.copy()
            dm1[occdiag] -= 1
            dm1 /= 2
            for a, atom1 in enumerate(atoms):
                tmp = np.dot(proj[a], dm1)
                for b, atom2 in enumerate(atoms):
                    #ssz[a,b] -= np.sum(np.dot(tmp, proj[b])[occdiag])  # N_atom^2 * N^3 scaling
                    ssz[a,b] -= np.sum(tmp[occ] * proj[b].T[occ])       # N_atom^2 * N^2 scaling

        # 2-DM contribution:
        if full_dm2:
            if dm2 is None:
                dm2 = self.make_rdm2_ccsd_proj_lambda()
            dm2aa = (dm2 - dm2.transpose(0,3,2,1))/6
            # ddm2 is equal to dm2aa - dm2ab, as
            # dm2ab = (dm2/2 - dm2aa)
            ddm2 = (2*dm2aa - dm2/2)

            for a, atom1 in enumerate(atoms):
                p1 = proj[a]
                #tmp = einsum('ij,ijkl->kl', p1, ddm2)
                tmp = np.tensordot(p1, ddm2)
                for b, atom2 in enumerate(atoms):
                    p2 = proj[b]
                    #ssz[i,j] = einsum('kl,kl->', tmp, p2)/2
                    ssz[a,b] += np.sum(tmp*p2)/2
        else:
            for x, fx in enumerate(self.get_fragments()):
                dm2 = fx.make_fragment_cumulant()
                dm2aa = (dm2 - dm2.transpose(0,3,2,1))/6
                ddm2 = (2*dm2aa - dm2/2)

                for a, atom1 in enumerate(atoms):
                    pa = proj_x[x][a]
                    tmp = np.tensordot(pa, ddm2)
                    for b, atom2 in enumerate(atoms):
                        pb = proj_x[x][b]
                        ssz[a,b] += np.sum(tmp*pb)/2

        self.log.timing("Time for <S_z^2>: %s", time_string(timer()-t0))
        return ssz


    #def get_wf_cisd(self, intermediate_norm=False, c0=None):
    #    c0_target = c0

    #    c0 = 1.0
    #    c1 = np.zeros((self.nocc, self.nvir))
    #    c2 = np.zeros((self.nocc, self.nocc, self.nvir, self.nvir))
    #    ovlp = self.get_ovlp()
    #    # Add fragment WFs in intermediate normalization
    #    for f in self.fragments:
    #        c1f, c2f = f.results.c1/f.results.c0, f.results.c2/f.results.c0
    #        #c1f, c2f = f.results.c1, f.results.c2
    #        c1f = f.project_amplitude_to_fragment(c1f, c_occ=f.c_active_occ)
    #        c2f = f.project_amplitude_to_fragment(c2f, c_occ=f.c_active_occ)
    #        ro = np.linalg.multi_dot((f.c_active_occ.T, ovlp, self.mo_coeff_occ))
    #        rv = np.linalg.multi_dot((f.c_active_vir.T, ovlp, self.mo_coeff_vir))
    #        c1 += einsum('ia,iI,aA->IA', c1f, ro, rv)
    #        #c2f = (c2f + c2f.transpose(1,0,3,2))/2
    #        c2 += einsum('ijab,iI,jJ,aA,bB->IJAB', c2f, ro, ro, rv, rv)

    #    # Symmetrize
    #    c2 = (c2 + c2.transpose(1,0,3,2))/2

    #    # Restore standard normalization
    #    if not intermediate_norm:
    #        #c0 = self.fragments[0].results.c0
    #        norm = np.sqrt(c0**2 + 2*np.dot(c1.flatten(), c1.flatten()) + 2*np.dot(c2.flatten(), c2.flatten())
    #                - einsum('jiab,ijab->', c2, c2))
    #        c0 = c0/norm
    #        c1 /= norm
    #        c2 /= norm
    #        # Check normalization
    #        norm = (c0**2 + 2*np.dot(c1.flatten(), c1.flatten()) + 2*np.dot(c2.flatten(), c2.flatten())
    #                - einsum('jiab,ijab->', c2, c2))
    #        assert np.isclose(norm, 1.0)

    #        if c0_target is not None:
    #            norm12 = (2*np.dot(c1.flatten(), c1.flatten()) + 2*np.dot(c2.flatten(), c2.flatten())
    #                    - einsum('jiab,ijab->', c2, c2))
    #            if norm12 > 1e-10:
    #                print('norm12= %.6e' % norm12)
    #                fac12 = np.sqrt((1.0-c0_target**2)/norm12)
    #                print('fac12= %.6e' % fac12)
    #                c0 = c0_target
    #                c1 *= fac12
    #                c2 *= fac12

    #                # Check normalization
    #                norm = (c0**2 + 2*np.dot(c1.flatten(), c1.flatten()) + 2*np.dot(c2.flatten(), c2.flatten())
    #                        - einsum('jiab,ijab->', c2, c2))
    #                assert np.isclose(norm, 1.0)

    #    return c0, c1, c2

    def get_dm_energy(self, global_dm1=True, global_dm2=False):
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
        return self.e_mf + self.get_dm_corr_energy(global_dm1=global_dm1, global_dm2=global_dm2)

    def get_dm_corr_energy(self, global_dm1=True, global_dm2=False):
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
            rdm1 = make_rdm1_ccsd_old(self, t_as_lambda=t_as_lambda)
        else:
            rdm1 = self.make_rdm1_ccsd(t_as_lambda=t_as_lambda)
        rdm1[np.diag_indices(nocc)] -= 2

        # Core Hamiltonian + Non Cumulant 2DM contribution
        e1 = einsum('pi,pq,qj,ij->', self.mo_coeff, self.get_fock_for_energy(with_exxdiv=False), self.mo_coeff, rdm1)

        # Cumulant 2DM contribution
        if global_dm2:
            # Calculate global 2RDM and contract with ERIs
            eri = self.get_eris_array(self.mo_coeff)
            rdm2 = self.make_rdm2_ccsd(slow=True, t_as_lambda=t_as_lambda, with_dm1=False)
            e2 = einsum('pqrs,pqrs', eri, rdm2) * 0.5
        else:
            # Fragment Local 2DM cumulant contribution
            e2 = sum(f.results.e_rdm2 for f in self.fragments)
        e_corr = (e1 + e2) / self.ncells
        return e_corr


    # -------------------------------------------------------------------------------------------- #

    # TODO: Reimplement
    #def make_local_nonorth_iao_orbitals(self, ao_indices, minao="minao"):
    #    C_occ = self.mo_coeff[:,self.mo_occ>0]
    #    C_ao = pyscf.lo.iao.iao(self.mol, C_occ, minao=minao)

    #    ao_labels = np.asarray(self.mol.ao_labels())[ao_indices]
    #    refmol = pyscf.lo.iao.reference_mol(self.mol, minao=minao)
    #    iao_labels = refmol.ao_labels()
    #    assert len(iao_labels) == C_ao.shape[-1]

    #    loc = np.isin(iao_labels, ao_labels)
    #    self.log.debug("Local NonOrth IAOs: %r", (np.asarray(iao_labels)[loc]).tolist())
    #    nlocal = np.count_nonzero(loc)
    #    self.log.debug("Number of local IAOs=%3d", nlocal)

    #    C_local = C_ao[:,loc]
    #    # Orthogonalize locally
    #    #S = self.mf.get_ovlp()
    #    S = self.get_ovlp()
    #    C_local = pyscf.lo.vec_lowdin(C_local, S)

    #    # Add remaining space
    #    # Transform to MO basis
    #    C_local_mo = np.linalg.multi_dot((self.mo_coeff.T, S, C_local))
    #    # Get eigenvectors of projector into complement
    #    P_local = np.dot(C_local_mo, C_local_mo.T)
    #    norb = self.mo_coeff.shape[-1]
    #    P_env = np.eye(norb) - P_local
    #    e, C = np.linalg.eigh(P_env)
    #    assert np.all(np.logical_or(abs(e) < 1e-10, abs(e)-1 < 1e-10))
    #    mask_env = (e > 1e-10)
    #    assert (np.sum(mask_env) + nlocal == norb)
    #    # Transform back to AO basis
    #    C_env = np.dot(self.mo_coeff, C[:,mask_env])

    #    # Test orthogonality
    #    C = np.hstack((C_local, C_env))
    #    assert np.allclose(C.T.dot(S).dot(C) - np.eye(norb), 0)

    #    return C_local, C_env

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

    #def show_cluster_sizes(self, results, show_largest=True):
    #    self.log.info("Cluster Sizes")
    #    self.log.info("*************")
    #    fmtstr = "  * %3d %-10s  :  active=%4d  frozen=%4d  ( %5.1f %%)"
    #    imax = [0]
    #    for i, frag in enumerate(self.loop()):
    #        nactive = results["nactive"][i]
    #        nfrozen = results["nfrozen"][i]
    #        self.log.info(fmtstr, frag.id, frag.trimmed_name(10), nactive, nfrozen, 100.0*nactive/self.nmo)
    #        if i == 0:
    #            continue
    #        if nactive > results["nactive"][imax[0]]:
    #            imax = [i]
    #        elif nactive == results["nactive"][imax[0]]:
    #            imax.append(i)

    #    if show_largest and self.nfrag > 1:
    #        self.log.info("Largest Cluster")
    #        self.log.info("***************")
    #        for i in imax:
    #            x = self.fragments[i]
    #            nactive = results["nactive"][i]
    #            nfrozen = results["nfrozen"][i]
    #            self.log.info(fmtstr, x.id, x.trimmed_name(10), nactive, nfrozen, 100.0*nactive/self.nmo)


    #def print_results(self, results):
    #    self.show_cluster_sizes(results)

    #    self.log.info("Fragment Energies")
    #    self.log.info("*****************")
    #    self.log.info("CCSD / CCSD+dMP2 / CCSD+dMP2+(T)")
    #    fmtstr = "  * %3d %-10s  :  %+16.8f Ha  %+16.8f Ha  %+16.8f Ha"
    #    for i, frag in enumerate(self.loop()):
    #        e_corr = results["e_corr"][i]
    #        e_pert_t = results["e_pert_t"][i]
    #        e_delta_mp2 = results["e_delta_mp2"][i]
    #        self.log.info(fmtstr, frag.id, frag.trimmed_name(10), e_corr, e_corr+e_delta_mp2, e_corr+e_delta_mp2+e_pert_t)

    #    self.log.info("  * %-14s  :  %+16.8f Ha  %+16.8f Ha  %+16.8f Ha", "total", self.e_corr, self.e_corr+self.e_delta_mp2, self.e_corr+self.e_delta_mp2+self.e_pert_t)
    #    self.log.info("E(corr)= %+16.8f Ha", self.e_corr)
    #    self.log.info("E(tot)=  %+16.8f Ha", self.e_tot)

    #def print_results(self, results):
    #    self.log.info("Energies")
    #    self.log.info("========")
    #    fmt = "%-20s %+16.8f Ha"
    #    for i, frag in enumerate(self.loop()):
    #        e_corr = results["e_corr"][i]
    #        self.log.output(fmt, 'E(corr)[' + frag.trimmed_name() + ']=', e_corr)
    #    self.log.output(fmt, 'E(corr)=', self.e_corr)
    #    self.log.output(fmt, 'E(MF)=', self.e_mf)
    #    self.log.output(fmt, 'E(nuc)=', self.mol.energy_nuc())
    #    self.log.output(fmt, 'E(tot)=', self.e_tot)

    #def print_results(self, results):
    #    self.log.info("Energies")
    #    self.log.info("========")
    #    fmt = "%-20s %s"
    #    for i, frag in enumerate(self.loop()):
    #        e_corr = results["e_corr"][i]
    #        self.log.output('E(corr)[' + frag.trimmed_name() + ']= %s', energy_string(e_corr))
    #    self.log.output('E(corr)= %s', energy_string(self.e_corr))
    #    self.log.output('E(MF)=   %s', energy_string(self.e_mf))
    #    self.log.output('E(nuc)=  %s', energy_string(self.mol.energy_nuc()))
    #    self.log.output('E(tot)=  %s', energy_string(self.e_tot))

    #def get_energies(self):
    #    """Get total energy."""
    #    return [(self.e_mf + r.e_corr) for r in self.results]

    #def get_cluster_sizes(self)
    #    sizes = np.zeros((self.nfrag, self.ncalc), dtype=np.int)
    #    for i, frag in enumerate(self.loop()):
    #        sizes[i] = frag.n_active
    #    return sizes

    #def print_clusters(self):
    #    """Print fragments of calculations."""
    #    self.log.info("%3s  %20s  %8s  %4s", "ID", "Name", "Solver", "Size")
    #    for frag in self.loop():
    #        self.log.info("%3d  %20s  %8s  %4d", frag.id, frag.name, frag.solver, frag.size)


    #def get_delta_mp2_correction(self, exchange=True):
    #    """(ia|L)(L|j'b') energy."""

    #    self.log.debug("Intracluster MP2 energies:")
    #    ovlp = self.get_ovlp()
    #    e_dmp2 = 0.0
    #    for x in self.get_fragments():
    #        c_occ_x = x.bath.dmet_bath.c_cluster_occ
    #        c_vir_x = self.mo_coeff_vir
    #        lx, lx_neg = self.get_cderi((c_occ_x, c_vir_x))
    #        eix = x.get_fragment_mo_energy(c_occ_x)
    #        eax = x.get_fragment_mo_energy(c_vir_x)
    #        eia_x = (eix[:,None] - eax[None,:])

    #        gijab = einsum('Lia,Ljb->ijab', lx, lx) # N - N^3
    #        if lx_neg is not None:
    #            gijab -= einsum('Lia,Ljb->ijab', lx_neg, lx_neg)
    #        eijab = (eia_x[:,None,:,None] + eia_x[None,:,None,:])
    #        t2 = (gijab / eijab)

    #        px = dot(x.c_proj.T, ovlp, c_occ_x)

    #        evir_d = 2*einsum('xi,ijab,xk,kjab->', px, t2, px, gijab)
    #        e_dmp2 += evir_d
    #        if exchange:
    #            evir_x = - einsum('xi,ijab,xk,kjba->', px, t2, px, gijab)
    #            e_dmp2 += evir_x
    #        else:
    #            evir_x = 0.0

    #        estr = energy_string
    #        self.log.debug("  %12s:  direct= %s  exchange= %s  total= %s", x.id_name, estr(evir_d), estr(evir_x), estr(evir_d + evir_x))

    #        # Double counting
    #        c_vir_x = x.cluster.c_active_vir
    #        lx, lx_neg = self.get_cderi((c_occ_x, c_vir_x))
    #        eax = x.get_fragment_mo_energy(c_vir_x)
    #        eia_x = (eix[:,None] - eax[None,:])

    #        gijab = einsum('Lia,Ljb->ijab', lx, lx) # N - N^3
    #        if lx_neg is not None:
    #            gijab -= einsum('Lia,Ljb->ijab', lx_neg, lx_neg)
    #        eijab = (eia_x[:,None,:,None] + eia_x[None,:,None,:])
    #        t2 = (gijab / eijab)

    #        edc_d = 2*einsum('xi,ijab,xk,kjab->', px, t2, px, gijab)
    #        e_dmp2 -= edc_d
    #        if exchange:
    #            edc_x = - einsum('xi,ijab,xk,kjba->', px, t2, px, gijab)
    #            e_dmp2 -= edc_x
    #        else:
    #            edc_x = 0.0

    #        estr = energy_string
    #        self.log.debug("DC:  %12s:  direct= %s  exchange= %s  total= %s", x.id_name, estr(edc_d), estr(edc_x), estr(edc_d + edc_x))

    #    return e_dmp2

    #def get_delta_mp2_correction_occ(self, exchange=True):
    #    """(ia|L)(L|j'b') energy."""

    #    self.log.debug("Intracluster MP2 energies:")
    #    ovlp = self.get_ovlp()
    #    e_dmp2 = 0.0
    #    for x in self.get_fragments():
    #        c_occ_x = self.mo_coeff_occ
    #        c_vir_x = x.bath.dmet_bath.c_cluster_vir
    #        lx, lx_neg = self.get_cderi((c_occ_x, c_vir_x))
    #        eix = x.get_fragment_mo_energy(c_occ_x)
    #        eax = x.get_fragment_mo_energy(c_vir_x)
    #        eia_x = (eix[:,None] - eax[None,:])

    #        gijab = einsum('Lia,Ljb->ijab', lx, lx) # N - N^3
    #        if lx_neg is not None:
    #            gijab -= einsum('Lia,Ljb->ijab', lx_neg, lx_neg)
    #        eijab = (eia_x[:,None,:,None] + eia_x[None,:,None,:])
    #        t2 = (gijab / eijab)

    #        px = dot(x.c_proj.T, ovlp, c_occ_x)

    #        evir_d = 2*einsum('xi,ijab,xk,kjab->', px, t2, px, gijab)
    #        e_dmp2 += evir_d
    #        if exchange:
    #            evir_x = - einsum('xi,ijab,xk,kjba->', px, t2, px, gijab)
    #            e_dmp2 += evir_x
    #        else:
    #            evir_x = 0.0

    #        estr = energy_string
    #        self.log.debug("  %12s:  direct= %s  exchange= %s  total= %s", x.id_name, estr(evir_d), estr(evir_x), estr(evir_d + evir_x))

    #        # Double counting
    #        c_occ_x = x.cluster.c_active_occ
    #        lx, lx_neg = self.get_cderi((c_occ_x, c_vir_x))
    #        eix = x.get_fragment_mo_energy(c_occ_x)
    #        eia_x = (eix[:,None] - eax[None,:])

    #        px = dot(x.c_proj.T, ovlp, c_occ_x)

    #        gijab = einsum('Lia,Ljb->ijab', lx, lx) # N - N^3
    #        if lx_neg is not None:
    #            gijab -= einsum('Lia,Ljb->ijab', lx_neg, lx_neg)
    #        eijab = (eia_x[:,None,:,None] + eia_x[None,:,None,:])
    #        t2 = (gijab / eijab)

    #        edc_d = 2*einsum('xi,ijab,xk,kjab->', px, t2, px, gijab)
    #        e_dmp2 -= edc_d
    #        if exchange:
    #            edc_x = - einsum('xi,ijab,xk,kjba->', px, t2, px, gijab)
    #            e_dmp2 -= edc_x
    #        else:
    #            edc_x = 0.0

    #        estr = energy_string
    #        self.log.debug("DC:  %12s:  direct= %s  exchange= %s  total= %s", x.id_name, estr(edc_d), estr(edc_x), estr(edc_d + edc_x))

    #    return e_dmp2

    #def get_intracluster_mp2_correction(self, exchange=True):
    #    """(ia|L)(L|j'b') energy."""

    #    self.log.debug("Intracluster MP2 energies:")
    #    ovlp = self.get_ovlp()
    #    e_dmp2 = 0.0
    #    for x in self.get_fragments():
    #        c_occ_x = x.bath.dmet_bath.c_cluster_occ
    #        c_vir_x = self.mo_coeff_vir
    #        lx, lx_neg = self.get_cderi((c_occ_x, c_vir_x))
    #        eix = x.get_fragment_mo_energy(c_occ_x)
    #        eax = x.get_fragment_mo_energy(c_vir_x)
    #        eia_x = (eix[:,None] - eax[None,:])

    #        gijab = einsum('Lia,Ljb->ijab', lx, lx) # N - N^3
    #        if lx_neg is not None:
    #            gijab -= einsum('Lia,Ljb->ijab', lx_neg, lx_neg)
    #        eijab = (eia_x[:,None,:,None] + eia_x[None,:,None,:])
    #        t2 = (gijab / eijab)

    #        px = dot(x.c_proj.T, ovlp, c_occ_x)

    #        # vir minus active vir
    #        pvirenv = dot(x.cluster.c_active_vir.T, ovlp, c_vir_x)
    #        pvirenv = np.eye(pvirenv.shape[-1]) - dot(pvirenv.T, pvirenv)

    #        # Virtual
    #        evir_d = 2*einsum('xi,ijAb,xk,kjab,aA->', px, t2, px, gijab, pvirenv)
    #        e_dmp2 += evir_d
    #        if exchange:
    #            evir_x = - einsum('xi,ijAb,xk,kjba,aA->', px, t2, px, gijab, pvirenv)
    #            e_dmp2 += evir_x
    #        else:
    #            evir_x = 0.0

    #        estr = energy_string
    #        self.log.debug("  %12s:  direct= %s  exchange= %s  total= %s", x.id_name, estr(evir_d), estr(evir_x), estr(evir_d + evir_x))

    #    return e_dmp2


REWF = EWF
