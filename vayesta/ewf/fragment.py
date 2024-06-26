# Standard libaries
import dataclasses
import typing
from typing import Optional, List

# External libaries
import numpy as np

# Internal libaries
import pyscf
import pyscf.cc

# Local modules
import vayesta
from vayesta.core.util import deprecated, dot, einsum, energy_string, getattr_recursive, hstack, log_method, log_time
from vayesta.core.qemb import Fragment as BaseFragment
from vayesta.core.fragmentation import IAO_Fragmentation
from vayesta.core.types import RFCI_WaveFunction, RCCSDTQ_WaveFunction, UCCSDTQ_WaveFunction, RDM_WaveFunction, RRDM_WaveFunction, URDM_WaveFunction
from vayesta.core.bath import DMET_Bath
from vayesta.mpi import mpi

from vayesta.ewf import ewf


# Get MPI rank of fragment
get_fragment_mpi_rank = lambda *args: args[0].mpi_rank


@dataclasses.dataclass
class Options(BaseFragment.Options):
    # Inherited from Embedding
    # ------------------------
    t_as_lambda: bool = None  # If True, use T-amplitudes inplace of Lambda-amplitudes
    bsse_correction: bool = None
    bsse_rmax: float = None
    sc_mode: int = None
    nelectron_target: float = (
        None  # If set, adjust bath chemical potential until electron number in fragment equals nelectron_target
    )
    nelectron_target_atol: float = 1e-6
    nelectron_target_rtol: float = 1e-6
    # Calculation modes
    calc_e_wf_corr: bool = None
    calc_e_dm_corr: bool = None
    store_wf_type: str = None  # If set, fragment WFs will be converted to the respective type, before storing them
    # Fragment specific
    # -----------------
    wf_factor: Optional[int] = None
    # TODO: move these:
    # CAS methods
    c_cas_occ: np.ndarray = None
    c_cas_vir: np.ndarray = None
    # --- Solver options
    # "TCCSD-solver":
    tcc_fci_opts: dict = dataclasses.field(default_factory=dict)
    # --- Couple embedding problems (currently only CCSD and MPI)
    # coupled_iterations: bool = None # Now accessible through solver=coupledCCSD setting.


class Fragment(BaseFragment):
    Options = Options

    @dataclasses.dataclass
    class Flags(BaseFragment.Flags):
        # Tailoring and external correction of CCSD
        external_corrections: Optional[List[typing.Any]] = dataclasses.field(default_factory=list)
        # Whether to perform additional checks on external corrections
        test_extcorr: bool = False

    @dataclasses.dataclass
    class Results(BaseFragment.Results):
        e_corr_dm2cumulant: float = None
        n_active: int = None
        ip_energy: np.ndarray = None
        ea_energy: np.ndarray = None
        gf_moments: tuple = None
        se_static: np.ndarray = None
        se_moments: tuple = None
        callback_results: dict = None
        
        @property
        def dm1(self):
            """Cluster 1DM"""
            return self.wf.make_rdm1()

        @property
        def dm2(self):
            """Cluster 2DM"""
            return self.wf.make_rdm2()


    def __init__(self, *args, **kwargs):
        """
        Parameters
        ----------
        base : EWF
            Base EWF object.
        fid : int
            Unique ID of fragment.
        name :
            Name of fragment.
        """
        super().__init__(*args, **kwargs)
        # For self-consistent mode
        self.solver_results = None

    def _reset(self, *args, **kwargs):
        super()._reset(*args, **kwargs)
        # Need to unset these so can be regenerated each iteration.
        self.opts.c_cas_occ = self.opts.c_cas_vir = None

    def set_cas(self, iaos=None, c_occ=None, c_vir=None, minao="auto", dmet_threshold=None):
        """Set complete active space for tailored CCSD and active-space CC methods."""
        if dmet_threshold is None:
            dmet_threshold = 2 * self.opts.bath_options["dmet_threshold"]
        if iaos is not None:
            # Create new IAO fragmentation
            frag = IAO_Fragmentation(self.base, minao=minao)
            frag.kernel()
            # Get IAO and environment coefficients from fragmentation
            indices = frag.get_orbital_fragment_indices(iaos)[1]
            c_iao = frag.get_frag_coeff(indices)
            c_env = frag.get_env_coeff(indices)
            bath = DMET_Bath(self, dmet_threshold=dmet_threshold)
            c_dmet = bath.make_dmet_bath(c_env)[0]
            tol = self.opts.bath_options["occupation_tolerance"]
            c_iao_occ, c_iao_vir = self.diagonalize_cluster_dm(c_iao, c_dmet, tol=2 * tol)
        else:
            c_iao_occ = c_iao_vir = None

        c_cas_occ = hstack(c_occ, c_iao_occ)
        c_cas_vir = hstack(c_vir, c_iao_vir)
        self.opts.c_cas_occ = c_cas_occ
        self.opts.c_cas_vir = c_cas_vir
        return c_cas_occ, c_cas_vir

    def add_external_corrections(
        self, fragments, correction_type="tailor", projectors=1, test_extcorr=False, low_level_coul=True
    ):
        """Add tailoring or external correction from other fragment solutions to CCSD solver.

        Parameters
        ----------
        fragments: list
            List of solved or auxiliary fragments, used for the correction.
        correction_type: str, optional
            Type of correction:
                'tailor': replace CCSD T1 and T2 amplitudes with FCI amplitudes.
                'delta-tailor': Add the difference of FCI and CCSD T1 and T2 amplitudes
                'external': externally correct CCSD T1 and T2 amplitudes from FCI T3 and T4 amplitudes.
            Default: 'tailor'.
        projectors: int, optional
            Maximum number of projections applied to the occupied dimensions of the amplitude corrections.
            Default: 1.
        test_extcorr: bool, optional
            Whether to perform additional checks on the external corrections.
        low_level_coul: bool, optional
            This is an option specific to the 'external' correction.
            If True, then the T3V term is contracted with integrals spanning the 'low-level' (i.e. CCSD) solver, i.e. the cluster being constrained.
            If False, then the T3V term is contracted with the integrals in the 'high-level' (i.e. FCI) solver, i.e. the cluster providing the constraints.
            In general, there should be a slight speed increase, and slight loss of accuracy for the low_level_coul=False option, but in practice, we find only
            minor differences.
            Default: True
        """
        if correction_type not in ("tailor", "delta-tailor", "external"):
            raise ValueError
        if self.solver == "CCSD":
            # Automatically update this cluster to use external correction solver.
            self.solver = "extCCSD"
            self.check_solver(self.solver)
        if self.solver != "extCCSD":
            raise RuntimeError
        if (not low_level_coul) and correction_type != "external":
            raise ValueError(
                "low_level_coul optional argument only meaningful with 'external' correction of fragments."
            )
        if np.any([(getattr_recursive(f, "results.wf", None) is None and not f.opts.auxiliary) for f in fragments]):
            raise ValueError(
                "Fragments for external correction need to be already solved or defined as auxiliary fragments."
            )
        self.flags.external_corrections.extend([(f.id, correction_type, projectors, low_level_coul) for f in fragments])
        self.flags.test_extcorr = test_extcorr

    def clear_external_corrections(self):
        """Remove all tailoring or external correction which were added via add_external_corrections."""
        self.flags.external_corrections = []
        self.flags.test_extcorr = False

    def get_init_guess(self, init_guess, solver, cluster):
        # FIXME
        return {}

    def kernel(self, solver=None, init_guess=None):
        solver = solver or self.solver
        self.check_solver(solver)
        if self.cluster is None:
            raise RuntimeError
        cluster = self.cluster

        if solver == "HF":
            return None

        init_guess = self.get_init_guess(init_guess, solver, cluster)

        # Create solver object
        cluster_solver = self.get_solver(solver)
        # Calculate cluster energy at the level of RPA.
        e_corr_rpa = self.get_local_rpa_correction(cluster_solver.hamil)
        # --- Chemical potential
        cpt_frag = self.base.opts.global_frag_chempot
        if self.opts.nelectron_target is not None:
            cluster_solver.optimize_cpt(
                self.opts.nelectron_target,
                c_frag=self.c_proj,
                atol=self.opts.nelectron_target_atol,
                rtol=self.opts.nelectron_target_rtol,
            )
        elif cpt_frag:
            # Add chemical potential to fragment space
            r = self.get_overlap("cluster|frag")
            if self.base.is_rhf:
                p_frag = np.dot(r, r.T)
                cluster_solver.v_ext = cpt_frag * p_frag
            else:
                p_frag = (np.dot(r[0], r[0].T), np.dot(r[1], r[1].T))
                cluster_solver.v_ext = (cpt_frag * p_frag[0], cpt_frag * p_frag[1])

        # --- Coupled fragments.
        # TODO rework this functionality to combine with external corrections/tailoring.
        if solver == "coupledCCSD":
            if not mpi:
                raise RuntimeError("coupled_iterations requires MPI.")
            if len(self.base.fragments) != len(mpi):
                raise RuntimeError("coupled_iterations requires as many MPI processes as there are fragments.")
            cluster_solver.set_coupled_fragments(self.base.fragments)

        # Normal solver
        if not self.base.opts._debug_wf:
            with log_time(self.log.info, ("Time for %s solver:" % solver) + " %s"):
                cluster_solver.kernel()

        # Special debug "solver"
        else:
            if self.base.opts._debug_wf == "random":
                cluster_solver._debug_random_wf()
            else:
                cluster_solver._debug_exact_wf(self.base._debug_wf)

        if solver.lower() == "dump":
            return

        wf = cluster_solver.wf
        # Multiply WF by factor [optional]
        if self.opts.wf_factor is not None:
            wf.multiply(self.opts.wf_factor)
        # Convert WF to different type [optional]
        if self.opts.store_wf_type is not None:
            wf = getattr(wf, "as_%s" % self.opts.store_wf_type.lower())()
        # ---Make T-projected WF
        pwf = wf
        # Projection of FCI wave function is not implemented - convert to CISD
        if isinstance(wf, RFCI_WaveFunction):
            pwf = wf.as_cisd()
        # Projection of CCSDTQ wave function is not implemented - convert to CCSD
        elif isinstance(wf, (RCCSDTQ_WaveFunction, UCCSDTQ_WaveFunction)):
            pwf = wf.as_ccsd()
        if isinstance(wf, (RRDM_WaveFunction, URDM_WaveFunction)):
            proj = self.get_overlap("cluster|frag")
            proj = proj @ proj.T
        else:
            proj = self.get_overlap("proj|cluster-occ")
        pwf = pwf.project(proj, inplace=False)

        # Moments

        gf_moments = cluster_solver.gf_hole_moments, cluster_solver.gf_particle_moments
        se_static = cluster_solver.se_static
        se_moments = cluster_solver.se_hole_moments, cluster_solver.se_particle_moments
        callback_results = cluster_solver.callback_results if solver.lower() == "callback" else None
        # --- Add to results data class
        self._results = results = self.Results(
            fid=self.id,
            n_active=cluster.norb_active,
            converged=cluster_solver.converged,
            wf=wf,
            pwf=pwf,
            gf_moments=gf_moments,
            se_static=se_static,
            se_moments=se_moments,
            e_corr_rpa=e_corr_rpa,
            callback_results=callback_results,
        )

        self.hamil = cluster_solver.hamil

        # --- Correlation energy contributions
        if self.opts.calc_e_wf_corr and not isinstance(wf, (RRDM_WaveFunction, URDM_WaveFunction)):
            ci = wf.as_cisd(c0=1.0)

            ci = ci.project(proj)
            es, ed, results.e_corr = self.get_fragment_energy(ci.c1, ci.c2, hamil=self.hamil)
            self.log.debug(
                "E(S)= %s  E(D)= %s  E(tot)= %s", energy_string(es), energy_string(ed), energy_string(results.e_corr)
            )
        if self.opts.calc_e_dm_corr:
            results.e_corr_dm2cumulant = self.make_fragment_dm2cumulant_energy(hamil=self.hamil)
        return results

    def get_solver_options(self, solver):
        # TODO: fix this mess...
        # Use those values from solver_options, which are not None
        # (conv_tol, max_cycle, solve_lambda,...)
        solver_opts = {key: val for (key, val) in self.opts.solver_options.items() if val is not None}
        pass_through = []
        if "CCSD" in solver.upper():
            pass_through += ["sc_mode", "dm_with_frozen"]
        for attr in pass_through:
            self.log.debugv("Passing fragment option %s to solver.", attr)
            solver_opts[attr] = getattr(self.opts, attr)

        has_actspace = (
            (solver == "TCCSD")
            or ("CCSDt'" in solver)
            or ("CCSDt" in solver)
            or (solver.upper() == "EBCC" and self.opts.solver_options["ansatz"] in ["CCSDt", "CCSDt'"])
        )
        if has_actspace:
            # Set CAS orbitals
            if self.opts.c_cas_occ is None:
                self.log.warning("Occupied CAS orbitals not set. Setting to occupied DMET cluster orbitals.")
                self.opts.c_cas_occ = self._dmet_bath.c_cluster_occ
            if self.opts.c_cas_vir is None:
                self.log.warning("Virtual CAS orbitals not set. Setting to virtual DMET cluster orbitals.")
                self.opts.c_cas_vir = self._dmet_bath.c_cluster_vir
            solver_opts["c_cas_occ"] = self.opts.c_cas_occ
            solver_opts["c_cas_vir"] = self.opts.c_cas_vir
            if solver == "TCCSD":
                solver_opts["tcc_fci_opts"] = self.opts.tcc_fci_opts
        elif solver.upper() == "DUMP":
            solver_opts["filename"] = self.opts.solver_options["dumpfile"]
        if solver.upper() == 'CALLBACK':
            solver_opts["callback"] = self.opts.solver_options["callback"]
        solver_opts["external_corrections"] = self.flags.external_corrections
        solver_opts["test_extcorr"] = self.flags.test_extcorr
        return solver_opts

    # --- Expectation values
    # ----------------------

    # --- Energies

    def get_fragment_energy(self, c1, c2, hamil=None, fock=None, c2ba_order="ba", axis1="fragment"):
        """Calculate fragment correlation energy contribution from projected C1, C2.

        Parameters
        ----------
        c1 : (n(occ-CO), n(vir-CO)) array
            Fragment projected C1-amplitudes.
        c2 : (n(occ-CO), n(occ-CO), n(vir-CO), n(vir-CO)) array
            Fragment projected C2-amplitudes.
        hamil : ClusterHamiltonian object.
            Object representing cluster hamiltonian, possibly including cached ERIs.
        fock : (n(AO), n(AO)) array, optional
            Fock matrix in AO representation. If None, self.base.get_fock_for_energy()
            is used. Default: None.

        Returns
        -------
        e_singles : float
            Fragment correlation energy contribution from single excitations.
        e_doubles : float
            Fragment correlation energy contribution from double excitations.
        e_corr : float
            Total fragment correlation energy contribution.
        """
        if axis1 == "fragment":
            px = self.get_overlap("proj|cluster-occ")

        # --- Singles energy (zero for HF-reference)
        if c1 is not None:
            if fock is None:
                fock = self.base.get_fock_for_energy()
            fov = dot(self.cluster.c_active_occ.T, fock, self.cluster.c_active_vir)
            if axis1 == "fragment":
                e_singles = 2 * einsum("ia,xi,xa->", fov, px, c1)
            else:
                e_singles = 2 * np.sum(fov * c1)
        else:
            e_singles = 0
        # --- Doubles energy
        if hamil is None:
            hamil = self.hamil
        # This automatically either slices a stored ERI tensor or calculates it on the fly.
        g_ovvo = hamil.get_eris_bare(block="ovvo")

        if axis1 == "fragment":
            e_doubles = 2 * einsum("xi,xjab,iabj", px, c2, g_ovvo) - einsum("xi,xjab,ibaj", px, c2, g_ovvo)
        else:
            e_doubles = 2 * einsum("ijab,iabj", c2, g_ovvo) - einsum("ijab,ibaj", c2, g_ovvo)

        e_singles = self.sym_factor * e_singles
        e_doubles = self.sym_factor * e_doubles
        e_corr = e_singles + e_doubles
        return e_singles, e_doubles, e_corr

    # --- Density-matrices

    def _ccsd_amplitudes_for_dm(self, t_as_lambda=False, sym_t2=True):
        wf = self.results.wf.as_ccsd()
        t1, t2 = wf.t1, wf.t2
        pwf = self.results.pwf.restore(sym=sym_t2).as_ccsd()
        t1x, t2x = pwf.t1, pwf.t2
        # Lambda amplitudes
        if t_as_lambda:
            l1, l2 = t1, t2
            l1x, l2x = t1x, t2x
        else:
            l1, l2 = wf.l1, wf.l2
            l1x, l2x = pwf.l1, pwf.l2
        return t1, t2, l1, l2, t1x, t2x, l1x, l2x

    def _get_projected_gamma1_intermediates(self, t_as_lambda=False, sym_t2=True):
        """Intermediates for 1-DM, projected in Lambda-amplitudes and linear T-term."""
        t1, t2, l1, l2, t1x, t2x, l1x, l2x = self._ccsd_amplitudes_for_dm(t_as_lambda=t_as_lambda, sym_t2=sym_t2)
        doo, dov, dvo, dvv = pyscf.cc.ccsd_rdm._gamma1_intermediates(None, t1, t2, l1x, l2x)
        # Correction for term without Lambda amplitude:
        dvo += (t1x - t1).T
        d1 = (doo, dov, dvo, dvv)
        return d1

    def _get_projected_gamma2_intermediates(self, t_as_lambda=False, sym_t2=True):
        """Intermediates for 2-DM, projected in Lambda-amplitudes and linear T-term."""
        t1, t2, l1, l2, t1x, t2x, l1x, l2x = self._ccsd_amplitudes_for_dm(t_as_lambda=t_as_lambda, sym_t2=sym_t2)
        cc = self.mf  # Only attributes stdout, verbose, and max_memory are needed, just use mean-field object
        dovov, *d2rest = pyscf.cc.ccsd_rdm._gamma2_intermediates(cc, t1, t2, l1x, l2x)
        # Correct D2[ovov] part (first element of d2 tuple)
        dtau = (t2x - t2) + einsum("ia,jb->ijab", (t1x - t1), t1)
        dovov += dtau.transpose(0, 2, 1, 3)
        dovov -= dtau.transpose(0, 3, 1, 2) / 2
        d2 = (dovov, *d2rest)
        return d2

    def make_fragment_dm1(self, t_as_lambda=False, sym_t2=True):
        """Currently CCSD only.

        Without mean-field contribution!"""
        d1 = self._get_projected_gamma1_intermediates(t_as_lambda=t_as_lambda, sym_t2=sym_t2)
        dm1 = pyscf.cc.ccsd_rdm._make_rdm1(None, d1, with_frozen=False, with_mf=False)
        return dm1

    def make_fragment_dm2cumulant(
        self, t_as_lambda=False, sym_t2=True, sym_dm2=True, full_shape=True, approx_cumulant=True
    ):
        """Currently MP2/CCSD only"""

        if self.solver == "MP2":
            if approx_cumulant not in (1, True):
                raise NotImplementedError
            t2x = self.results.pwf.restore(sym=sym_t2).as_ccsd().t2
            dovov = 2 * (2 * t2x - t2x.transpose(0, 1, 3, 2)).transpose(0, 2, 1, 3)
            if not full_shape:
                return dovov
            nocc, nvir = dovov.shape[:2]
            norb = nocc + nvir
            dm2 = np.zeros(4 * [norb])
            occ, vir = np.s_[:nocc], np.s_[nocc:]
            dm2[occ, vir, occ, vir] = dovov
            dm2[vir, occ, vir, occ] = dovov.transpose(1, 0, 3, 2)
            return dm2

        cc = d1 = None
        d2 = self._get_projected_gamma2_intermediates(t_as_lambda=t_as_lambda, sym_t2=sym_t2)
        dm2 = pyscf.cc.ccsd_rdm._make_rdm2(cc, d1, d2, with_dm1=False, with_frozen=False)
        if approx_cumulant == 2:
            raise NotImplementedError
        elif approx_cumulant in (1, True):
            pass
        elif not approx_cumulant:
            # Remove dm1(cc)^2
            dm1x = self.make_fragment_dm1(t_as_lambda=t_as_lambda, sym_t2=sym_t2)
            dm1 = self.results.wf.make_rdm1(with_mf=False)
            dm2 -= (
                einsum("ij,kl->ijkl", dm1, dm1x) / 2
                + einsum("ij,kl->ijkl", dm1x, dm1) / 2
                - einsum("ij,kl->iklj", dm1, dm1x) / 4
                - einsum("ij,kl->iklj", dm1x, dm1) / 4
            )

        if sym_dm2 and not sym_t2:
            dm2 = (dm2 + dm2.transpose(1, 0, 3, 2) + dm2.transpose(2, 3, 0, 1) + dm2.transpose(3, 2, 1, 0)) / 4
        return dm2

    # def make_partial_dm1_energy(self, t_as_lambda=False):
    #    dm1 = self.make_partial_dm1(t_as_lambda=t_as_lambda)
    #    c_act = self.cluster.c_active
    #    fock = np.linalg.multi_dot((c_act.T, self.base.get_fock(), c_act))
    #    e_dm1 = einsum('ij,ji->', fock, dm1)
    #    return e_dm1

    @log_method()
    def make_fragment_dm2cumulant_energy(self, hamil=None, t_as_lambda=False, sym_t2=True, approx_cumulant=True):
        if hamil is None:
            hamil = self.hamil

        # This is a refactor of original functionality with three forks.
        #   - MP2 solver so dm2 cumulant is just ovov, and we just want to contract this.
        #   - CCSD solver so want to use approximate cumulant and can use optimal contraction of different ERI blocks
        #     making use of permutational symmetries.
        #   - All other solvers where we just use a dense eri contraction and may or may not use the approximate
        #     cumulant.
        # With the new hamiltonian object we can always use the optimal contraction for the approximate cumulant,
        # regardless of solver, and we support `approx_cumulant=False` for CCSD.

        if self.solver == "MP2":
            # This is just ovov shape in this case. TODO neater way to handle this?
            dm2 = self.make_fragment_dm2cumulant(
                t_as_lambda=t_as_lambda, sym_t2=sym_t2, approx_cumulant=approx_cumulant, full_shape=False
            )
            return 2 * einsum("ijkl,ijkl->", hamil.get_eris_bare("ovov"), dm2) / 2
        elif approx_cumulant:
            # Working hypothesis: this branch will effectively always uses `approx_cumulant=True`.
            eris = hamil.get_dummy_eri_object(force_bare=True, with_vext=False)
            d2 = self._get_projected_gamma2_intermediates(t_as_lambda=t_as_lambda, sym_t2=sym_t2)
            return vayesta.core.ao2mo.helper.contract_dm2intermeds_eris_rhf(d2, eris) / 2
        else:
            dm2 = self.make_fragment_dm2cumulant(
                t_as_lambda=t_as_lambda, sym_t2=sym_t2, approx_cumulant=approx_cumulant, full_shape=True
            )
            e_dm2 = einsum("ijkl,ijkl->", hamil.get_eris_bare(), dm2) / 2
        return e_dm2

    # --- Other
    # ---------

    def get_fragment_bsse(self, rmax=None, nimages=5, unit="A"):
        self.log.info("Counterpoise Calculation")
        self.log.info("************************")
        if rmax is None:
            rmax = self.opts.bsse_rmax

        # Atomic calculation with atomic basis functions:
        # mol = self.mol.copy()
        # atom = mol.atom[self.atoms]
        # self.log.debugv("Keeping atoms %r", atom)
        # mol.atom = atom
        # mol.a = None
        # mol.build(False, False)

        natom0, e_mf0, e_cm0, dm = self.counterpoise_calculation(rmax=0.0, nimages=0)
        assert natom0 == len(self.atoms)
        self.log.debugv("Counterpoise: E(atom)= % 16.8f Ha", e_cm0)

        # natom_list = []
        # e_mf_list = []
        # e_cm_list = []
        r_values = np.hstack((np.arange(1.0, int(rmax) + 1, 1.0), rmax))
        # for r in r_values:
        r = rmax
        natom, e_mf, e_cm, dm = self.counterpoise_calculation(rmax=r, dm0=dm)
        self.log.debugv(
            "Counterpoise: n(atom)= %3d  E(mf)= %16.8f Ha  E(%s)= % 16.8f Ha", natom, e_mf, self.solver, e_cm
        )

        e_bsse = self.sym_factor * (e_cm - e_cm0)
        self.log.debugv("Counterpoise: E(BSSE)= % 16.8f Ha", e_bsse)
        return e_bsse

    def counterpoise_calculation(self, rmax, dm0=None, nimages=5, unit="A"):
        mol = self.make_counterpoise_mol(rmax, nimages=nimages, unit=unit, output="pyscf-cp.txt")
        # Mean-field
        # mf = type(self.mf)(mol)
        mf = pyscf.scf.RHF(mol)
        mf.conv_tol = self.mf.conv_tol
        # if self.mf.with_df is not None:
        #    self.log.debugv("Setting GDF")
        #    self.log.debugv("%s", type(self.mf.with_df))
        #    # ONLY GDF SO FAR!
        # TODO: generalize
        if self.base.kdf is not None:
            auxbasis = self.base.kdf.auxbasis
        elif self.mf.with_df is not None:
            auxbasis = self.mf.with_df.auxbasis
        else:
            auxbasis = None
        if auxbasis:
            mf = mf.density_fit(auxbasis=auxbasis)
        # TODO:
        # use dm0 as starting point
        mf.kernel()
        dm0 = mf.make_rdm1()
        # Embedded calculation with same options
        ecc = ewf.EWF(mf, solver=self.solver, bno_threshold=self.bno_threshold, options=self.base.opts)
        ecc.make_atom_cluster(self.atoms, options=self.opts)
        ecc.kernel()

        return mol.natm, mf.e_tot, ecc.e_tot, dm0
