import dataclasses
import copy

import numpy as np
import scipy.linalg

from pyscf import lib
from pyscf.agf2 import mpi_helper, aux, chempot

import vayesta
from vayesta.ewf.helper import orthogonalize_mo
from vayesta.core import QEmbeddingMethod, foldscf
from vayesta.core.util import time_string
from vayesta.eagf2 import helper
from vayesta.eagf2.fragment import EAGF2Fragment
from vayesta.eagf2.ragf2 import RAGF2, RAGF2Options
from vayesta.eagf2.kragf2 import KRAGF2

try:
    from mpi4py import MPI
    timer = MPI.Wtime
except ImportError:
    from timeit import default_timer as timer


class RAGF2Solver(RAGF2):
    def __init__(self, emb):
        eri = np.empty(())
        veff = np.empty(())
        super().__init__(emb.mf, eri=eri, veff=veff, log=emb.log, options=emb.opts)

    def get_supercell_qmos(self, qmo_energy, qmo_coeff, qmo_occ=None, se=None, imag_tol=1e-8):
        ''' For compatibility.
        '''

        if qmo_occ is None:
            return qmo_energy, qmo_coeff
        else:
            return qmo_energy, qmo_coeff, qmo_occ

    def build_empty_self_energy(self):
        return aux.SelfEnergy(np.empty((0,)), np.empty((self.nmo, 0)))

    def _build_kspace_se_from_moments(self, *args, **kwargs):
        return super()._build_se_from_moments(*args, **kwargs)


class KRAGF2Solver(KRAGF2):
    def __init__(self, emb):
        eri = np.empty(())
        veff = np.empty(())
        self.phase = emb.mf.kphase
        super().__init__(emb.mf.kmf, eri=eri, veff=veff, log=emb.log, options=emb.opts)

    def get_supercell_qmos(self, qmo_energy, qmo_coeff, qmo_occ=None, se=None, imag_tol=1e-8):
        ''' Convert QMOs from k-space to supercell.
        '''

        return_occ = qmo_occ is not None
        if qmo_occ is None:
            qmo_occ = np.zeros_like(qmo_energy)

        ovlp = np.eye(sum([c.shape[0] for c in qmo_coeff]))
        qmo_energy_sc, qmo_coeff_sc, qmo_occ_sc = \
                foldscf.fold_mos(qmo_energy, qmo_coeff, qmo_occ, self.phase, ovlp)

        # Reorder physical and auxiliary part:
        #  +--------------+      +--------------+
        #  | phys (kpt 0) |      | phys (kpt 0) |
        #  | aux (kpt 0)  |      | phys (kpt 1) |
        #  | phys (kpt 1) | ---\ |      ...     |
        #  | aux (kpt 1)  | ---/ | aux (kpt 0)  |
        #  |      ...     |      | aux (kpt 1)  |
        #  |      ...     |      |      ...     |
        #  +--------------+      +--------------+
        phys = []
        aux = []
        tot = 0
        for c in qmo_coeff:
            phys.append(list(range(tot, tot+self.nmo)))
            aux.append(list(range(tot+self.nmo, tot+c.shape[0])))
            tot += c.shape[0]
        mask = np.array(sum(phys + aux, []))
        assert np.allclose(np.sort(mask), list(range(tot)))
        qmo_coeff_sc = qmo_coeff_sc[mask]

        # Reorder physical part:
        #  +-------------+      +-------------+
        #  | occ (kpt 0) |      | occ (kpt 0) |
        #  | vir (kpt 0) |      | occ (kpt 1) |
        #  | occ (kpt 1) | ---\ |      ...    |
        #  | vir (kpt 1) | ---/ | vir (kpt 0) |
        #  |      ...    |      | vir (kpt 1) |
        #  |      ...    |      |      ...    |
        #  +-------------+      +-------------+
        mask = np.argsort(np.argsort(np.concatenate(self.mf.mo_energy)))
        nmo = self.nmo * self.nkpts
        qmo_coeff_sc[:nmo] = qmo_coeff_sc[:nmo][mask]

        # Reorder auxiliary part:
        #  +-------------+      +-------------+
        #  | occ (kpt 0) |      | occ (kpt 0) |
        #  | vir (kpt 0) |      | occ (kpt 1) |
        #  | occ (kpt 1) | ---\ |      ...    |
        #  | vir (kpt 1) | ---/ | vir (kpt 0) |
        #  |      ...    |      | vir (kpt 1) |
        #  |      ...    |      |      ...    |
        #  +-------------+      +-------------+
        if qmo_coeff_sc.shape[0] != nmo:
            if se is None:
                c_aux = qmo_coeff_sc[nmo:]
                aux_energy = np.einsum('xi,xi,i->x', c_aux, c_aux.conj(), qmo_energy_sc)
            else:
                aux_energy = np.concatenate([s.energy for s in se])
            mask = np.argsort(np.argsort(aux_energy))
            qmo_coeff_sc[nmo:] = qmo_coeff_sc[nmo:][mask]

        if return_occ:
            return qmo_energy_sc, qmo_coeff_sc, qmo_occ_sc
        else:
            return qmo_energy_sc, qmo_coeff_sc

    def _build_kspace_se_from_moments(self, t, chempot=0.0, eps=1e-16, debug=True):
        '''
        Build the occupied or virtual self-energy in k-space from the
        supercell moments.
        '''

        phase = self.phase
        nkpts, nr = phase.shape
        t = t.reshape(2*self.opts.nmom_lanczos+2, nr, self.nmo, nr, self.nmo)
        t = np.einsum('nRiSj,kR,kS->knij', t, phase.conj(), phase)

        se = [RAGF2._build_se_from_moments(self, tk, chempot=chempot, eps=eps) for tk in t]

        return se

    def _combine_se(self, se_occ, se_vir, gf=None):
        if isinstance(se_occ, (list, tuple)):
            if gf is None:
                return [RAGF2._combine_se(self, o, v) for o, v in zip(se_occ, se_vir)]
            else:
                return [RAGF2._combine_se(self, o, v, gf=g) for o, v, g in zip(se_occ, se_vir, gf)]
        else:
            return RAGF2._combine_se(self, se_occ, se_vir, gf=gf)

    def build_empty_self_energy(self):
        return [aux.SelfEnergy(np.empty((0,)), np.empty((self.nmo, 0))) for k in self.kpts]


@dataclasses.dataclass
class EAGF2Options(RAGF2Options):
    ''' Options for EAGF2 calculations - see `RAGF2Options`.
    '''

    # --- Bath settings
    bath_type: str = 'POWER'    # 'MP2-BNO', 'POWER', 'ALL', 'NONE'
    max_bath_order: int = 2
    bno_threshold: float = 1e-8
    bno_threshold_factor: float = 1.0
    dmet_threshold: float = 1e-8

    # --- Other
    strict: bool = False
    orthogonal_mo_tol: float = 1e-10
    recalc_vhf: bool = False
    copy_mf: bool = False
    solver: 'typing.Any' = RAGF2Solver

    # --- Different defaults for some RAGF2 settings
    conv_tol: float = 1e-6
    conv_tol_rdm1: float = 1e-10
    conv_tol_nelec: float = 1e-8
    conv_tol_nelec_factor: float = 1e-2
    max_cycle_inner: int = 50
    max_cycle_outer: int = 25
    fock_basis: str = 'ao'


@dataclasses.dataclass
class EAGF2Results:
    ''' Results for EAGF2 calculations.

    Attributes
    ----------
    converged : bool
        Convergence flag.
    e_corr : float
        Correlation energy.
    e_1b : float
        One-body part of total energy, including nuclear repulsion.
    e_2b : float
        Two-body part of total energy.
    gf: pyscf.agf2.GreensFunction
        Green's function object.
    se: pyscf.agf2.SelfEnergy
        Self-energy object.
    solver: RAGF2
        RAGF2 solver object.
    '''

    converged: bool = None
    e_corr: float = None
    e_1b: float = None
    e_2b: float = None
    gf: aux.GreensFunction = None
    se: aux.SelfEnergy = None
    solver: RAGF2 = None


class EAGF2(QEmbeddingMethod):

    Options = EAGF2Options
    Results = EAGF2Results
    Fragment = EAGF2Fragment

    def __init__(self, mf, options=None, log=None, **kwargs):
        ''' Embedded AGF2 calculation.

        Parameters
        ----------
        mf : pyscf.scf ojbect
            Converged mean-field object.
        options : EAGF2Options
            Options `dataclass`.
        log : logging.Logger
            Logger object. If None, the default Vayesta logger is used
            (default value is None).
        fragment_type : {'Lowdin-AO', 'IAO'}
            Fragmentation method (default value is 'Lowdin-AO').
        iao_minao : str
            Minimal basis for IAOs (default value is 'auto').
        bath_type : {'MP2-BNO', 'POWER', 'ALL', 'NONE'}
            Bath orbital method (default value is 'POWER').
        max_bath_order : int
            Maximum order of power orbitals (default value is 2).
        bno_threshold : float
            Threshold for BNO cutoff when `bath_type` is 'MP2-BNO'
            (default value is 1e-8).
        bno_threshold_factor : float
            Additional factor for `bno_threshold` (default value is 1).
        dmet_threshold : float
            Threshold for idempotency of cluster DM in DMET bath
            construction (default value is 1e-4).
        strict : bool
            Force convergence in the mean-field calculations (default
            value is True).
        orthogonal_mo_tol : float
            Threshold for orthogonality in molecular orbitals (default
            value is 1e-9).

        Plus any keyword argument from RAGF2Options.

        Attributes
        ----------
        results : EAGF2Results
            Results of EAGF2 calculation, see `EAGF2Results` for a list
            of attributes.
        e_tot : float
            Total energy.
        e_corr : float
            Correlation energy.
        e_ip : float
            Ionisation potential.
        e_ea : float
            Electron affinity.
        '''

        super().__init__(mf, log=log)
        t0 = timer()

        # --- Options for EAGF2
        self.opts = options
        if self.opts is None:
            self.opts = EAGF2Options(**kwargs)
        else:
            self.opts = self.opts.replace(kwargs)
        self.log.info("EAGF2 parameters:")
        for key, val in self.opts.items():
            self.log.info("  > %-24s %r", key + ":", val)

        # --- Check input
        if not mf.converged:
            if self.opts.strict:
                raise RuntimeError("Mean-field calculation not converged.")
            else:
                self.log.error("Mean-field calculation not converged.")
        if getattr(mf, 'with_df', None) is None:
            self.log.warning("EAGF2 will not scale well without density fitting.")

        # --- Orthogonalize insufficiently orthogonal MOs
        # (For example as a result of k2gamma conversion with low cell.precision)
        c = self.mo_coeff.copy()
        assert np.all(c.imag == 0), "max|Im(C)|= %.2e" % abs(c.imag).max()
        ctsc = np.linalg.multi_dot((c.T, self.get_ovlp(), c))
        nonorth = abs(ctsc - np.eye(ctsc.shape[-1])).max()
        self.log.info("Max. non-orthogonality of input orbitals= %.2e%s", nonorth,
                      " (!!!)" if nonorth > 1e-5 else "")
        if self.opts.orthogonal_mo_tol and nonorth > self.opts.orthogonal_mo_tol:
            t0 = timer()
            self.log.info("Orthogonalizing orbitals...")
            self.mo_coeff = orthogonalize_mo(c, self.get_ovlp())
            change = abs(np.diag(np.linalg.multi_dot((self.mo_coeff.T, self.get_ovlp(), c)))-1)
            self.log.info("Max. orbital change= %.2e%s", change.max(),
                          " (!!!)" if change.max() > 1e-4 else "")
            self.log.timing("Time for orbital orthogonalization: %s", time_string(timer()-t0))

        self.log.timing("Time for EAGF2 setup: %s", time_string(timer() - t0))

        # --- Initialise QMOs:
        self.qmo_energy = self.mf.mo_energy
        self.qmo_coeff = np.eye(self.qmo_energy.size)
        self.qmo_occ = self.mf.get_occ(self.qmo_energy, self.qmo_coeff)

        self.cluster_results = {}
        self.results = None


    @property
    def e_tot(self):
        return self.results.e_1b + self.results.e_2b

    @property
    def e_corr(self):
        return self.results.e_corr

    @property
    def e_ip(self):
        if isinstance(self.results.gf, (list, tuple)):
            return np.min([-g.get_occupied().energy.max() for g in self.results.gf])
        else:
            return -self.results.gf.get_occupied().energy.max()

    @property
    def e_ea(self):
        if isinstance(self.results.gf, (list, tuple)):
            return np.min([g.get_virtual().energy.min() for g in self.results.gf])
        else:
            return self.results.gf.get_virtual().energy.min()

    @property
    def converged(self):
        return self.results.converged


    def build_self_energy(self, solver, fock):
        '''
        Build the self-energy using the current Green's function.

        Updates the QMO quantities in self.
        '''

        t0 = timer()

        qmo_energy, qmo_coeff = solver.solve_dyson(se=solver.se, fock=fock)
        qmo_energy, qmo_coeff = solver.get_supercell_qmos(qmo_energy, qmo_coeff, se=solver.se)
        cpt = chempot.binsearch_chempot((qmo_energy, qmo_coeff), self.nmo, self.nocc*2)[0]
        qmo_occ = 2.0 * (qmo_energy < cpt)
        nqmo = qmo_energy.size

        self.qmo_energy = qmo_energy
        self.qmo_coeff = qmo_coeff
        self.qmo_occ = qmo_occ

        for x, frag in enumerate(self.fragments):
            #TODO is this needed? are the projectors the same as the sym_parent ones?
            #if frag.sym_parent is not None:
            #    continue
            self.log.info("Building fragment space for fragments %d", x)

            with self.log.withIndentLevel(1):
                n = frag.c_frag.shape[0]
                c_frag = np.zeros((nqmo, frag.c_frag.shape[-1]))
                c_frag[:self.nmo] = frag.c_frag[:self.nmo]
                c_env = helper.null_space(c_frag, nvecs=nqmo-c_frag.shape[-1])
                frag.c_frag, frag.c_env = c_frag, c_env

        for x, frag in enumerate(self.fragments):
            if frag.sym_parent is not None:
                continue
            self.log.info("Building cluster space for fragment %d", x)

            coeffs = frag.make_bath()
            frag.c_cls_occ, frag.c_cls_vir, frag.c_env_occ, frag.c_env_vir = coeffs

            qmos = frag.make_qmo_integrals()
            frag.pija, frag.pabi, frag.c_qmo_occ, frag.c_qmo_vir = qmos

        moms = np.zeros((2, 2, self.nmo, self.nmo))  #TODO higher orders
        for x, frag in enumerate(self.fragments):
            for y, other in enumerate(self.fragments[:x+1]):
                if frag.sym_parent is None and other.sym_parent is None:
                    results = frag.kernel(other_frag=other)
                    self.cluster_results[frag.id, other.id] = results
                    self.log.debugv("%s - %s is done", frag, other)

                else:
                    frag_parent = frag if frag.sym_parent is None else frag.sym_parent
                    other_parent = other if other.sym_parent is None else other.sym_parent
                    results = self.cluster_results[frag_parent.id, other_parent.id]
                    self.log.debugv("%s - %s is symmetry related, parent: %s - %s",
                                    frag, other, frag_parent, other_parent)

                c = np.dot(frag.c_frag.T.conj(), results.c_active)
                p = np.dot(c.T.conj(), c)
                p_frag = np.dot(p, results.c_active[:self.nmo].T.conj())

                c = np.dot(other.c_frag.T.conj(), results.c_active_other)
                p = np.dot(c.T.conj(), c)
                p_other = np.dot(p, results.c_active_other[:self.nmo].T.conj())

                for i in range(2):
                    for j in range(2*self.opts.nmom_lanczos+2):
                        m = np.linalg.multi_dot((p_frag.T.conj(), results.moms[i, j], p_other))
                        moms[i, j] += m
                        if x != y:
                            moms[i, j] += m.T.conj()

            self.log.info("%s is done.", frag)

        self.log.info("Building the self-energy")
        self.log.info("************************")

        t_occ, t_vir = moms

        for i in range(2*self.opts.nmom_lanczos+2):
            if not np.allclose(t_occ[i], t_occ[i].T.conj()):
                error = np.max(np.abs(t_occ[i] - t_occ[i].T.conj()))
                self.log.warning("Occupied n=%d moment not hermitian, error = %.6g", i, error)
            if not np.allclose(t_vir[i], t_vir[i].T.conj()):
                error = np.max(np.abs(t_vir[i] - t_vir[i].T.conj()))
                self.log.warning("Virtual n=%d moment not hermitian, error = %.6g", i, error)
            self.log.debug(
                    "Trace of n=%d moments:  Occupied = %.5g  Virtual = %.5g",
                    i, np.trace(t_occ[i]).real, np.trace(t_vir[i]).real,
            )


        #TODO:
        # === Occupied:

        self.log.info("Occupied self-energy:")
        with self.log.withIndentLevel(1):
            #w = np.linalg.eigvalsh(t_occ[0])
            #wmin, wmax = w.min(), w.max()
            #(self.log.warning if wmin < 1e-8 else self.log.debug)(
            #        'Eigenvalue range:  %.5g -> %.5g', wmin, wmax,
            #)

            se_occ = solver._build_kspace_se_from_moments(t_occ, eps=self.opts.weight_tol, chempot=cpt)

            #self.log.info("Built %d occupied auxiliaries", se_occ.naux)


        # === Virtual:
        
        self.log.info("Virtual self-energy:")
        with self.log.withIndentLevel(1):
            #w = np.linalg.eigvalsh(t_vir[0])
            #wmin, wmax = w.min(), w.max()
            #(self.log.warning if wmin < 1e-8 else self.log.debug)(
            #        'Eigenvalue range:  %.5g -> %.5g', wmin, wmax,
            #)

            se_vir = solver._build_kspace_se_from_moments(t_vir, eps=self.opts.weight_tol, chempot=cpt)

            #self.log.info("Built %d virtual auxiliaries", se_vir.naux)

        #nh = solver.nocc-solver.frozen[0]
        #wt = lambda v: np.sum(v * v)
        #self.log.infov("Total weights of coupling blocks:")
        #self.log.infov("        %6s  %6s", "2h1p", "1h2p")
        #self.log.infov("    1h  %6.4f  %6.4f", wt(se_occ.coupling[:nh]), wt(se_vir.coupling[:nh]))
        #self.log.infov("    1p  %6.4f  %6.4f", wt(se_occ.coupling[nh:]), wt(se_vir.coupling[nh:]))


        se = solver._combine_se(se_occ, se_vir)

        #FIXME
        self.log.debugv("Auxiliary energies:")
        if isinstance(se, (list, tuple)):
            energy = np.sort(np.concatenate([s.energy for s in se]))
        else:
            energy = se.energy
        with self.log.withIndentLevel(1):
            for p0, p1 in lib.prange(0, energy.size, 6):
                self.log.debugv("%12.6f " * (p1-p0), *energy[p0:p1])
        self.log.info("Number of auxiliaries built:  %s", energy.size)
        self.log.timing("Time for self-energy build:  %s", time_string(timer() - t0))

        return se


    def fock_loop(self, solver, fock):
        ''' Run the Fock loop.
        '''

        gf, se, fconv, fock = solver.fock_loop(fock=fock, return_fock=True)

        #FIXME
        if isinstance(gf, list):
            [g.remove_uncoupled(tol=self.opts.weight_tol) for g in gf]
        else:
            gf.remove_uncoupled(tol=self.opts.weight_tol)

        return gf, se, fconv, fock


    def run_diis(self, solver, diis, se_prev=None):
        ''' Run DIIS and damping.
        '''

        #TODO: use an option for delaying start
        se = solver.run_diis(solver.se, None, diis, se_prev=se_prev)

        if isinstance(se, list):
            [s.remove_uncoupled(tol=self.opts.weight_tol) for s in se]
        else:
            se.remove_uncoupled(tol=self.opts.weight_tol)

        return se


    def kernel(self):
        ''' Run EAGF2.
        '''

        t0 = timer()

        if self.nfrag == 0:
            raise ValueError("No fragments defined for calculation.")

        nelec_frags = sum([f.sym_factor*f.nelectron for f in self.loop()])
        self.log.info("Total number of mean-field electrons over all fragments= %.8f", nelec_frags)
        if abs(nelec_frags - np.rint(nelec_frags)) > 1e-4:
            self.log.warning("Number of electrons not integer!")

        self.log.info("Initialising solver:")
        with self.log.withIndentLevel(1):
            solver = self.opts.solver(self)

        #TODO allow GF input
        solver.gf = solver.g0 = solver.build_init_greens_function()
        fock = solver.get_fock()

        #TODO allow SE input
        solver.se = solver.build_empty_self_energy()
        solver.se = self.build_self_energy(solver, fock)

        diis = solver.DIIS(space=self.opts.diis_space, min_space=self.opts.diis_min_space)

        e_mp2 = solver.e_init = solver.energy_mp2()
        e_nuc = solver.e_nuc
        self.log.info("Initial energies")
        self.log.info("****************")
        self.log.info("E(nuc)  = %20.12f", self.e_nuc)
        self.log.info("E(MF)   = %20.12f", self.e_mf)
        self.log.info("E(corr) = %20.12f", e_mp2)
        self.log.info("E(tot)  = %20.12f", self.mf.e_tot + e_mp2)

        converged = False
        se_prev = None
        for niter in range(1, self.opts.max_cycle+1):
            t1 = timer()
            self.log.info("Iteration %d", niter)
            self.log.info("**********%s", "*" * len(str(niter)))
            with self.log.withIndentLevel(1):

                se_prev = copy.deepcopy(solver.se)
                e_prev = solver.e_tot
                
                # one-body terms
                solver.gf, solver.se, fconv, fock = self.fock_loop(solver, fock)
                solver.e_1b = solver.energy_1body(e_nuc=e_nuc)

                # two-body terms
                solver.se = self.build_self_energy(solver, fock)
                solver.se = self.run_diis(solver, diis, se_prev=se_prev)
                solver.e_2b = solver.energy_2body()

                solver.print_excitations()
                solver.print_energies()

                deltas = solver._convergence_checks(se=solver.se, se_prev=se_prev, e_prev=e_prev)

                self.log.info("Change in energy:     %10.3g", deltas[0])
                self.log.info("Change in 0th moment: %10.3g", deltas[1])
                self.log.info("Change in 1st moment: %10.3g", deltas[2])

                if self.opts.dump_chkfile and solver.chkfile is not None:
                    self.log.debug("Dumping current iteration to chkfile")
                    solver.dump_chk()

                self.log.timing("Time for AGF2 iteration:  %s", time_string(timer() - t1))

            checks = all([
                    deltas[0] < self.opts.conv_tol,
                    deltas[1] < self.opts.conv_tol_t0,
                    deltas[2] < self.opts.conv_tol_t1,
            ])

            if self.opts.extra_cycle:
                if converged and checks:
                    converged = True
                    break
                converged = checks
            else:
                if checks:
                    converged = True
                    break

        self.results = EAGF2Results(
                converged=converged,
                e_corr=solver.e_corr,
                e_1b=solver.e_1b,
                e_2b=solver.e_2b,
                gf=solver.gf,
                se=solver.se,
                solver=solver,
        )

        (self.log.info if converged else self.log.warning)("Converged = %r", converged)

        if self.opts.dump_chkfile and solver.chkfile is not None:
            self.log.debug("Dumping output to chkfile")
            solver.dump_chk()

        if self.opts.pop_analysis:
            solver.population_analysis()

        if self.opts.dip_moment:
            solver.dip_moment()

        if self.opts.dump_cubefiles:
            #TODO test
            self.log.debug("Dumping orbitals to .cube files")
            gf_occ, gf_vir = solver.gf.get_occupied(), solver.gf.get_virtual()
            for i in range(self.opts.dump_cubefiles):
                if (gf_occ.naux-1-i) >= 0:
                    self.dump_cube(gf_occ.naux-1-i, cubefile="hoqmo%d.cube" % i)
                if (gf_vir.naux+i) < solver.gf.naux:
                    self.dump_cube(gf_vir.naux+i, cubefile="luqmo%d.cube" % i)

        solver.print_energies(output=True)

        self.log.info("Time elapsed:  %s", time_string(timer() - t0))

        return self.results


    def run(self):
        ''' Run self.kernel and return self

        Returns
        -------
        self: EAGF2
            `EAGF2` object containing calculation results.
        '''

        self.kernel()

        return self


    def print_clusters(self):
        """Print fragments of calculations."""
        self.log.info("%3s  %20s  %8s  %4s", "ID", "Name", "Solver", "Size")
        for frag in self.loop():
            self.log.info("%3d  %20s  %8s  %4d", frag.id, frag.name, frag.solver, frag.size)


    def __repr__(self):
        keys = ['mf']
        fmt = ('%s(' + len(keys)*'%s: %r, ')[:-2] + ')'
        values = [self.__dict__[k] for k in keys]
        return fmt % (self.__class__.__name__, *[x for y in zip(keys, values) for x in y])


if __name__ == '__main__':
    from pyscf import gto, scf

    mol = gto.Mole()
    mol.atom = ';'.join(['H 0 0 %d' % x for x in range(10)])
    mol.basis = 'sto6g'
    mol.verbose = 0
    mol.build()

    mf = scf.RHF(mol)
    mf = mf.density_fit(auxbasis='aug-cc-pvqz-ri')
    mf.conv_tol = 1e-12
    mf.kernel()
    assert mf.converged

    opts = {
        'conv_tol': 1e-8,
        'conv_tol_rdm1': 1e-12,
        'conv_tol_nelec': 1e-10,
        'conv_tol_nelec_factor': 1e-4,
    }

    eagf2 = EAGF2(mf, fragment_type='Lowdin-AO', max_bath_order=20)
    for i in range(mol.natm//2):
        eagf2.make_atom_fragment([i*2, i*2+1])
    eagf2.kernel()
    assert eagf2.converged

    from vayesta import log
    log.info("Full AGF2:")
    log.setLevel(25)
    gf2 = RAGF2(mf, log=log)
    gf2.kernel()
    assert gf2.converged
