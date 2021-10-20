import dataclasses
import copy

import numpy as np
import scipy.linalg

from pyscf.agf2 import mpi_helper, aux
from pyscf.pbc.scf.rsjk import RangeSeparationJKBuilder

import vayesta
from vayesta.ewf.helper import orthogonalize_mo
from vayesta.core import QEmbeddingMethod
from vayesta.core.util import time_string
from vayesta.eagf2 import helper
from vayesta.eagf2.fragment import EAGF2Fragment
from vayesta.eagf2.ragf2 import RAGF2, RAGF2Options, DIIS

try:
    from mpi4py import MPI
    timer = MPI.Wtime
except ImportError:
    from timeit import default_timer as timer


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
    with_rsjk: bool = False

    # --- Different defaults for some RAGF2 settings
    conv_tol: float = 1e-6
    conv_tol_rdm1: float = 1e-10
    conv_tol_nelec: float = 1e-8
    conv_tol_nelec_factor: float = 1e-2
    max_cycle_inner: int = 50
    max_cycle_outer: int = 25


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
    DIIS = DIIS

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
        return -self.results.gf.get_occupied().energy.max()

    @property
    def e_ea(self):
        return self.results.gf.get_virtual().energy.min()

    @property
    def converged(self):
        return self.results.converged


    #TODO break up into functions
    def kernel(self):
        ''' Run the EAGF2 calculation.

        Returns
        -------
        results : EAGF2Results
            Object containing results of `EAGF2` calculation, see
            `EAGF2Results` for a list of attributes.
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
            solver = RAGF2(
                    self.mf,
                    eri=np.empty(()),
                    veff=np.empty(()),
                    log=self.log,
                    options=self.opts,
                    fock_basis='ao' if not self.opts.with_rsjk else 'rsjk',
            )
            solver.log = self.log

        if self.opts.with_rsjk:
            rsjk = RangeSeparationJKBuilder(self.kcell, self.kpts)
            rsjk.verbose = self.kcell.verbose
            rsjk.build(direct_scf_tol=self.opts.conv_tol_rdm1)
            solver.rsjk = rsjk

        diis = self.DIIS(space=self.opts.diis_space, min_space=self.opts.diis_min_space)
        solver.se = aux.SelfEnergy(np.empty((0)), np.empty((self.nmo, 0)))
        fock = np.diag(self.mf.mo_energy)
        e_nuc = solver.e_nuc

        converged = False
        for niter in range(0, self.opts.max_cycle+1):
            t1 = timer()
            self.log.info("Iteration %d" % niter)
            self.log.info("**********%s" % ('*'*len(str(niter))))
            with self.log.withIndentLevel(1):

                se_prev = copy.deepcopy(solver.se)
                e_prev = solver.e_tot

                for x, frag in enumerate(self.fragments):
                    #TODO is this needed? are the projectors the same as the sym_parent ones?
                    #if frag.sym_parent is not None:
                    #    continue
                    self.log.info("Building fragment space for fragment %d", x)

                    with self.log.withIndentLevel(1):

                        n = frag.c_frag.shape[0]
                        nqmo = self.nmo + solver.se.naux
                        c_frag = np.zeros((nqmo, frag.c_frag.shape[-1]))
                        c_frag[:self.nmo] = frag.c_frag[:self.nmo]
                        c_env = helper.null_space(c_frag, nvecs=nqmo-c_frag.shape[-1])

                        frag.c_frag, frag.c_env = c_frag, c_env

                        frag.se = solver.se
                        frag.fock = fock
                        frag.qmo_energy, frag.qmo_coeff = solver.se.eig(fock)
                        frag.qmo_occ = np.array([2.0 * (x < solver.se.chempot) for x in frag.qmo_energy])

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
                            results = frag.kernel(solver, other_frag=other)
                            self.cluster_results[frag.id, other.id] = results
                            self.log.debugv("%s - %s is done.", frag, other)
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

                        for i in range(results.moms.shape[0]):
                            for j in range(results.moms.shape[1]):
                                m = np.linalg.multi_dot((p_frag.T.conj(), results.moms[i, j], p_other))
                                moms[i, j] += m
                                if x != y:
                                    moms[i, j] += m.T.conj()

                    self.log.info("%s is done.", frag)

                assert np.allclose(moms[0,0], moms[0,0].T.conj())
                assert np.allclose(moms[0,1], moms[0,1].T.conj())
                assert np.allclose(moms[1,0], moms[1,0].T.conj())
                assert np.allclose(moms[1,1], moms[1,1].T.conj())


                # === Occupied:

                self.log.info("Occupied self-energy:")
                with self.log.withIndentLevel(1):
                    se_occ = solver._build_se_from_moments(moms[0], eps=self.opts.weight_tol)
                    w = np.linalg.eigvalsh(moms[0][0])
                    wmin, wmax = w.min(), w.max()
                    (self.log.warning if wmin < 1e-8 else self.log.debug)(
                            'Eigenvalue range:  %.5g -> %.5g', wmin, wmax,
                    )
                    self.log.info("Built %d occupied auxiliaries", se_occ.naux)


                # === Virtual:
                
                self.log.info("Virtual self-energy:")
                with self.log.withIndentLevel(1):
                    se_vir = solver._build_se_from_moments(moms[1], eps=self.opts.weight_tol)
                    w = np.linalg.eigvalsh(moms[1][0])
                    wmin, wmax = w.min(), w.max()
                    (self.log.warning if wmin < 1e-8 else self.log.debug)(
                            'Eigenvalue range:  %.5g -> %.5g', wmin, wmax,
                    )
                    self.log.info("Built %d virtual auxiliaries", se_vir.naux)

                nh = solver.nocc-solver.frozen[0]
                wt = lambda v: np.sum(v * v)
                self.log.infov("Total weights of coupling blocks:")
                self.log.infov("        %6s  %6s", "2h1p", "1h2p")
                self.log.infov("    1h  %6.4f  %6.4f", wt(se_occ.coupling[:nh]), wt(se_occ.coupling[nh:]))
                self.log.infov("    1p  %6.4f  %6.4f", wt(se_vir.coupling[:nh]), wt(se_vir.coupling[nh:]))


                solver.se = solver._combine_se(se_occ, se_vir)

                if niter != 0:
                    solver.run_diis(solver.se, None, diis, se_prev=se_prev)
                    solver.se.remove_uncoupled(tol=self.opts.weight_tol)

                w, v = solver.solve_dyson(fock=fock)
                solver.gf = aux.GreensFunction(w, v[:self.nmo])
                solver.gf.remove_uncoupled(tol=self.opts.weight_tol)

                if self.opts.fock_loop:
                    solver.gf, solver.se, fconv, fock = solver.fock_loop(fock=fock, return_fock=True)
                solver.gf.remove_uncoupled(tol=self.opts.weight_tol)

                solver.e_1b = solver.energy_1body(e_nuc=e_nuc)
                solver.e_2b = solver.energy_2body()
                solver.print_energies()
                solver.print_excitations()

                deltas = solver._convergence_checks(se=solver.se, se_prev=se_prev, e_prev=e_prev)

                self.log.info("Change in energy:     %10.3g", deltas[0])
                self.log.info("Change in 0th moment: %10.3g", deltas[1])
                self.log.info("Change in 1st moment: %10.3g", deltas[2])

                if self.opts.dump_chkfile and solver.chkfile is not None:
                    self.log.debug("Dumping current iteration to chkfile")
                    solver.dump_chk()

                self.log.timing("Time for AGF2 iteration:  %s", time_string(timer() - t1))

            if deltas[0] < self.opts.conv_tol \
                    and deltas[1] < self.opts.conv_tol_t0 \
                    and deltas[2] < self.opts.conv_tol_t1:
                converged = True
                break

        solver.e_1b = solver.energy_1body()
        solver.e_2b = solver.energy_2body()

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
