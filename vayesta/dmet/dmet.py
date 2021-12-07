import os.path
import functools
from datetime import datetime
import dataclasses

import numpy as np
import scipy
import scipy.linalg

from vayesta.core.util import *
from vayesta.core import QEmbeddingMethod

from vayesta.ewf import helper
from .fragment import VALID_SOLVERS, DMETFragment, DMETFragmentExit
from .sdp_sc import perform_SDP_fit
from .updates import MixUpdate, DIISUpdate

from timeit import default_timer as timer
import copy


@dataclasses.dataclass
class DMETResults:
    cluster_sizes: np.ndarray = None
    e_corr: float = None


class DMET(QEmbeddingMethod):
    @dataclasses.dataclass
    class Options(QEmbeddingMethod.Options):
        """Options for EWF calculations."""
        # --- Fragment settings
        # fragment_type: str = 'IAO'
        localize_fragment: bool = False  # Perform numerical localization on fragment orbitals
        iao_minao: str = 'auto'  # Minimal basis for IAOs
        # --- Bath settings
        bath_type: str = None
        dmet_threshold: float = 1e-6
        orthogonal_mo_tol: float = False
        # Orbital file
        plot_orbitals: str = False  # {True, False, 'dmet-exit'}
        plot_orbitals_dir: str = 'orbitals'
        plot_orbitals_kwargs: dict = dataclasses.field(default_factory=dict)
        # --- Solver settings
        solver_options: dict = dataclasses.field(default_factory=dict)
        make_rdm1: bool = True
        make_rdm2: bool = True
        dm_with_frozen: bool = False  # Add frozen parts to cluster DMs
        # Counterpoise correction of BSSE
        bsse_correction: bool = True
        bsse_rmax: float = 5.0  # In Angstrom
        # -- Self-consistency
        maxiter: int = 30
        sc_mode: int = 0
        sc_energy_tol: float = 1e-6
        charge_consistent: bool = True
        max_elec_err: float = 1e-4
        conv_tol: float = 1e-6
        diis: bool = True
        mixing_param: float = 0.5
        mixing_variable: str = "hl rdm"
        # --- Other
        energy_partitioning: str = 'first-occ'
        strict: bool = False  # Stop if cluster not converged

    Fragment = DMETFragment

    def __init__(self, mf, bno_threshold=np.inf, solver='CCSD', options=None, log=None, **kwargs):
        """Density matrix embedding theory (DMET) calculation object.

        Parameters
        ----------

        """

        t_start = timer()
        super().__init__(mf, options=options, log=log, **kwargs)

        self.log.info("DMET parameters:")
        for key, val in self.opts.items():
            self.log.info('  > %-24s %r', key + ':', val)

        # --- Check input
        if not mf.converged:
            if self.opts.strict:
                raise RuntimeError("Mean-field calculation not converged.")
            else:
                self.log.error("Mean-field calculation not converged.")
        self.bno_threshold = bno_threshold
        self.check_solver(solver)
        self.solver = solver

        self.vcorr = None

        self.iteration = 0
        self.cluster_results = {}
        self.results = []
        self.e_dmet = self.e_mf - self.mf.energy_nuc()

        self.log.timing("Time for DMET setup: %s", time_string(timer() - t_start))

    def check_solver(self, solver):
        if solver not in VALID_SOLVERS:
            raise ValueError("Unknown solver: %s" % solver)

    def __repr__(self):
        keys = ['mf', 'bno_threshold', 'solver']
        fmt = ('%s(' + len(keys) * '%s: %r, ')[:-2] + ')'
        values = [self.__dict__[k] for k in keys]
        return fmt % (self.__class__.__name__, *[x for y in zip(keys, values) for x in y])

    @property
    def e_corr(self):
        """Total energy."""
        return self.e_tot - self.e_mf

    @property
    def e_tot(self):
        """Total energy."""
        return self.e_dmet + self.mf.energy_nuc()

    def kernel(self, bno_threshold=None):
        """Run DMET calculation.
        """
        t_start = timer()

        if self.nfrag == 0:
            raise ValueError("No fragments defined for calculation.")

        maxiter = self.opts.maxiter
        # View this as a single number for now.
        bno_thr = bno_threshold or self.bno_threshold
        if bno_thr < np.inf:
            raise NotImplementedError("MP2 bath calculation is currently ignoring the correlation potential, so does"
                                      " not work properly for self-consistent calculations.")
        # rdm = self.mf.make_rdm1()
        fock = self.get_fock()
        self.vcorr = np.zeros((self.nao,) * 2)

        cpt = 0.0
        mf = self.mf

        sym_parents = self.get_symmetry_parent_fragments()
        sym_children = self.get_symmetry_child_fragments()
        nsym = [len(x) + 1 for x in sym_children]

        if not self.opts.mixing_variable == "hl rdm":
            raise ValueError("Only DIIS extrapolation of the high-level rdms is current implemented.")

        if self.opts.diis:
            self.updater = DIISUpdate()
        else:
            self.updater = MixUpdate(self.opts.mixing_param)

        impurity_projectors = [
            [parent.c_frag] + [c.c_frag for c in children] for (parent, children) in zip(sym_parents, sym_children)
        ]

        self.converged = False
        for iteration in range(1, maxiter + 1):
            self.iteration = iteration
            self.log.info("Now running iteration= %2d", iteration)
            self.log.info("****************************************************")
            mf.mo_energy, mf.mo_coeff = mf.eig(fock + self.vcorr, self.get_ovlp())
            mf.mo_occ = self.mf.get_occ(mf.mo_energy, mf.mo_coeff)

            if self.opts.charge_consistent: fock = mf.get_fock()
            # Need to optimise a global chemical potential to ensure electron number is converged.

            nelec_mf = 0.0
            rdm = self.mf.make_rdm1()
            # This could loop over parents and multiply. Leave simple for now.
            for x, frag in enumerate(self.fragments):
                c = frag.c_frag.T @ self.get_ovlp()  # / np.sqrt(2)
                nelec_mf += np.linalg.multi_dot((c, rdm, c.T)).trace()
                # Print local 1rdm
                # print(np.linalg.multi_dot((c, rdm, c.T))/2)
                # print(np.linalg.multi_dot((c, rdm, c.T)).trace())

            def electron_err(cpt):
                err = self.calc_electron_number_defect(cpt, bno_thr, nelec_mf, sym_parents, nsym)
                return err

            err = electron_err(cpt)

            if abs(err) > self.opts.max_elec_err * nelec_mf:
                # Need to find chemical potential bracket.
                # Error is positive if excess electrons at high-level, and negative if too few electrons at high-level.
                # Changing chemical potential should introduces similar change in high-level electron number, so we want
                # our new chemical potential to be shifted in the opposite direction as electron error.
                new_cpt = cpt - np.sign(err) * 0.1
                # Set this in case of errors later on.
                new_err = err
                try:
                    new_err = electron_err(new_cpt)
                except np.linalg.LinAlgError as e:
                    if self.solver == "CCSD":
                        self.log.info("Caught DIIS error in CCSD; trying smaller chemical potential deviation.")
                        # Want to end up with 3/4 of current value after multiplied by two.
                        new_cpt = cpt - (new_cpt - cpt) * 3 / 8
                    else:
                        raise e
                if err * new_err > 0:  # Check if errors have same sign.
                    for ntry in range(10):
                        new_cpt = cpt + (new_cpt - cpt) * 2
                        try:
                            new_err = electron_err(new_cpt)
                        except np.linalg.LinAlgError as e:
                            if self.solver == "CCSD":
                                self.log.info("Caught DIIS error in CCSD; trying smaller chemical potential deviation.")
                                # Want to end up with 3/4 of current value after multiplied by two.
                                new_cpt = cpt - (new_cpt - cpt) * 3 / 8
                            else:
                                raise e
                        if err * new_err < 0:
                            break
                    else:
                        self.log.fatal("Could not find chemical potential bracket.")
                        break
                # If we've got to here we've found a bracket.
                [lo, hi] = sorted([cpt, new_cpt])
                cpt, res = scipy.optimize.brentq(electron_err, a=lo, b=hi, full_output=True,
                                                 xtol=self.opts.max_elec_err * nelec_mf)  # self.opts.max_elec_err * nelec_mf)
                self.log.info("Converged chemical potential: {:6.4e}".format(cpt))

            else:
                self.log.info("Previous chemical potential still suitable")

            e1, e2 = 0.0, 0.0
            for x, frag in enumerate(sym_parents):
                e1_contrib, e2_contrib = frag.get_dmet_energy_contrib()
                e1 += e1_contrib * nsym[x]
                e2 += e2_contrib * nsym[x]
                # print(e1 + e2, e1, e2)
                # print(frag.get_fragment_dmet_energy())
            self.e_dmet = e1 + e2
            self.log.info("Total DMET energy {:8.4f}".format(self.e_tot))
            self.log.info("Energy Contributions: 1-body={:8.4f}, 2-body={:8.4f}".format(e1, e2))

            curr_rdms, delta_rdms = self.updater.update(self.hl_rdms)
            self.log.info("Change in high-level RDMs: {:6.4e}".format(delta_rdms))
            # Now for the DMET self-consistency!
            self.log.info("Now running DMET correlation potential fitting")
            vcorr_new = perform_SDP_fit(self.mol.nelec[0], fock, impurity_projectors, [x / 2 for x in curr_rdms],
                                        self.get_ovlp(), self.log)
            delta = sum((vcorr_new - self.vcorr).reshape(-1) ** 2) ** (0.5)
            self.log.info("Delta Vcorr {:6.4e}".format(delta))
            if delta < self.opts.conv_tol:
                self.converged = True
                self.log.info("DMET converged after %d iterations" % iteration)
                break
            self.vcorr = vcorr_new
        else:
            self.log.error("Self-consistency not reached in {} iterations.".format(maxiter))

        self.print_results()

        self.log.info("Total wall time:  %s", time_string(timer() - t_start))
        self.log.info("All done.")

    def calc_electron_number_defect(self, chempot, bno_thr, nelec_target, parent_fragments, nsym, construct_bath=True):
        self.log.info("Running chemical potential={:8.6e}".format(chempot))

        hl_rdms = [None] * len(parent_fragments)
        nelec_hl = 0.0
        exit = False
        for x, frag in enumerate(parent_fragments):
            msg = "Now running %s" % (frag)
            self.log.info(msg)
            self.log.info(len(msg) * "*")
            self.log.changeIndentLevel(1)

            try:
                result = frag.kernel(bno_threshold=bno_thr, construct_bath=construct_bath, chempot=chempot)
            except DMETFragmentExit as e:
                exit = True
                self.log.info("Exiting %s", frag)
                self.log.changeIndentLevel(-1)
                raise e

            self.cluster_results[frag.id] = result
            if not result.converged:
                self.log.error("%s is not converged!", frag)
            else:
                self.log.info("%s is done.", frag)
            self.log.changeIndentLevel(-1)
            if exit:
                break
            # Project rdm into fragment space; currently in cluster canonical orbitals.
            c = dot(frag.c_frag.T, self.mf.get_ovlp(), frag.cluster.c_active)
            hl_rdms[x] = dot(c, frag.results.dm1, c.T)  # / 2
            nelec_hl += hl_rdms[x].trace() * nsym[x]
        self.hl_rdms = hl_rdms
        self.log.info("Chemical Potential {:8.6e} gives Total electron deviation {:6.4e}".format(
            chempot, nelec_hl - nelec_target))
        return nelec_hl - nelec_target

    def print_results(self):  # , results):
        self.log.info("Energies")
        self.log.info("********")
        fmt = "%-20s %+16.8f Ha"
        # for i, frag in enumerate(self.loop()):
        #    e_corr = results["e_corr"][i]
        #    self.log.output(fmt, 'E(corr)[' + frag.trimmed_name() + ']=', e_corr)
        self.log.output(fmt, 'E(corr)=', self.e_corr)
        self.log.output(fmt, 'E(MF)=', self.e_mf)
        self.log.output(fmt, 'E(nuc)=', self.mol.energy_nuc())
        self.log.output(fmt, 'E(tot)=', self.e_tot)

    def print_clusters(self):
        """Print fragments of calculations."""
        self.log.info("%3s  %20s  %8s  %4s", "ID", "Name", "Solver", "Size")
        for frag in self.loop():
            self.log.info("%3d  %20s  %8s  %4d", frag.id, frag.name, frag.solver, frag.size)

    def update_params(self, params, ):
        """Given list of different parameter vectors, perform DIIS on them all at once and return the corresponding
        separate parameter strings.
        :param params: list of vectors for different parameter values.
        :return:
        """
        conv_grad = self.check_convergence_grad(params)
        inp = np.concatenate(params)
        new_params = self.adiis.update(inp)
        self.prev_params = new_params
        x = 0
        res = []
        for i in params:
            res += [new_params[x:x + len(i)]]
            x += len(i)
        self.iter += 1
        return res, conv_grad
