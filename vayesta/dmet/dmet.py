import dataclasses
from timeit import default_timer as timer

import numpy as np
import scipy
import scipy.linalg

from vayesta.core.qemb import Embedding
from vayesta.core.util import *
from .fragment import DMETFragment, DMETFragmentExit

from .sdp_sc import perform_SDP_fit
from .updates import MixUpdate, DIISUpdate


@dataclasses.dataclass
class Options(Embedding.Options):
    """Options for DMET calculations."""
    iao_minao: str = 'auto'  # Minimal basis for IAOs
    dm_with_frozen: bool = False  # Add frozen parts to cluster DMs
    # -- Self-consistency
    maxiter: int = 30
    charge_consistent: bool = True
    max_elec_err: float = 1e-4
    conv_tol: float = 1e-6
    diis: bool = True
    mixing_param: float = 0.5
    mixing_variable: str = "hl rdm"
    oneshot: bool = False
    # --- Solver options
    solver_options: dict = Embedding.Options.change_dict_defaults('solver_options',
            # CCSD
            solve_lambda=True)

@dataclasses.dataclass
class DMETResults:
    cluster_sizes: np.ndarray = None
    e_corr: float = None

class DMET(Embedding):

    Fragment = DMETFragment
    Options = Options
    valid_solvers = ['MP2', 'CISD', 'CCSD', 'FCI', 'FCI-SPIN0', 'FCI-SPIN1']

    def __init__(self, mf, solver='CCSD', log=None, **kwargs):
        t_start = timer()
        # If we're running in oneshot mode will only do a single iteration, regardless of this setting, but good to have
        # consistent settings.
        if kwargs.get("oneshot", False):
            kwargs["maxiter"] = 1

        super().__init__(mf, solver=solver, log=log, **kwargs)

        self.log.info("Parameters of %s:", self.__class__.__name__)
        self.log.info(break_into_lines(str(self.opts), newline='\n    '))

        # --- Check input
        if not mf.converged:
            self.log.error("Mean-field calculation not converged.")

        self.vcorr = None

        self.iteration = 0
        self.cluster_results = {}
        self.results = []
        self.e_dmet = self.e_mf - self.mf.energy_nuc()

        self.log.timing("Time for DMET setup: %s", time_string(timer() - t_start))

    @property
    def e_tot(self):
        return self.e_mf + self.e_corr

    def __repr__(self):
        keys = ['mf', 'solver']
        fmt = ('%s(' + len(keys) * '%s: %r, ')[:-2] + ')'
        values = [self.__dict__[k] for k in keys]
        return fmt % (self.__class__.__name__, *[x for y in zip(keys, values) for x in y])

    def kernel(self):
        """Run DMET calculation.
        """
        t_start = timer()

        if self.nfrag == 0:
            raise ValueError("No fragments defined for calculation.")

        maxiter = self.opts.maxiter
        # View this as a single number for now.
        if self.opts.bath_options['bathtype'] == 'mp2' and maxiter > 1:
            raise NotImplementedError("MP2 bath calculation is currently ignoring the correlation potential, so does"
                                      " not work properly for self-consistent calculations.")

        fock = self.get_fock()
        if self.vcorr is None:
            self.vcorr = np.zeros((self.nao,) * 2)
        else:
            self.log.info("Starting from previous correlation potential.")

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
        self.converged = False
        for iteration in range(1, maxiter + 1):
            self.iteration = iteration
            self.log.info("Now running iteration %2d", iteration)
            self.log.info("------------------------")
            if iteration > 1:
                # For first iteration want to run on provided mean-field state.
                mo_energy, mo_coeff = mf.eig(fock + self.vcorr, self.get_ovlp())
                self.update_mf(mo_coeff, mo_energy)

                if self.opts.charge_consistent:
                    fock = self.get_fock()
            # Need to optimise a global chemical potential to ensure electron number is converged.
            nelec_mf = self._check_fragment_nelectron()
            if type(nelec_mf) == tuple:
                nelec_mf = sum(nelec_mf)

            def electron_err(cpt):
                err = self.calc_electron_number_defect(cpt, nelec_mf, sym_parents, nsym)
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
                # Recalculate to ensure all fragments have up-to-date info. Brentq strangely seems to do an extra
                # calculation at the end...
                electron_err(cpt)
            else:
                self.log.info("Previous chemical potential still suitable")

            e1, e2, emf = 0.0, 0.0, 0.0
            for x, frag in enumerate(sym_parents):
                e1_contrib, e2_contrib = frag.results.e1, frag.results.e2
                e1 += e1_contrib * nsym[x]
                e2 += e2_contrib * nsym[x]
                emf += frag.get_fragment_mf_energy() * nsym[x]
                # print(e1 + e2, e1, e2)
                # print(frag.get_fragment_dmet_energy())
            self.e_corr = e1 + e2 - emf
            self.log.info("Total DMET energy {:8.4f}".format(self.e_tot))
            self.log.info("Energy Contributions: 1-body={:8.4f}, 2-body={:8.4f}".format(e1, e2))
            if self.opts.oneshot:
                break
            curr_rdms, delta_rdms = self.updater.update(self.hl_rdms)
            self.log.info("Change in high-level RDMs: {:6.4e}".format(delta_rdms))
            vcorr_new = self.update_vcorr(fock, curr_rdms)
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

    def calc_electron_number_defect(self, chempot, nelec_target, parent_fragments, nsym, construct_bath=True):
        self.log.info("Running chemical potential={:8.6e}".format(chempot))

        nelec_hl = 0.0
        exit = False
        for x, frag in enumerate(parent_fragments):
            msg = "Now running %s" % (frag)
            self.log.info(msg)
            self.log.info(len(msg) * "-")
            self.log.changeIndentLevel(1)

            try:
                result = frag.kernel(construct_bath=construct_bath, chempot=chempot)
            except DMETFragmentExit as e:
                exit = True
                self.log.info("Exiting %s", frag)
                self.log.changeIndentLevel(-1)
                raise e
            self.cluster_results[frag.id] = result
            self.log.changeIndentLevel(-1)
            if exit:
                break
            # Project rdm into fragment space; currently in cluster canonical orbitals.
            nelec_hl += frag.get_nelectron_hl() * nsym[x]

        self.hl_rdms = [f.get_frag_hl_dm() for f in parent_fragments]
        self.log.info("Chemical Potential {:8.6e} gives Total electron deviation {:6.4e}".format(
            chempot, nelec_hl - nelec_target))
        return nelec_hl - nelec_target

    def update_vcorr(self, fock, curr_rdms):
        # Now for the DMET self-consistency!
        self.log.info("Now running DMET correlation potential fitting")
        # Note that we want the total number of electrons, not just in fragments, and that this treats different spin
        # channels separately; for RHF the resultant problems are identical and so can just be solved once.
        # As such need to use the spin-dm, rather than spatial.
        vcorr_new = perform_SDP_fit(self.mol.nelec[0], fock, self.get_impurity_coeffs(), [x / 2 for x in curr_rdms],
                                    self.get_ovlp(), self.log)
        return vcorr_new

    def get_impurity_coeffs(self):
        sym_parents = self.get_symmetry_parent_fragments()
        sym_children = self.get_symmetry_child_fragments()

        return [
                [parent.c_frag] + [c.c_frag for c in children] for (parent, children) in zip(sym_parents, sym_children)
        ]

    def print_results(self):  # , results):
        self.log.info("Energies")
        self.log.info("========")
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

    def make_rdm1(self, *args, **kwargs):
        return self.make_rdm1_demo(*args, **kwargs)

    def make_rdm2(self, *args, **kwargs):
        return self.make_rdm2_demo(*args, **kwargs)

    def get_corrfunc(self, kind, dm1=None, dm2=None, **kwargs):
        if dm1 is None:
            dm1 = self.make_rdm1()
        if dm2 is None:
            dm2 = self.make_rdm2()
        return super().get_corrfunc(kind, dm1=dm1, dm2=dm2, **kwargs)

DMET.make_rdm1.__doc__ = DMET.make_rdm1_demo.__doc__
DMET.make_rdm2.__doc__ = DMET.make_rdm2_demo.__doc__

RDMET = DMET
