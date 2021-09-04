
import dataclasses

import numpy as np
import scipy
import scipy.linalg

from timeit import default_timer as timer
import copy
from vayesta.core.util import *

from .fragment import EDMETFragment, EDMETFragmentExit

from vayesta.dmet import DMET
from vayesta.rpa import dRPA

from vayesta.dmet.updates import MixUpdate, DIISUpdate



@dataclasses.dataclass
class EDMETResults:
    cluster_sizes: np.ndarray = None
    e_corr: float = None


class EDMET(DMET):

    Fragment = EDMETFragment

    VALID_SOLVERS = ["EBFCI"]  # , "EBFCIQMC"]

    def __init__(self, mf, bno_threshold=np.inf, solver='EBFCI', options=None, log=None, **kwargs):
        super().__init__(mf, bno_threshold, solver, options, log, **kwargs)

    def kernel(self):

        t_start = timer()

        if self.nfrag == 0:
            raise ValueError("No fragments defined for calculation.")

        maxiter = self.opts.maxiter
        #rdm = self.mf.make_rdm1()
        fock = self.get_fock()
        cpt = 0.0
        mf = self.ll_mf

        sym_parents = self.get_symmetry_parent_fragments()
        sym_children = self.get_symmetry_child_fragments()
        nsym = [len(x) + 1 for x in sym_children]


        if self.opts.mixing_variable == "hl rdm":
            param_shape =  [(x.c_frag.shape[1],x.c_frag.shape[1]) for x in sym_parents]
        else:
            raise ValueError("Only DIIS extrapolation of the high-level rdms is current implemented.")

        if self.opts.diis:
            self.updater = DIISUpdate(param_shape)
        else:
            self.updater = MixUpdate(param_shape, self.opts.mixing_param)

        impurity_projectors = [
            [parent.c_frag] + [c.c_frag for c in children] for (parent, children) in zip(sym_parents, sym_children)
        ]


        # Just a single-shot application initially. Sadly will still need chemical potential.
        # First, set up and run RPA.
        rpa = dRPA(self.mf, self.log)
        rpa.kernel()

        # Then generate RPA moments, currently just up to mean.
        rpa_moms = rpa.gen_moms(1)

        # Then optimise chemical potential to match local electron number...
        nelec_mf = 0.0
        rdm = self.ll_mf.make_rdm1()
        # This could loop over parents and multiply. Leave simple for now.
        for x, frag in enumerate(self.fragments):
            c = frag.c_frag.T @ self.get_ovlp()  # / np.sqrt(2)
            nelec_mf += np.linalg.multi_dot((c, rdm, c.T)).trace()
            # Print local 1rdm
            # print(np.linalg.multi_dot((c, rdm, c.T))/2)
            # print(np.linalg.multi_dot((c, rdm, c.T)).trace())

        def electron_err(cpt):
            err = self.calc_electron_number_defect(cpt, np.inf, nelec_mf, sym_parents, nsym, rpa_moms)
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
            # If we've got to here we've found a bracket.
            [lo, hi] = sorted([cpt, new_cpt])
            cpt, res = scipy.optimize.brentq(electron_err, a=lo, b=hi, full_output=True,
                                             xtol=self.opts.max_elec_err * nelec_mf)  # self.opts.max_elec_err * nelec_mf)
            self.log.info("Converged chemical potential: {:6.4e}".format(cpt))

        else:
            self.log.info("Previous chemical potential still suitable")

        e1, e2, efb = 0.0, 0.0, 0.0
        for x, frag in enumerate(sym_parents):
            e1_contrib, e2_contrib, efb_contrib = frag.get_edmet_energy_contrib()
            e1 += e1_contrib * nsym[x]
            e2 += e2_contrib * nsym[x]
            efb += efb_contrib * nsym[x]
        self.e_dmet = e1 + e2 + efb
        self.log.info("Total EDMET energy {:8.4f}".format(self.e_tot))
        self.log.info("Energy Contributions: 1-body={:8.4f}, 2-body={:8.4f}, coupled-boson={:8.4f}".format(e1,e2,efb))

        # Now have final results.
        self.print_results()

        self.log.info("Total wall time:  %s", time_string(timer()-t_start))
        self.log.info("All done.")


    def calc_electron_number_defect(self, chempot, bno_thr, nelec_target, parent_fragments, nsym, rpa_moms, construct_bath = True):
        self.log.info("Running chemical potential={:8.6e}".format(chempot))
        # Save original one-body hamiltonian calculation.
        saved_hcore = self.ll_mf.get_hcore

        hl_rdms = [None] * len(parent_fragments)
        nelec_hl = 0.0
        exit = False
        for x, frag in enumerate(parent_fragments):
            msg = "Now running %s" % (frag)
            self.log.info(msg)
            self.log.info(len(msg) * "*")
            self.log.changeIndentLevel(1)

            self.ll_mf.get_hcore = lambda *args: self.mf.get_hcore(*args) - chempot * np.dot(frag.c_frag, frag.c_frag.T)
            self._hcore = self.ll_mf.get_hcore()

            try:
                result = frag.kernel(rpa_moms, bno_threshold=bno_thr, construct_bath=construct_bath)
            except EDMETFragmentExit as e:
                exit = True
                self.log.info("Exiting %s", frag)
                self.log.changeIndentLevel(-1)
                self.ll_mf.get_hcore = saved_hcore
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
            c = np.linalg.multi_dot((
                frag.c_frag.T, self.mf.get_ovlp(), np.hstack((frag.c_active_occ, frag.c_active_vir))))
            hl_rdms[x] = np.linalg.multi_dot((c, frag.results.dm1, c.T))# / 2
            nelec_hl += hl_rdms[x].trace() * nsym[x]
        # Set hcore back to original calculation.
        self.ll_mf.get_hcore = saved_hcore
        self._hcore = self.ll_mf.get_hcore()
        self.hl_rdms = hl_rdms
        self.log.info("Chemical Potential {:8.6e} gives Total electron deviation {:6.4e}".format(
                        chempot, nelec_hl - nelec_target))
        return nelec_hl - nelec_target