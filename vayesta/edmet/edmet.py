import dataclasses
from timeit import default_timer as timer

import numpy as np
import scipy
import scipy.linalg

from vayesta.core.util import *
from vayesta.dmet import RDMET
from vayesta.dmet.sdp_sc import perform_SDP_fit
from vayesta.dmet.updates import MixUpdate, DIISUpdate
from vayesta.rpa import ssRPA, ssRIRPA
from .fragment import VALID_SOLVERS, EDMETFragment, EDMETFragmentExit


@dataclasses.dataclass
class EDMETResults:
    cluster_sizes: np.ndarray = None
    e_corr: float = None


class EDMET(RDMET):
    @dataclasses.dataclass
    class Options(RDMET.Options):
        maxiter: int = 1
        make_dd_moments: bool = NotSet
        old_sc_condition: bool = False
        max_bos: int = np.inf

    Fragment = EDMETFragment

    def __init__(self, mf, bno_threshold=np.inf, solver='EBFCI', options=None, log=None, max_boson_occ=2, **kwargs):
        super().__init__(mf, bno_threshold, solver, options, log, **kwargs)
        self.interaction_kernel = None
        # Need to calculate dd moments for self-consistency to work.
        self.opts.make_dd_moments = True  # self.opts.maxiter > 1
        self.opts.solver_options["max_boson_occ"] = max_boson_occ

    @property
    def with_df(self):
        return hasattr(self.mf, "with_df")

    @property
    def eps(self):
        eps = np.zeros((self.nocc, self.nvir))
        eps = eps + self.mo_energy[self.nocc:]
        eps = (eps.T - self.mo_energy[:self.nocc]).T
        eps = eps.reshape(-1)
        return eps, eps

    def check_solver(self, solver):
        if solver not in VALID_SOLVERS:
            raise ValueError("Unknown solver: %s" % solver)

    def kernel(self, bno_threshold=None):

        t_start = timer()

        if self.nfrag == 0:
            raise ValueError("No fragments defined for calculation.")

        maxiter = self.opts.maxiter
        # rdm = self.mf.make_rdm1()
        bno_thr = bno_threshold or self.bno_threshold
        #if bno_thr < np.inf and maxiter > 1:
        #    raise NotImplementedError("MP2 bath calculation is currently ignoring the correlation potential, so does"
        #                              " not work properly for self-consistent calculations.")
        # Initialise parameters for self-consistency iteration
        fock = self.get_fock()
        if self.vcorr is None:
            self.vcorr = np.zeros((self.nao,) * 2)
        else:
            self.log.info("Starting from previous correlation potential.")
        if self.with_df:
            # Store alpha and beta components separately.
            self.xc_kernel = [[np.zeros((0, self.nao, self.nao))] * 2] * 2
        else:
            # Store alpha-alpha, alpha-beta and beta-beta components separately.
            self.xc_kernel = [np.zeros((self.nao,) * 4)] * 3

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
            self.log.info("Now running iteration= %2d", iteration)
            self.log.info("****************************************************")
            mo_energy, mo_coeff = mf.eig(fock + self.vcorr, self.get_ovlp())
            self.update_mf(mo_coeff, mo_energy)
            if self.opts.charge_consistent:
                fock = mf.get_fock()
            self.set_up_fragments(sym_parents, bno_threshold=bno_thr)
            # Need to optimise a global chemical potential to ensure electron number is converged.
            nelec_mf = self.check_fragment_nelectron()
            if type(nelec_mf) == tuple:
                nelec_mf = sum(nelec_mf)

            def electron_err(cpt):
                err = self.calc_electron_number_defect(cpt, np.inf, nelec_mf, sym_parents, nsym)
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
                                                 xtol=self.opts.max_elec_err * nelec_mf)
                # self.opts.max_elec_err * nelec_mf)
                self.log.info("Converged chemical potential: %6.4e", cpt)
                # Recalculate to ensure all fragments have up-to-date info. Brentq strangely seems to do an extra
                # calculation at the end...
                electron_err(cpt)
            else:
                self.log.info("Previous chemical potential still suitable")

            e1, e2, efb, emf = 0.0, 0.0, 0.0, 0.0
            for x, frag in enumerate(sym_parents):
                e1_contrib, e2_contrib, efb_contrib = frag.get_edmet_energy_contrib()
                e1 += e1_contrib * nsym[x]
                e2 += e2_contrib * nsym[x]
                efb += efb_contrib * nsym[x]
                emf += frag.get_fragment_mf_energy()

            self.e_corr = e1 + e2 + efb + self.e_nonlocal - emf
            self.log.info("Total EDMET energy {:8.4f}".format(self.e_tot))
            self.log.info(
                "Energy Contributions: 1-body={:8.4f}, 2-body={:8.4f}, coupled-boson={:8.4f}, nonlocal correlation "
                "energy={:8.4f}".format(e1, e2, efb, self.e_nonlocal))

            # Want to do coupled DIIS optimisation of high-level rdms and local dd response moments.
            [curr_rdms, curr_dd0, curr_dd1], delta_prop = self.updater.update([self.hl_rdms, self.hl_dd0, self.hl_dd1])

            self.log.info("Change in high-level properties: {:6.4e}".format(delta_prop))
            # Now for the DMET self-consistency!
            self.log.info("Now running DMET correlation potential fitting")
            vcorr_new = self.update_vcorr(fock, curr_rdms)
            delta = sum((vcorr_new - self.vcorr).reshape(-1) ** 2) ** (0.5)
            self.log.info("Delta Vcorr {:6.4e}".format(delta))

            xc_kernel_new = self.get_updated_correlation_kernel(curr_dd0, curr_dd1, sym_parents, sym_children)
            if delta < self.opts.conv_tol and delta_prop < self.opts.conv_tol:
                self.converged = True
                self.log.info("DMET converged after %d iterations" % iteration)
                break
            else:
                self.vcorr = vcorr_new
                self.xc_kernel = xc_kernel_new
        else:
            self.log.error("Self-consistency not reached in {} iterations.".format(maxiter))
        # Now have final results.
        self.print_results()
        self.timing = timer() - t_start
        self.log.info("Total wall time:  %s", time_string(self.timing))

        self.log.info("All done.")

    def set_up_fragments(self, sym_parents, bno_threshold=None):

        # First, set up and run RPA. Note that our self-consistency only couples same-spin excitations so we can
        # solve a subset of the RPA equations.
        if self.with_df:
            # Set up for RIRPPA zeroth moment calculation.
            rpa = ssRIRPA(self.mf, self.xc_kernel, self.log)
            # Get fermionic bath set up, and calculate the cluster excitation space.
            rot_ovs = [f.set_up_fermionic_bath(bno_threshold) for f in sym_parents]
            target_rot = np.concatenate(rot_ovs, axis=0)
            if target_rot.shape[0] > 0:
                mom0_interact, est_error = rpa.kernel_moms(target_rot, npoints=48)
            else:
                mom0_interact = np.zeros_like(target_rot)
            # Get appropriate slices to obtain required active spaces.
            ovs_active = [f.ov_active_tot for f in sym_parents]
            ovs_active_slices = [slice(sum(ovs_active[:i]), sum(ovs_active[:i + 1])) for i in
                                 range(len(sym_parents))]
            # Use interaction component of moment to generate bosonic degrees of freedom.
            rot_bos = [f.define_bosons(mom0_interact[sl, :]) for (f, sl) in zip(sym_parents, ovs_active_slices)]
            nbos = [x.shape[0] for x in rot_bos]
            bos_slices = [slice(sum(nbos[:i]), sum(nbos[:i + 1])) for i in range(len(sym_parents))]
            if sum(nbos) > 0:
                # Calculate zeroth moment of bosonic degrees of freedom.
                mom0_bos, est_error = rpa.kernel_moms(np.concatenate(rot_bos, axis=0), npoints=48)
            else:
                mom0_bos = np.zeros((sum(nbos), mom0_interact.shape[1]))
            eps = np.concatenate(self.eps)
            # Can then invert relation to generate coupled electron-boson Hamiltonian.
            for f, sl in zip(sym_parents, bos_slices):
                f.construct_boson_hamil(mom0_bos[sl, :], eps, self.xc_kernel)
        else:
            rpa = ssRPA(self.mf, self.log)
            # We need to explicitly solve RPA equations before anything.
            rpa.kernel(xc_kernel=self.xc_kernel)
            self.log.info("RPA particle-hole gap %4.2e", rpa.freqs_ss.min())
            # Then generate full RPA moments.
            mom0 = rpa.gen_moms(0, self.xc_kernel)[0]
            eps = np.concatenate(self.eps)
            e_nonlocal = rpa.calc_energy_correction(self.xc_kernel, version=3)
            self.log.info("RPA total energy=%6.4e", e_nonlocal)
            #k = rpa.get_k()
            #e_nonlocal = (einsum("pq,qp->", mom0, k) - k.trace()) / 4.0

            for f in sym_parents:
                rot_ov = f.set_up_fermionic_bath(bno_threshold)
                mom0_interact = dot(rot_ov, mom0)
                rot_bos = f.define_bosons(mom0_interact)
                mom0_bos = dot(rot_bos, mom0)
                e_nonlocal -= f.construct_boson_hamil(mom0_bos, eps, self.xc_kernel)
            self.e_nonlocal = e_nonlocal

    def calc_electron_number_defect(self, chempot, bno_thr, nelec_target, parent_fragments, nsym,
                                    construct_bath=True):
        self.log.info("Running chemical potential={:8.6e}".format(chempot))
        # Save original one-body hamiltonian calculation.
        saved_hcore = self.mf.get_hcore

        hl_rdms = [None] * len(parent_fragments)
        hl_dd0 = [None] * len(parent_fragments)
        hl_dd1 = [None] * len(parent_fragments)
        nelec_hl = 0.0
        exit = False
        for x, frag in enumerate(parent_fragments):
            msg = "Now running %s" % (frag)
            self.log.info(msg)
            self.log.info(len(msg) * "*")
            self.log.changeIndentLevel(1)

            try:
                result = frag.kernel(bno_threshold=bno_thr, construct_bath=construct_bath, chempot=chempot)
            except EDMETFragmentExit as e:
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

            # dd moments are already in fragment basis
            hl_dd0[x] = frag.results.dd_mom0
            hl_dd1[x] = frag.results.dd_mom1
            nelec_hl += frag.get_nelectron_hl() * nsym[x]

        self.hl_rdms = [f.get_frag_hl_dm() for f in parent_fragments]
        self.hl_dd0 = hl_dd0
        self.hl_dd1 = hl_dd1
        self.log.info("Chemical Potential {:8.6e} gives Total electron deviation {:6.4e}".format(
            chempot, nelec_hl - nelec_target))
        return nelec_hl - nelec_target

    def get_updated_correlation_kernel(self, curr_dd0, curr_dd1, sym_parents, sym_children):
        """
        Generate the update to our RPA exchange-correlation kernel this iteration.
        """
        eps = np.concatenate(self.eps)
        # Separate into spin components; in RHF case we still expect aaaa and aabb components to differ.
        if self.with_df:
            k = [[np.zeros((0, self.nao, self.nao))] * 2, [np.zeros((0, self.nao, self.nao))] * 2]

            def combine(old, new):
                return [[np.concatenate([a, b], axis=0) for a, b in zip(x, y)] for (x, y) in zip(old, new)]
        else:
            k = [np.zeros([self.nao] * 4) for x in range(3)]

            def combine(old, new):
                return [old[x] + new[x] for x in range(3)]
        for d0, d1, parent, children in zip(curr_dd0, curr_dd1, sym_parents, sym_children):
            local_contrib = parent.construct_correlation_kernel_contrib(eps, d0, d1, eris=None)
            contrib = parent.get_correlation_kernel_contrib(local_contrib)
            k = combine(k, contrib)
            for child in children:
                contrib = child.get_correlation_kernel_contrib(local_contrib)
                k = combine(k, contrib)
        return tuple(k)

REDMET = EDMET