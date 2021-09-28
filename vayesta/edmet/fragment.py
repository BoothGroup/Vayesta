
import gc
import scipy.linalg

from vayesta.dmet.fragment import DMETFragment
from vayesta.core.util import *
import dataclasses

from vayesta.solver import get_solver_class
from vayesta.ewf import helper, psubspace

from timeit import default_timer as timer

import numpy as np


class EDMETFragmentExit(Exception):
    pass


class EDMETFragment(DMETFragment):

    @dataclasses.dataclass
    class Options(DMETFragment.Options):
        bos_occ_cutoff: int = NotSet

    @dataclasses.dataclass
    class Results(DMETFragment.Results):
        dm_eb: np.ndarray = None
        eb_couplings: np.ndarray = None

#    def __init__(self, *args, solver=None, **kwargs):
#        super().__init__(*args, solver, **kwargs)

    def construct_bosons(self, rpa_moms):

        m0_aa, m0_ab, m0_bb = rpa_moms[0]
        m1_aa, m1_ab, m1_bb = rpa_moms[1]

        # Now just need to transform appropriately, svd as required and have bosonic degrees of freedom.
        # First, original coeffs.
        c_occ = self.mf.mo_coeff[:,self.mf.mo_occ>0]
        c_vir = self.mf.mo_coeff[:, self.mf.mo_occ == 0]
        s = self.mf.get_ovlp()
        # Generate transformations of hf orbitals to generate cluster orbitals
        loc_transform_occ = np.dot(self.c_active_occ.T, np.dot(s, c_occ))
        loc_transform_vir = np.dot(self.c_active_vir.T, np.dot(s, c_vir))

        #print("transforms:", np.linalg.svd(loc_transform_occ)[1], np.linalg.svd(loc_transform_vir)[1])

        nocc_loc = loc_transform_occ.shape[0]
        nvir_loc = loc_transform_vir.shape[0]
        ov_loc = nocc_loc * nvir_loc
        # Local transform of overall p-h excitation space.
        loc_transform_ov = einsum("ij,ab->iajb", loc_transform_occ, loc_transform_vir).reshape((ov_loc, -1))
        ov_full = loc_transform_ov.shape[1]
        # Now grab null space of this, giving all environmental excitations.
        env_transform_ov = scipy.linalg.null_space(loc_transform_ov).T
        #print(m0_aa.shape, loc_transform_ov.shape, env_transform_ov.shape)
        nspat_env = env_transform_ov.shape[0]
        m0_a_interaction = np.concatenate(
            (einsum("pq,rp,sq->rs", m0_aa, loc_transform_ov, env_transform_ov),
            einsum("pq,rp,sq->rs", m0_ab, loc_transform_ov, env_transform_ov)
        ), axis = 1)
        m0_b_interaction = np.concatenate(
            (einsum("pq,rp,sq->rs", m0_ab.T, loc_transform_ov, env_transform_ov),
             einsum("pq,rp,sq->rs", m0_bb, loc_transform_ov, env_transform_ov)
             ), axis=1)
        # Could now svd alpha and beta excitations separately; do together to ensure orthogonality

        #ua, sa, va = np.linalg.svd(m0_a_interaction)
        #ub, sb, vb = np.linalg.svd(m0_b_interaction)
        m0_interaction = np.concatenate((m0_a_interaction, m0_b_interaction), axis = 0)
        #print(m0_a_interaction.shape, m0_b_interaction.shape, m0_interaction.shape)
        u, s, v = np.linalg.svd(m0_interaction, full_matrices=False)
        want = s > 1e-8
        nbos = sum(want)
        self.log.info("Zeroth moment matching generated {:2d} cluster bosons".format(nbos))
        #print("NBOSONS:", nbos)
        #print(s)
        # v gives rotation of environmental excitations to obtain effective bosonic degree of freedom.
        bosrot = v[want,:]

        # A-B = eta_0 (A+B) eta_0

        m0_loc = np.zeros((2 * ov_loc, 2 * ov_loc))
        #print(loc_transform_ov.shape, bosrot.shape, np.linalg.svd(loc_transform_ov)[1])
        m0_loc[:ov_loc,:ov_loc] = einsum("pq,rp,sq->rs", m0_aa, loc_transform_ov, loc_transform_ov)
        m0_loc[ov_loc:, ov_loc:] = einsum("pq,rp,sq->rs", m0_bb, loc_transform_ov, loc_transform_ov)
        m0_loc[:ov_loc, ov_loc:] = einsum("pq,rp,sq->rs", m0_ab, loc_transform_ov, loc_transform_ov)
        m0_loc[ov_loc:, :ov_loc] = m0_loc[:ov_loc, ov_loc:].T

        #print("m0 loc", m0_loc.shape)
        #print(m0_loc)

        # Check that interaction block is correctly structured.
        m0_interact = dot(m0_interaction,v.T)#bosrot.T)
        #print("m0 interact", m0_interact.shape, m0_interaction.shape, v.shape)
        #print(m0_interact)
        # Projection of the full alpha excitation space into bosons.
        #print(ov_full, bosrot.shape, env_transform_ov.shape)
        bos_proj_a = np.dot(bosrot[:,:nspat_env], env_transform_ov) # (nbos, ov_full)
        bos_proj_b = np.dot(bosrot[:, nspat_env:], env_transform_ov)  # (nbos, ov_full)
        bos_proj = np.concatenate((bos_proj_a, bos_proj_b), axis=1)
        #print(bos_proj.shape, np.linalg.svd(bos_proj)[1])

        # Define full rotation
        locrot = np.concatenate((np.concatenate((loc_transform_ov, np.zeros_like(loc_transform_ov)), axis=1),
                                  np.concatenate((np.zeros_like(loc_transform_ov), loc_transform_ov), axis=1)), axis = 0)
        fullrot = np.concatenate((locrot,
                                  bos_proj), axis = 0)
        #print("fullrot", fullrot.shape)
        #print(fullrot)
        m0_full = np.zeros((2*ov_full, 2*ov_full))
        m0_full[:ov_full, :ov_full] = m0_aa
        m0_full[ov_full:, ov_full:] = m0_bb
        m0_full[:ov_full, ov_full:] = m0_ab
        m0_full[ov_full:, :ov_full] = m0_ab.T

        n = scipy.linalg.null_space(fullrot)
        #print("Rotated m0:")
        #print(np.dot(fullrot, np.dot(m0_full, n)))

        m0_new = np.dot(fullrot, np.dot(m0_full, fullrot.T))
        ApB_new = np.dot(fullrot, np.dot(rpa_moms["ApB"], fullrot.T))

        AmB_new = np.dot(m0_new, np.dot(ApB_new, m0_new))
        #print("M0:")
        #print(m0_new)
        #print("ApB:")
        #print(ApB_new)
        #print("AmB:")
        #print(AmB_new)
        #print("Alternative AmB calculation:", rpa_moms["AmB"].shape, locrot.shape)
        #print(einsum("pn,n,qn->pq",locrot, rpa_moms["AmB"], locrot))

        maxdev = abs(AmB_new[:2*ov_loc,:2*ov_loc] -  einsum("pn,n,qn->pq",locrot, rpa_moms["AmB"], locrot)).max()
        if maxdev > 1e-8:
            self.log.fatal("Unexpected deviation from exact irreducible polarisation propagator: {:6.4e}".format(maxdev))
            raise EDMETFragmentExit

        # Now grab our bosonic parameters.
        Va = np.zeros((nbos, self.n_active, self.n_active))
        Vb = np.zeros((nbos, self.n_active, self.n_active))

        AmB_bos = AmB_new[2*ov_loc:, 2*ov_loc:]
        ApB_bos = ApB_new[2 * ov_loc:, 2 * ov_loc:]
        AmB_eb = AmB_new[2 * ov_loc:, :2 * ov_loc]
        ApB_eb = ApB_new[2 * ov_loc:, :2 * ov_loc]

        A_eb = 0.5 * (ApB_eb + AmB_eb)
        B_eb = 0.5 * (ApB_eb - AmB_eb)

        # Grab the bosonic couplings.
        Va[:, :self.n_active_occ, self.n_active_occ:] =\
            A_eb[:,:ov_loc].reshape(nbos, self.n_active_occ, self.n_active_vir)
        Vb[:, :self.n_active_occ, self.n_active_occ:] = \
            A_eb[:, ov_loc:2*ov_loc].reshape(nbos, self.n_active_occ, self.n_active_vir)
        Va[:, self.n_active_occ:, :self.n_active_occ] = \
            B_eb[:, :ov_loc].reshape(nbos, self.n_active_occ, self.n_active_vir).transpose([0,2,1])
        Vb[:, self.n_active_occ:, :self.n_active_occ] = \
            B_eb[:, ov_loc:2*ov_loc].reshape(nbos, self.n_active_occ, self.n_active_vir).transpose([0,2,1])

        # Perform quick bogliubov transform to decouple our bosons.
        rt_AmB_bos = scipy.linalg.sqrtm(AmB_bos)
        M = np.dot(rt_AmB_bos, np.dot(ApB_bos, rt_AmB_bos))
        # Th
        e,c = np.linalg.eigh(M)
        freqs =  e ** (0.5)

        XpY = np.einsum("n,qp,pn->qn", freqs ** (-0.5), rt_AmB_bos, c)
        XmY = np.einsum("n,qp,pn->qn", freqs ** (0.5), np.linalg.inv(rt_AmB_bos), c)
        X = 0.5 * (XpY + XmY)
        Y = 0.5 * (XpY - XmY)
        # Transform our couplings.
        Va = np.einsum("npq,nm->mpq", Va, X) + np.einsum("npq,nm->mqp", Va, Y)
        Vb = np.einsum("npq,nm->mpq", Vb, X) + np.einsum("npq,nm->mqp", Vb, Y)
        self.log.info("Cluster Bosons frequencies: " + str(freqs))
        # Check couplings are spin symmetric; this can be relaxed once we're using UHF and appropriate solvers.
        spin_deviation = abs(Va - Vb).max()
        if spin_deviation > 1e-6:
            self.log.warning("Boson couplings to different spin channels are significantly different; "
                             "largest deviation {:6.4e}".format(spin_deviation))

        #print(np.einsum("npq,rp,sq->nrs", Va, self.c_active, self.c_active))
        #print(np.einsum("npq,rp,sq->nrs", Vb, self.c_active, self.c_active))
        return freqs, Va, Vb

    def kernel(self, rpa_moms, bno_threshold=None, bno_number=None, solver=None, eris=None, construct_bath=False,
               chempot = None):
        # First set up fermionic degrees of freedom
        mo_coeff, mo_occ, nocc_frozen, nvir_frozen, nactive = \
                            self.set_up_orbitals(bno_threshold, bno_number, construct_bath)

        # Now generate bosonic bath.
        freqs, Va, Vb = self.construct_bosons(rpa_moms)

        solver = solver or self.solver

        # Create solver object
        t0 = timer()
        solver_opts = {}
        solver_opts['make_rdm1'] = self.opts.make_rdm1
        solver_opts['make_rdm2'] = self.opts.make_rdm2
        solver_opts['make_rdm_eb'] = self.opts.make_rdm1

        v_ext = None if chempot is None else - chempot * self.get_fragment_projector(self.c_active)


        cluster_solver_cls = get_solver_class(solver)
        cluster_solver = cluster_solver_cls(
            freqs, (Va, Vb), self, mo_coeff, mo_occ, nocc_frozen=nocc_frozen, nvir_frozen=nvir_frozen, v_ext = v_ext,
            **solver_opts)
        solver_results = cluster_solver.kernel(bos_occ_cutoff=self.opts.bos_occ_cutoff, eris=eris)
        self.log.timing("Time for %s solver:  %s", solver, time_string(timer()-t0))

        results = self.Results(
                fid=self.id,
                bno_threshold=bno_threshold,
                n_active=nactive,
                converged=solver_results.converged,
                e_corr=solver_results.e_corr,
                dm1 = solver_results.dm1,
                dm2 = solver_results.dm2,
                dm_eb = solver_results.rdm_eb,
                eb_couplings=np.array((Va,Vb)))

        self.solver_results = solver_results
        self._results = results

        # Force GC to free memory
        m0 = get_used_memory()
        del cluster_solver, solver_results
        ndel = gc.collect()
        self.log.debugv("GC deleted %d objects and freed %.3f MB of memory", ndel, (get_used_memory()-m0)/1e6)

        return results

    def get_edmet_energy_contrib(self):
        """Generate EDMET energy contribution, according to expression given in appendix of EDMET preprint"""
        e1, e2 = self.get_dmet_energy_contrib()
        c_act = self.c_active
        P_imp = self.get_fragment_projector(c_act)
        # Taken spin-averaged couplings for now; should actually be spin symmetric.
        couplings = (self._results.eb_couplings[0] + self._results.eb_couplings[1]) / 2
        dm_eb = self._results.dm_eb
        efb = 0.5 * (
            np.einsum("pr,npq,rqn", P_imp, couplings, dm_eb) +
            np.einsum("qr,npq,prn", P_imp, couplings, dm_eb)
        )
        return e1, e2, efb




