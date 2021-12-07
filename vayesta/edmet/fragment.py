import gc
import scipy.linalg
import pyscf.lib

from vayesta.dmet.fragment import DMETFragment
from vayesta.core.util import *
import dataclasses

from vayesta.solver import get_solver_class

from timeit import default_timer as timer

import numpy as np


class EDMETFragmentExit(Exception):
    pass


VALID_SOLVERS = ["EBFCI"]  # , "EBFCIQMC"]


class EDMETFragment(DMETFragment):
    @dataclasses.dataclass
    class Options(DMETFragment.Options):
        make_dd_moments: bool = True
        bos_occ_cutoff: int = NotSet
        old_sc_condition: bool = NotSet

    @dataclasses.dataclass
    class Results(DMETFragment.Results):
        dm_eb: np.ndarray = None
        eb_couplings: np.ndarray = None
        boson_freqs: tuple = None
        dd_mom0: np.ndarray = None
        dd_mom1: np.ndarray = None

    #    def __init__(self, *args, solver=None, **kwargs):
    #        super().__init__(*args, solver, **kwargs)

    def check_solver(self, solver):
        if solver not in VALID_SOLVERS:
            raise ValueError("Unknown solver: %s" % solver)

    @property
    def ov_active(self):
        return self.n_active_occ * self.n_active_vir

    @property
    def ov_mf(self):
        return self.base.nocc * self.base.nvir

    def get_rot_to_mf_ov(self):
        r_o, r_v = self.get_rot_to_mf()
        spat_rot = einsum("ij,ab->iajb", r_o, r_v).reshape((self.ov_active, self.ov_mf))
        res = np.zeros((2 * self.ov_active, 2 * self.ov_mf()))
        res[:self.ov_active, :self.ov_mf] = spat_rot
        res[self.ov_active:2 * self.ov_active, self.ov_mf:2 * self.ov_mf] = spat_rot
        return res

    def construct_bosons(self, rpa_moms):

        m0_aa, m0_ab, m0_bb = rpa_moms[0]
        m1_aa, m1_ab, m1_bb = rpa_moms[1]

        # Now just need to transform appropriately, svd as required and have bosonic degrees of freedom.
        # First, original coeffs.
        c_occ = self.mf.mo_coeff[:, self.mf.mo_occ > 0]
        c_vir = self.mf.mo_coeff[:, self.mf.mo_occ == 0]
        s = self.mf.get_ovlp()
        # Generate transformations of hf orbitals to generate cluster orbitals
        loc_transform_occ = np.dot(self.c_active_occ.T, np.dot(s, c_occ))
        loc_transform_vir = np.dot(self.c_active_vir.T, np.dot(s, c_vir))

        # print("transforms:", np.linalg.svd(loc_transform_occ)[1], np.linalg.svd(loc_transform_vir)[1])

        nocc_loc = self.n_active_occ
        nvir_loc = self.n_active_vir
        ov_loc = nocc_loc * nvir_loc
        # Local transform of overall p-h excitation space.
        loc_transform_ov = einsum("ij,ab->iajb", loc_transform_occ, loc_transform_vir).reshape((ov_loc, -1))
        ov_full = loc_transform_ov.shape[1]
        # Now grab null space of this, giving all environmental excitations.
        env_transform_ov = scipy.linalg.null_space(loc_transform_ov).T
        # print(m0_aa.shape, loc_transform_ov.shape, env_transform_ov.shape)
        nspat_env = env_transform_ov.shape[0]
        m0_a_interaction = np.concatenate(
            (einsum("pq,rp,sq->rs", m0_aa, loc_transform_ov, env_transform_ov),
             einsum("pq,rp,sq->rs", m0_ab, loc_transform_ov, env_transform_ov)
             ), axis=1)
        m0_b_interaction = np.concatenate(
            (einsum("pq,rp,sq->rs", m0_ab.T, loc_transform_ov, env_transform_ov),
             einsum("pq,rp,sq->rs", m0_bb, loc_transform_ov, env_transform_ov)
             ), axis=1)
        # Could now svd alpha and beta excitations separately; do together to ensure orthogonality of resultant degrees
        # of freedom.
        # ua, sa, va = np.linalg.svd(m0_a_interaction)
        # ub, sb, vb = np.linalg.svd(m0_b_interaction)
        m0_interaction = np.concatenate((m0_a_interaction, m0_b_interaction), axis=0)
        # print(m0_a_interaction.shape, m0_b_interaction.shape, m0_interaction.shape)
        u, s, v = np.linalg.svd(m0_interaction, full_matrices=False)
        want = s > 1e-8
        nbos = sum(want)
        self.log.info("Zeroth moment matching generated {:2d} cluster bosons".format(nbos))
        # print(s)
        # v gives rotation of environmental excitations to obtain effective bosonic degree of freedom.
        bosrot = v[want, :]

        # A-B = eta_0 (A+B) eta_0

        m0_loc = np.zeros((2 * ov_loc, 2 * ov_loc))
        # print(loc_transform_ov.shape, bosrot.shape, np.linalg.svd(loc_transform_ov)[1])
        m0_loc[:ov_loc, :ov_loc] = einsum("pq,rp,sq->rs", m0_aa, loc_transform_ov, loc_transform_ov)
        m0_loc[ov_loc:, ov_loc:] = einsum("pq,rp,sq->rs", m0_bb, loc_transform_ov, loc_transform_ov)
        m0_loc[:ov_loc, ov_loc:] = einsum("pq,rp,sq->rs", m0_ab, loc_transform_ov, loc_transform_ov)
        m0_loc[ov_loc:, :ov_loc] = m0_loc[:ov_loc, ov_loc:].T

        # print("m0 loc", m0_loc.shape)
        # print(m0_loc)

        # Check that interaction block is correctly structured.
        # m0_interact = dot(m0_interaction,v.T)#bosrot.T)
        # print("m0 interact", m0_interact.shape, m0_interaction.shape, v.shape)
        # print(m0_interact)
        # Projection of the full alpha excitation space into bosons.
        # print(ov_full, bosrot.shape, env_transform_ov.shape)
        bos_proj_a = np.dot(bosrot[:, :nspat_env], env_transform_ov)  # (nbos, ov_full)
        bos_proj_b = np.dot(bosrot[:, nspat_env:], env_transform_ov)  # (nbos, ov_full)
        bos_proj = np.concatenate((bos_proj_a, bos_proj_b), axis=1)
        # print(bos_proj.shape, np.linalg.svd(bos_proj)[1])

        # Define full rotation
        locrot = np.concatenate((np.concatenate((loc_transform_ov, np.zeros_like(loc_transform_ov)), axis=1),
                                 np.concatenate((np.zeros_like(loc_transform_ov), loc_transform_ov), axis=1)), axis=0)
        fullrot = np.concatenate((locrot,
                                  bos_proj), axis=0)
        # print("fullrot", fullrot.shape)
        # print(fullrot)
        m0_full = np.zeros((2 * ov_full, 2 * ov_full))
        m0_full[:ov_full, :ov_full] = m0_aa
        m0_full[ov_full:, ov_full:] = m0_bb
        m0_full[:ov_full, ov_full:] = m0_ab
        m0_full[ov_full:, :ov_full] = m0_ab.T

        n = scipy.linalg.null_space(fullrot)
        # print("Rotated m0:")
        # print(np.dot(fullrot, np.dot(m0_full, n)))

        m0_new = np.dot(fullrot, np.dot(m0_full, fullrot.T))
        apb_new = np.dot(fullrot, np.dot(rpa_moms["ApB"], fullrot.T))

        amb_new = np.dot(m0_new, np.dot(apb_new, m0_new))
        self.ApB_new = apb_new
        self.AmB_new = amb_new
        self.m0_new = m0_new
        # print("M0:")
        # print(m0_new)
        # print("ApB:")
        # print(apb_new)
        # print("AmB:")
        # print(amb_new)
        # print("Alternative AmB calculation:", rpa_moms["AmB"].shape, locrot.shape)
        # print(einsum("pn,n,qn->pq",locrot, rpa_moms["AmB"], locrot))
        maxdev = abs(amb_new[:2 * ov_loc, :2 * ov_loc] - einsum("pn,nm,qm->pq", locrot, rpa_moms["AmB"], locrot)).max()
        if maxdev > 1e-8:
            self.log.fatal(
                "Unexpected deviation from exact irreducible polarisation propagator: {:6.4e}".format(maxdev))
            raise EDMETFragmentExit

        # Now grab our bosonic parameters.
        va = np.zeros((nbos, self.n_active, self.n_active))
        vb = np.zeros((nbos, self.n_active, self.n_active))

        amb_bos = amb_new[2 * ov_loc:, 2 * ov_loc:]
        apb_bos = apb_new[2 * ov_loc:, 2 * ov_loc:]
        amb_eb = amb_new[2 * ov_loc:, :2 * ov_loc]
        apb_eb = apb_new[2 * ov_loc:, :2 * ov_loc]

        a_eb = 0.5 * (apb_eb + amb_eb)
        b_eb = 0.5 * (apb_eb - amb_eb)

        # Grab the bosonic couplings.
        va[:, :self.n_active_occ, self.n_active_occ:] = \
            a_eb[:, :ov_loc].reshape(nbos, self.n_active_occ, self.n_active_vir)
        vb[:, :self.n_active_occ, self.n_active_occ:] = \
            a_eb[:, ov_loc:2 * ov_loc].reshape(nbos, self.n_active_occ, self.n_active_vir)
        va[:, self.n_active_occ:, :self.n_active_occ] = \
            b_eb[:, :ov_loc].reshape(nbos, self.n_active_occ, self.n_active_vir).transpose([0, 2, 1])
        vb[:, self.n_active_occ:, :self.n_active_occ] = \
            b_eb[:, ov_loc:2 * ov_loc].reshape(nbos, self.n_active_occ, self.n_active_vir).transpose([0, 2, 1])

        # Perform quick bogliubov transform to decouple our bosons.
        rt_amb_bos = scipy.linalg.sqrtm(amb_bos)
        m = np.dot(rt_amb_bos, np.dot(apb_bos, rt_amb_bos))
        e, c = np.linalg.eigh(m)
        freqs = e ** (0.5)

        xpy = np.einsum("n,qp,pn->qn", freqs ** (-0.5), rt_amb_bos, c)
        xmy = np.einsum("n,qp,pn->qn", freqs ** (0.5), np.linalg.inv(rt_amb_bos), c)
        x = 0.5 * (xpy + xmy)
        y = 0.5 * (xpy - xmy)
        # Transform our couplings.
        va = np.einsum("npq,nm->mpq", va, x) + np.einsum("npq,nm->mqp", va, y)
        vb = np.einsum("npq,nm->mpq", vb, x) + np.einsum("npq,nm->mqp", vb, y)
        self.log.info("Cluster Bosons frequencies: " + str(freqs))
        # Check couplings are spin symmetric; this can be relaxed once we're using UHF and appropriate solvers.
        spin_deviation = abs(va - vb).max()
        if spin_deviation > 1e-6:
            self.log.warning("Boson couplings to different spin channels are significantly different; "
                             "largest deviation %6.4e", spin_deviation)
        # print(np.einsum("np8q,rp,sq->nrs", va, self.c_active, self.c_active))
        # print(np.einsum("npq,rp,sq->nrs", vb, self.c_active, self.c_active))
        return freqs, va, vb

    def set_up_fermionic_bath(self, bno_threshold=None, bno_number=None):
        """Set up the fermionic bath orbitals"""
        mo_coeff, mo_occ, nocc_frozen, nvir_frozen, nactive = \
            self.set_up_orbitals(bno_threshold, bno_number, construct_bath=True)
        # Want to return the rotation of the canonical HF orbitals which produce the cluster canonical orbitals.
        return self.get_rot_to_mf_ov()

    def define_bosons(self, rpa_mom, rot_ov, tol=1e-8):
        """Given the RPA zeroth moment between the fermionic cluster excitations and the rest of the space, define
        our cluster bosons.
        Note that this doesn't define our Hamiltonian, since we don't yet have the required portion of our
        zeroth moment for the bosonic degrees of freedom.
        """
        # Need to remove fermionic degrees of freedom from moment contribution. Null space of rotation matrix is size
        # N^4, so instead deduct projection onto fermionic space.
        env_mom = rpa_mom - dot(rpa_mom, rot_ov.T, np.linalg.pinv(rot_ov.T))
        # v defines the rotation of the mean-field excitation space specifying our bosons.
        u, s, v = np.linalg.svd(env_mom)
        want = s > tol
        self.nbos = sum(want)
        if self.nbos < len(s):
            self.log.info("Zeroth moment matching generated %d cluster bosons.Largest discarded singular value: %4.2e.",
                          self.nbos, s[~want].max())
        else:
            self.log.info("Zeroth moment matching generated %d cluster bosons.", self.nbos)

        # Calculate the relevant components of the zeroth moment- we don't want to recalculate these.
        self.r_bos = v[want, :]
        self.eta0_ferm = np.dot(rpa_mom, rot_ov.T)
        self.eta0_coupling = np.dot(env_mom, self.r_bos.T)
        return self.r_bos

    def construct_boson_hamil(self, eta0_bos, eps, xc_kernel):
        """Given the zeroth moment coupling of our bosons to the remainder of the space, along with stored information,
        generate the components of our interacting electron-boson Hamiltonian.
        """
        self.eta0_bos = np.dot(eta0_bos, self.r_bos.T)
        ov_rot = self.get_rot_to_mf_ov()
        # Get couplings between all fermionic and boson degrees of freedom.
        eris = self.get_eri_couplings(np.concatenate([ov_rot, self.r_bos], axis=0))
        # Depending upon the specifics of our construction we may need to deduct the contribution from this cluster
        # in the previous iteration to avoid double counting in future.
        xc_apb, xc_amb = self.get_xc_couplings(xc_kernel, np.concatenate([ov_rot, self.r_bos], axis=0))
        eps_loc = self.get_loc_eps(eps, np.concatenate([ov_rot, self.r_bos], axis=0))
        apb = eps_loc + 2 * eris + xc_apb
        # This is the bare amb.
        amb = eps_loc + xc_amb
        eta0 = np.zeros_Like(apb)
        eta0[:2 * self.ov_active, :2 * self.ov_active] = self.eta0_ferm
        eta0[:2 * self.ov_active, 2 * self.ov_active:] = self.eta0_coupling
        eta0[2 * self.ov_active:, :2 * self.ov_active] = self.eta0_coupling.T
        eta0[:2 * self.ov_active, :2 * self.ov_active] = self.eta0_bos

        renorm_amb = dot(eta0, apb, eta0)
        self.log.info("Maximum deviation in irreducible polarisation propagator=%6,4e",
                      abs(amb - renorm_amb)[:2 * self.ov_active, :2 * self.ov_active].max())
        a = 0.5 * (apb + renorm_amb)
        b = 0.5 * (apb - renorm_amb)
        couplings_aa = np.zeros((self.nbos, self.n_active, self.n_active))
        couplings_bb = np.zeros((self.nbos, self.n_active, self.n_active))

        couplings_aa[:, :self.n_active_occ, self.n_active_occ:] = a[2 * self.ov_active:, :self.ov_active].reshape(
            self.nbos, self.n_active_occ, self.n_active_vir)
        couplings_aa[:, self.n_active_occ:, :self.n_active_occ] = b[2 * self.ov_active:, :self.ov_active].reshape(
            self.nbos, self.n_active_occ, self.n_active_vir).transpose([0, 2, 1])
        couplings_bb[:, :self.n_active_occ, self.n_active_occ:] = \
            a[2 * self.ov_active:, self.ov_active:2 * self.ov_active].reshape(
                self.nbos, self.n_active_occ, self.n_active_vir)
        couplings_bb[:, self.n_active_occ:, :self.n_active_occ] = \
            b[2 * self.ov_active:, self.ov_active:2 * self.ov_active].reshape(
                self.nbos, self.n_active_occ, self.n_active_vir).transpose([0, 2, 1])

        a_bos = a[2 * self.ov_active:, 2 * self.ov_active:]
        b_bos = b[2 * self.ov_active:, 2 * self.ov_active:]
        self.bos_freqs, x, y = bogoliubov_decouple(a_bos + b_bos, a_bos - b_bos)

        couplings_aa = einsum("npq,nm->mpq", couplings_aa, x) + einsum("npq,nm->mqp", couplings_aa, y)
        couplings_bb = np.einsum("npq,nm->mpq", couplings_bb, x) + np.einsum("npq,nm->mqp", couplings_bb, y)
        self.couplings = (couplings_aa, couplings_bb)
        # These are currently the quantities before decoupling- shouldn't make any difference.
        self.apb = apb
        self.amb = renorm_amb
        self.eta0 = eta0

    def get_eri_couplings(self, rot):
        """Obtain eri in a space defined by an arbitrary rotation of the mean-field particle-hole excitations of our
        systems. Note that this should only really be used in the case that such a rotation cannot be described by a
        rotation of the underlying single-particle basis, since highly efficient routines already exist for this case..
        """
        if hasattr(self.base.mf, "with_df"):
            # Convert rot from full-space particle-hole excitations into AO pairs.
            rot = einsum("lia,pi,qa->lpq", rot.reshape((-1, self.base.nocc, self.base.nvir)),
                         dot(self.base.get_ovlp(), self.base.mo_coeff_occ),
                         dot(self.base.get_ovlp(), self.base.mo_coeff_vir))
            # Loop through cderis
            res = np.zeros((rot.shape[0], rot.shape[0]))
            for eri1 in self.mf.with_df.loop():
                l_ = einsum("npq,lpq->nl", pyscf.lib.unpack_tril(eri1), rot)
                res += dot(l_.T, l_)
            return res
        else:
            if self.mf._eri is not None:
                eris = pyscf.ao2mo.full(self.mf._eri, self.mf.mo_coeff, compact=False)
            else:
                eris = self.mol.ao2mo(self.mf.mo_coeff, compact=False)
            eris = eris[:self.base.nocc, self.base.nocc:, :self.base.nocc, self.base.nocc:].reshape(
                (self.ov_mf, self.ov_mf))
            return dot(rot, eris, rot.T)

    def get_xc_couplings(self, xc_kernel, rot):
        # Convert rot from full-space particle-hole excitations into AO pairs.
        rot = einsum("lia,pi,qa->lpq", rot.reshape((-1, self.base.nocc, self.base.nvir)),
                     dot(self.base.get_ovlp(), self.base.mo_coeff_occ),
                     dot(self.base.get_ovlp(), self.base.mo_coeff_vir))
        if hasattr(self.base.mf, "with_df"):
            # Store low-rank expression for xc kernel.
            # The contribution coupling just has an index change in our summation.
            la = einsum("npq,lpq->nl", xc_kernel, rot)
            lb = einsum("npq,lqp->nl", xc_kernel, rot)
            acontrib = dot(la.T, la)
            bcontrib = dot(la.T, lb)
            apb = acontrib + bcontrib
            amb = acontrib - bcontrib
        else:
            # Have full-rank expression for xc kernel.
            acontrib = einsum("lpq,pqrs,mrs->lm", rot, xc_kernel, rot)
            bcontrib = einsum("lpq,pqrs,msr->lm", rot, xc_kernel, rot)
            apb = acontrib + bcontrib
            amb = acontrib - bcontrib
        return apb, amb

    def get_loc_eps(self, eps, rot):
        return einsum("ln,n,mn->lm", rot, eps, rot)

    def kernel(self, bno_threshold=None, bno_number=None, solver=None, eris=None, construct_bath=False,
               chempot=None):
        """Solve the fragment with the specified solver and chemical potential."""
        solver = solver or self.solver

        # Create solver object
        t0 = timer()
        solver_opts = {}
        solver_opts['make_rdm1'] = self.opts.make_rdm1
        solver_opts['make_rdm2'] = self.opts.make_rdm2
        solver_opts['make_rdm_eb'] = self.opts.make_rdm1
        solver_opts['make_01_dd_mom'] = self.opts.make_dd_moments

        v_ext = None if chempot is None else - chempot * self.get_fragment_projector(self.c_active)

        cluster_solver_cls = get_solver_class(self.mf, solver)
        cluster_solver = cluster_solver_cls(
            self.bos_freqs, (self.couplings_aa, self.couplings_bb), self, self.mo_coeff, self.mf.mo_occ,
            nocc_frozen=self.n_frozen_occ, nvir_frozen=self.n_frozen_vir,
            v_ext=v_ext,
            bos_occ_cutoff=self.opts.bos_occ_cutoff, **solver_opts)
        solver_results = cluster_solver.kernel(eris=eris)
        self.log.timing("Time for %s solver:  %s", solver, time_string(timer() - t0))

        dd0 = solver_results.dd_mom0
        dd1 = solver_results.dd_mom1
        if self.opts.old_sc_condition:
            dd0 = [np.einsum("ppqq->pq", x) for x in dd0]
            dd1 = [np.einsum("ppqq->pq", x) for x in dd1]

        results = self.Results(
            fid=self.id,
            bno_threshold=bno_threshold,
            n_active=self.n_active,
            converged=solver_results.converged,
            e_corr=solver_results.e_corr,
            dm1=solver_results.dm1,
            dm2=solver_results.dm2,
            dm_eb=solver_results.rdm_eb,
            eb_couplings=np.array(self.couplings),
            boson_freqs=self.bos_freqs,
            dd_mom0=dd0,
            dd_mom1=dd1,
        )

        self.solver_results = solver_results
        self._results = results

        # Force GC to free memory
        m0 = get_used_memory()
        del cluster_solver, solver_results
        ndel = gc.collect()
        self.log.debugv("GC deleted %d objects and freed %.3f MB of memory", ndel, (get_used_memory() - m0) / 1e6)

        return results

    def get_edmet_energy_contrib(self):
        """Generate EDMET energy contribution, according to expression given in appendix of EDMET preprint"""
        e1, e2 = self.get_dmet_energy_contrib()
        c_act = self.c_active
        p_imp = self.get_fragment_projector(c_act)
        # Taken spin-averaged couplings for now; should actually be spin symmetric.
        couplings = (self._results.eb_couplings[0] + self._results.eb_couplings[1]) / 2
        dm_eb = self._results.dm_eb
        efb = 0.5 * (
                np.einsum("pr,npq,rqn", p_imp, couplings, dm_eb) +
                np.einsum("qr,npq,prn", p_imp, couplings, dm_eb)
        )
        return e1, e2, efb

    def construct_correlation_kernel_contrib(self, epsilon, m0_new, m1_new, eris=None):
        """
        Generate the contribution to the correlation kernel arising from this fragment.
        """
        # Get the ApB, AmB and m0 for this cluster. Note that this is pre-boson decoupling, but we don't actually care
        # about that here and it shouldn't change our answer.
        apb_orig = self.apb
        amb_orig = self.amb
        m0_orig = self.eta0

        # m0_new = self.results.dd_mom0
        # m1_new = self.results.dd_mom1

        nocc_loc = self.n_active_occ
        nvir_loc = self.n_active_vir
        ov_loc = nocc_loc * nvir_loc

        # Now want to construct rotations defining which degrees of freedom contribute to two-point quantities.
        occ_frag_rot = np.linalg.multi_dot([self.c_frag.T, self.base.get_ovlp(), self.c_active_occ])
        vir_frag_rot = np.linalg.multi_dot([self.c_frag.T, self.base.get_ovlp(), self.c_active_vir])

        if self.opts.old_sc_condition:
            # Then get projectors to local quantities in ov-basis. Note this needs to be stacked to apply to each spin
            # pairing separately.
            rot_ov_frag = np.einsum("pi,pa->pia", occ_frag_rot, vir_frag_rot).reshape((-1, ov_loc))
            # Get pseudo-inverse to map from frag to loc. Since occupied-virtual excitations aren't spanning this
            # isn't a simple transpose.
            rot_frag_ov = np.linalg.pinv(rot_ov_frag)
        else:
            # First, grab rotations from particle-hole excitations to fragment degrees of freedom, ignoring reordering
            rot_ov_frag = np.einsum("pi,qa->pqia", occ_frag_rot, vir_frag_rot).reshape((-1, ov_loc))
            # Set up matrix to map down to only a single index ordering.
            proj_to_order = np.zeros((self.n_frag,) * 4)
            for p in range(self.n_frag):
                for q in range(p + 1):
                    proj_to_order[p, q, p, q] = proj_to_order[q, p, p, q] = 1.0
            proj_to_order = proj_to_order.reshape((self.n_frag ** 2, self.n_frag, self.n_frag))
            # Now restrict to triangular portion of array
            proj_to_order = pyscf.lib.pack_tril(proj_to_order)
            proj_from_order = np.linalg.pinv(proj_to_order)
            # Now have rotation between single fragment ordering, and fragment particle-hole excits.
            rot_ov_frag = dot(proj_to_order.T, rot_ov_frag)
            # Get pseudo-inverse to map from frag to loc. Since occupied-virtual excitations aren't spanning this
            # isn't a simple transpose.
            rot_frag_ov = np.linalg.pinv(rot_ov_frag)
            m0_new = [dot(proj_to_order.T, x.reshape((self.n_frag ** 2,) * 2), proj_to_order) for x in m0_new]
            m1_new = [dot(proj_to_order.T, x.reshape((self.n_frag ** 2,) * 2), proj_to_order) for x in m1_new]

        # newmat = amb_orig.copy()

        def get_updated(orig, update, rot_ovf, rot_fov):
            """Given the original value of a block, the updated solver value, and rotations between appropriate spaces
            generate the updated value of the appropriate block."""
            # Generate difference in local, two-point excitation basis.
            diff = update - np.linalg.multi_dot([rot_ovf, orig, rot_ovf.T])
            return orig + np.linalg.multi_dot([rot_fov, diff, rot_fov.T])

        def get_updated_spincomponents(orig, update, rot_ov_frag, rot_frag_ov):
            newmat = orig.copy()

            newmat[:ov_loc, :ov_loc] = get_updated(newmat[:ov_loc, :ov_loc], update[0], rot_ov_frag, rot_frag_ov)
            newmat[:ov_loc, ov_loc:2 * ov_loc] = get_updated(newmat[:ov_loc, ov_loc:2 * ov_loc], update[1], rot_ov_frag,
                                                             rot_frag_ov)
            newmat[ov_loc:2 * ov_loc, :ov_loc] = newmat[:ov_loc, ov_loc:2 * ov_loc].T
            newmat[ov_loc:2 * ov_loc, ov_loc:2 * ov_loc] = get_updated(newmat[ov_loc:2 * ov_loc, ov_loc:2 * ov_loc],
                                                                       update[2],
                                                                       rot_ov_frag, rot_frag_ov)
            return newmat

        new_amb = get_updated_spincomponents(amb_orig, m1_new, rot_ov_frag, rot_frag_ov)
        new_m0 = get_updated_spincomponents(m0_orig, m0_new, rot_ov_frag, rot_frag_ov)
        new_m0_inv = np.linalg.inv(new_m0)
        new_apb = np.linalg.multi_dot([new_m0_inv, new_amb, new_m0_inv])

        new_a = 0.5 * (new_apb + new_amb)
        new_b = 0.5 * (new_apb - new_amb)

        r_occ, r_vir = self.get_rot_to_mf()
        # Given that our active orbitals are also canonical this should be diagonal, but calculating the whole
        # thing isn't prohibitive and might save pain.
        loc_eps = einsum("ia,ji,ba,ki,ca->jbkc", epsilon, r_occ, r_vir, r_occ, r_vir).reshape((ov_loc, ov_loc))
        # We want to actually consider the difference from the dRPA kernel. This is just the local eris in an OV basis.
        if eris is None:
            eris = self.base.get_eris_array(self.c_active)

        v = eris[:nocc_loc, nocc_loc:, :nocc_loc, nocc_loc:].reshape((ov_loc, ov_loc))

        occ_proj = self.get_fragment_projector(self.c_active_occ)
        vir_proj = self.get_fragment_projector(self.c_active_vir)

        def proj_all_indices(mat):
            """Obtains average over all possible projections of provided matrix, giving contribution to democratic
            partitioning from this cluster.
            """
            return (einsum("iajb,ik->kajb", mat, occ_proj) +
                    einsum("iajb,jk->iakb", mat, occ_proj) +
                    einsum("iajb,ac->icjb", mat, vir_proj) +
                    einsum("iajb,bc->iajc", mat, vir_proj)) / 4.0

        # Now calculate all spin components; could double check spin symmetry of ab terms if wanted.
        # This deducts the equivalent values at the level of dRPA, reshapes into fermionic indices, and performs
        # projection to only the fragment portions of all indices.
        newshape = (nocc_loc, nvir_loc, nocc_loc, nvir_loc)
        v_a_aa = proj_all_indices((new_a[:ov_loc, :ov_loc] - loc_eps - v).reshape(newshape))
        v_a_bb = proj_all_indices((new_a[ov_loc: 2 * ov_loc, ov_loc: 2 * ov_loc] - loc_eps - v).reshape(newshape))
        v_a_ab = proj_all_indices((new_a[:ov_loc:, ov_loc: 2 * ov_loc] - v).reshape(newshape))
        v_b_aa = proj_all_indices((new_b[:ov_loc, :ov_loc] - v).reshape(newshape))
        v_b_bb = proj_all_indices((new_b[ov_loc: 2 * ov_loc, ov_loc: 2 * ov_loc] - v).reshape(newshape))
        v_b_ab = proj_all_indices((new_b[:ov_loc:, ov_loc: 2 * ov_loc] - v).reshape(newshape))

        return v_a_aa, v_a_ab, v_a_bb, v_b_aa, v_b_ab, v_b_bb

    def get_correlation_kernel_contrib(self, epsilon, dd0, dd1, eris=None):

        if self.sym_parent is None:
            v_a_aa, v_a_ab, v_a_bb, v_b_aa, v_b_ab, v_b_bb = self.construct_correlation_kernel_contrib(
                epsilon, dd0, dd1, eris)
        else:
            v_a_aa, v_a_ab, v_a_bb, v_b_aa, v_b_ab, v_b_bb = self.sym_parent.construct_correlation_kernel_contrib(
                epsilon, dd0, dd1, eris)
        # Now need to project back out to full space. This requires an additional factor of the overlap in ou
        # coefficients.
        c_occ = np.dot(self.base.get_ovlp(), self.c_active_occ)
        c_vir = np.dot(self.base.get_ovlp(), self.c_active_vir)
        v_aa = (
                einsum("iajb,pi,qa,rj,sb->pqrs", v_a_aa, c_occ, c_vir, c_occ, c_vir) +
                einsum("iajb,pi,qa,rj,sb->pqsr", v_b_aa, c_occ, c_vir, c_occ, c_vir))
        v_ab = (
                einsum("iajb,pi,qa,rj,sb->pqrs", v_a_ab, c_occ, c_vir, c_occ, c_vir) +
                einsum("iajb,pi,qa,rj,sb->pqsr", v_b_ab, c_occ, c_vir, c_occ, c_vir))
        v_bb = (
                einsum("iajb,pi,qa,rj,sb->pqrs", v_a_bb, c_occ, c_vir, c_occ, c_vir) +
                einsum("iajb,pi,qa,rj,sb->pqsr", v_b_bb, c_occ, c_vir, c_occ, c_vir))
        return v_aa, v_ab, v_bb


def bogoliubov_decouple(apb, amb):
    # Perform quick bogliubov transform to decouple our bosons.
    rt_amb = scipy.linalg.sqrtm(amb)
    m = dot(rt_amb, apb, rt_amb)
    e, c = np.linalg.eigh(m)
    freqs = e ** (0.5)

    xpy = np.einsum("n,qp,pn->qn", freqs ** (-0.5), rt_amb, c)
    xmy = np.einsum("n,qp,pn->qn", freqs ** (0.5), np.linalg.inv(rt_amb), c)
    x = 0.5 * (xpy + xmy)
    y = 0.5 * (xpy - xmy)
    return freqs, x, y
