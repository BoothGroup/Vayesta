
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
        make_dd_moments: bool = True
        bos_occ_cutoff: int = NotSet

    @dataclasses.dataclass
    class Results(DMETFragment.Results):
        dm_eb: np.ndarray = None
        eb_couplings: np.ndarray = None
        boson_freqs: np.ndarray = None
        dd_mom0: np.ndarray = None
        dd_mom1: np.ndarray = None

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

        nocc_loc = self.n_active_occ
        nvir_loc = self.n_active_vir
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
        # Could now svd alpha and beta excitations separately; do together to ensure orthogonality of resultant degrees
        # of freedom.
        #ua, sa, va = np.linalg.svd(m0_a_interaction)
        #ub, sb, vb = np.linalg.svd(m0_b_interaction)
        m0_interaction = np.concatenate((m0_a_interaction, m0_b_interaction), axis = 0)
        #print(m0_a_interaction.shape, m0_b_interaction.shape, m0_interaction.shape)
        u, s, v = np.linalg.svd(m0_interaction, full_matrices=False)
        want = s > 1e-8
        nbos = sum(want)
        self.log.info("Zeroth moment matching generated {:2d} cluster bosons".format(nbos))
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
        #m0_interact = dot(m0_interaction,v.T)#bosrot.T)
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
        self.ApB_new = ApB_new
        self.AmB_new = AmB_new
        self.m0_new = m0_new
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
        freqs = e ** (0.5)

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
        solver_opts['make_01_dd_mom'] = self.opts.make_dd_moments

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
                eb_couplings=np.array((Va,Vb)),
                boson_freqs=freqs,
                dd_mom0=solver_results.dd_mom0,
                dd_mom1=solver_results.dd_mom1,
        )

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

    def construct_correlation_kernel_contrib(self, epsilon, m0_new, m1_new, eris = None):
        """
        Generate the contribution to the correlation kernel arising from this fragment.
        """
        # Get the ApB, AmB and m0 for this cluster. Note that this is pre-boson decoupling, but we don't actually care
        # about that here and it shouldn't change our answer.
        ApB_orig = self.ApB_new
        AmB_orig = self.AmB_new
        m0_orig = self.m0_new

        #m0_new = self.results.dd_mom0
        #m1_new = self.results.dd_mom1


        nocc_loc = self.n_active_occ
        nvir_loc = self.n_active_vir
        ov_loc = nocc_loc * nvir_loc

        # Now want to construct rotations defining which degrees of freedom contribute to two-point quantities.
        occ_frag_rot = np.linalg.multi_dot([self.c_frag.T, self.base.get_ovlp(), self.c_active_occ])
        vir_frag_rot = np.linalg.multi_dot([self.c_frag.T, self.base.get_ovlp(), self.c_active_vir])
        # Then get projectors to local quantities in ov-basis. Note this needs to be stacked to apply to each spin
        # pairing separately.
        rot_loc_frag = np.einsum("pi,pa->pia", occ_frag_rot, vir_frag_rot).reshape((-1, ov_loc))
        # Get pseudo-inverse to map from frag to loc. Since occupied-virtual excitations aren't spanning this isn't a
        # simple transpose.
        rot_frag_loc = np.linalg.pinv(rot_loc_frag)

        #newmat = AmB_orig.copy()

        def get_updated(orig, update, rot_lf, rot_fl):
            """Given the original value of a block, the updated solver value, and rotations between appropriate spaces
            generate the updated value of the appropriate block."""
            # Generate difference in local, two-point excitation basis.
            diff = update - np.linalg.multi_dot([rot_lf, orig, rot_lf.T])
            return orig + np.linalg.multi_dot([rot_fl, diff, rot_fl.T])

        def get_updated_spincomponents(orig, update, rot_loc_frag, rot_frag_loc):
            newmat = orig.copy()

            newmat[:ov_loc, :ov_loc] = get_updated(newmat[:ov_loc, :ov_loc], update[0], rot_loc_frag, rot_frag_loc)
            newmat[:ov_loc, ov_loc:2*ov_loc] = get_updated(newmat[:ov_loc, ov_loc:2*ov_loc], update[1], rot_loc_frag, rot_frag_loc)
            newmat[ov_loc:2*ov_loc, :ov_loc] = newmat[:ov_loc, ov_loc:2 * ov_loc].T
            newmat[ov_loc:2*ov_loc, ov_loc:2*ov_loc] = get_updated(newmat[ov_loc:2*ov_loc, ov_loc:2*ov_loc], update[2], rot_loc_frag, rot_frag_loc)
            return newmat
        new_AmB = get_updated_spincomponents(AmB_orig, m1_new, rot_loc_frag, rot_frag_loc)
        new_m0 = get_updated_spincomponents(m0_orig, m0_new, rot_loc_frag, rot_frag_loc)
        new_m0_inv = np.linalg.inv(new_m0)
        new_ApB = np.linalg.multi_dot([new_m0_inv, new_AmB, new_m0_inv])

        new_A = 0.5 * (new_ApB + new_AmB)
        new_B = 0.5 * (new_ApB - new_AmB)

        r_occ, r_vir =  self.get_rot_to_mf()
        # Given that our active orbitals are also canonical this should be diagonal, but calculating the whole
        # thing isn't prohibitive and might save pain.
        loc_eps = einsum("ia,ji,ba,ki,ca->jbkc", epsilon, r_occ, r_vir, r_occ, r_vir).reshape((ov_loc, ov_loc))
        # We want to actually consider the difference from the dRPA kernel. This is just the local eris in an OV basis.
        if eris is None:
            eris = self.base.get_eris(self.c_active)

        v = eris[:nocc_loc, nocc_loc:, :nocc_loc, nocc_loc:].reshape((ov_loc, ov_loc))

        occ_proj = self.get_fragment_projector(self.c_active_occ)
        vir_proj = self.get_fragment_projector(self.c_active_vir)
        def proj_all_indices(mat):
            """Obtains average over all possible projections of provided matrix, giving contribution to democratic
            partitioning from this cluster.
            """
            return (einsum("iajb,ik->kajb",mat, occ_proj) +
                  einsum("iajb,jk->iakb", mat, occ_proj) +
                  einsum("iajb,ac->icjb", mat, vir_proj) +
                  einsum("iajb,bc->iajc", mat, vir_proj)) / 4.0

        # Now calculate all spin components; could double check spin symmetry of ab terms if wanted.
        # This deducts the equivalent values at the level of dRPA, reshapes into fermionic indices, and performs
        # projection to only the fragment portions of all indices.
        newshape = (nocc_loc, nvir_loc, nocc_loc, nvir_loc)
        V_A_aa = proj_all_indices((new_A[:ov_loc, :ov_loc] - loc_eps - v).reshape(newshape))
        V_A_bb = proj_all_indices((new_A[ov_loc : 2 * ov_loc, ov_loc : 2 * ov_loc] - loc_eps - v).reshape(newshape))
        V_A_ab = proj_all_indices((new_A[:ov_loc:, ov_loc: 2 * ov_loc] - v).reshape(newshape))
        V_B_aa = proj_all_indices((new_B[:ov_loc, :ov_loc] - v).reshape(newshape))
        V_B_bb = proj_all_indices((new_B[ov_loc: 2 * ov_loc, ov_loc: 2 * ov_loc] - v).reshape(newshape))
        V_B_ab = proj_all_indices((new_B[:ov_loc:, ov_loc: 2 * ov_loc] - v).reshape(newshape))

        return V_A_aa, V_A_ab, V_A_bb, V_B_aa, V_B_ab, V_B_bb

    def get_correlation_kernel_contrib(self, epsilon, dd0, dd1, eris = None):

        if self.sym_parent is None:
            V_A_aa, V_A_ab, V_A_bb, V_B_aa, V_B_ab, V_B_bb = self.construct_correlation_kernel_contrib(
                                                                                    epsilon, dd0, dd1, eris)
        else:
            V_A_aa, V_A_ab, V_A_bb, V_B_aa, V_B_ab, V_B_bb = self.sym_parent.construct_correlation_kernel_contrib(
                                                                                    epsilon, dd0, dd1, eris)
        # Now need to project back out to full space. This requires an additional factor of the overlap in ou
        # coefficients.
        c_occ = np.dot(self.base.get_ovlp(), self.c_active_occ)
        c_vir = np.dot(self.base.get_ovlp(), self.c_active_vir)
        V_aa = einsum("iajb,pi,qa,rj,sb->pqrs", V_A_aa, c_occ, c_vir, c_occ, c_vir) + \
               einsum("iajb,pi,qa,rj,sb->pqsr", V_B_aa, c_occ, c_vir, c_occ, c_vir)
        V_ab = einsum("iajb,pi,qa,rj,sb->pqrs", V_A_ab, c_occ, c_vir, c_occ, c_vir) + \
               einsum("iajb,pi,qa,rj,sb->pqsr", V_B_ab, c_occ, c_vir, c_occ, c_vir)
        V_bb = einsum("iajb,pi,qa,rj,sb->pqrs", V_A_bb, c_occ, c_vir, c_occ, c_vir) + \
               einsum("iajb,pi,qa,rj,sb->pqsr", V_B_bb, c_occ, c_vir, c_occ, c_vir)
        return V_aa, V_ab, V_bb
