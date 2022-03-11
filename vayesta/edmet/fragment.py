import dataclasses
from timeit import default_timer as timer

import numpy as np
import scipy.linalg

import pyscf.lib
from vayesta.core.util import *
from vayesta.dmet.fragment import DMETFragment
from vayesta.core.bath import BNO_Threshold
from vayesta.solver import get_solver_class2 as get_solver_class


class EDMETFragmentExit(Exception):
    pass


VALID_SOLVERS = ["EBFCI"]  # , "EBFCIQMC"]


class EDMETFragment(DMETFragment):
    @dataclasses.dataclass
    class Options(DMETFragment.Options):
        make_rdm_eb: bool = True
        make_dd_moments: bool = True
        old_sc_condition: bool = NotSet
        max_bos: int = NotSet

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
        return self.cluster.nocc_active * self.cluster.nvir_active

    @property
    def ov_mf(self):
        return self.base.nocc * self.base.nvir

    def get_rot_to_mf_ov(self):
        r_o, r_v = self.get_overlap_m2c()
        spat_rot = einsum("iJ,aB->iaJB", r_o, r_v).reshape((self.ov_mf, self.ov_active)).T
        res = np.zeros((2 * self.ov_active, 2 * self.ov_mf))
        res[:self.ov_active, :self.ov_mf] = spat_rot
        res[self.ov_active:2 * self.ov_active, self.ov_mf:2 * self.ov_mf] = spat_rot
        return res

    def set_up_fermionic_bath(self, bno_threshold=None, bno_number=None):
        """Set up the fermionic bath orbitals"""
        self.make_bath()

        if bno_number is not None:
            bno_threshold = BNO_Threshold('number', bno_number)
        else:
            bno_threshold = BNO_Threshold('occupation', bno_threshold)

        cluster = self.make_cluster(self.bath, bno_threshold=bno_threshold)
        cluster.log_sizes(self.log.info, header="Orbitals for %s" % self)
        self._c_active_occ = cluster.c_active_occ
        self._c_active_vir = cluster.c_active_vir
        # Want to return the rotation of the canonical HF orbitals which produce the cluster canonical orbitals.
        return self.get_rot_to_mf_ov()

    def define_bosons(self, rpa_mom, rot_ov=None, tol=1e-10):
        """Given the RPA zeroth moment between the fermionic cluster excitations and the rest of the space, define
        our cluster bosons.
        Note that this doesn't define our Hamiltonian, since we don't yet have the required portion of our
        zeroth moment for the bosonic degrees of freedom.
        """
        if rot_ov is None:
            rot_ov = self.get_rot_to_mf_ov()
        self.rpa_mom = rpa_mom
        # Need to remove fermionic degrees of freedom from moment contribution. Null space of rotation matrix is size
        # N^4, so instead deduct projection onto fermionic space.
        rot_ov_pinv = np.linalg.pinv(rot_ov.T)
        env_mom = rpa_mom - dot(rpa_mom, rot_ov.T, rot_ov_pinv)
        # v defines the rotation of the mean-field excitation space specifying our bosons.
        u, s, v = np.linalg.svd(env_mom, full_matrices=False)
        want = s > tol
        self.nbos = min(sum(want), self.opts.max_bos)
        if self.nbos < len(s):
            self.log.info("Zeroth moment matching generated %d cluster bosons.Largest discarded singular value: %4.2e.",
                          self.nbos, s[self.nbos:].max())
        else:
            self.log.info("Zeroth moment matching generated %d cluster bosons.", self.nbos)
        # Calculate the relevant components of the zeroth moment- we don't want to recalculate these.
        self.r_bos = v[:self.nbos, :]
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
        eta0 = np.zeros_like(apb)
        eta0[:2 * self.ov_active, :2 * self.ov_active] = self.eta0_ferm
        eta0[:2 * self.ov_active, 2 * self.ov_active:] = self.eta0_coupling
        eta0[2 * self.ov_active:, :2 * self.ov_active] = self.eta0_coupling.T
        eta0[2 * self.ov_active:, 2 * self.ov_active:] = self.eta0_bos

        renorm_amb = dot(eta0, apb, eta0)

        maxdev = abs(amb - renorm_amb)[:2 * self.ov_active, :2 * self.ov_active].max()
        if maxdev > 1e-8:
            self.log.error("Maximum deviation in irreducible polarisation propagator=%6.4e",
                           abs(amb - renorm_amb)[:2 * self.ov_active, :2 * self.ov_active].max())
        a = 0.5 * (apb + renorm_amb)
        b = 0.5 * (apb - renorm_amb)
        couplings_aa = np.zeros((self.nbos, self.cluster.norb_active, self.cluster.norb_active))
        couplings_bb = np.zeros((self.nbos, self.cluster.norb_active, self.cluster.norb_active))

        couplings_aa[:, :self.cluster.nocc_active, self.cluster.nocc_active:] = a[2 * self.ov_active:,
                                                                                :self.ov_active].reshape(
            self.nbos, self.cluster.nocc_active, self.cluster.nvir_active)
        couplings_aa[:, self.cluster.nocc_active:, :self.cluster.nocc_active] = b[2 * self.ov_active:,
                                                                                :self.ov_active].reshape(
            self.nbos, self.cluster.nocc_active, self.cluster.nvir_active).transpose([0, 2, 1])
        couplings_bb[:, :self.cluster.nocc_active, self.cluster.nocc_active:] = \
            a[2 * self.ov_active:, self.ov_active:2 * self.ov_active].reshape(
                self.nbos, self.cluster.nocc_active, self.cluster.nvir_active)
        couplings_bb[:, self.cluster.nocc_active:, :self.cluster.nocc_active] = \
            b[2 * self.ov_active:, self.ov_active:2 * self.ov_active].reshape(
                self.nbos, self.cluster.nocc_active, self.cluster.nvir_active).transpose([0, 2, 1])

        a_bos = a[2 * self.ov_active:, 2 * self.ov_active:]
        b_bos = b[2 * self.ov_active:, 2 * self.ov_active:]
        self.bos_freqs, x, y = bogoliubov_decouple(a_bos + b_bos, a_bos - b_bos)

        couplings_aa = einsum("npq,nm->mpq", couplings_aa, x) + einsum("npq,nm->mqp", couplings_aa, y)
        couplings_bb = np.einsum("npq,nm->mpq", couplings_bb, x) + np.einsum("npq,nm->mqp", couplings_bb, y)
        self.couplings = (couplings_aa, couplings_bb)
        # These are the quantities before decoupling, since these in some sense represent the `physical' excitations of
        # the system.
        self.apb = apb
        self.amb = renorm_amb
        self.eta0 = eta0

    def get_eri_couplings(self, rot):
        """Obtain eri in a space defined by an arbitrary rotation of the mean-field particle-hole excitations of our
        systems. Note that this should only really be used in the case that such a rotation cannot be described by a
        rotation of the underlying single-particle basis, since highly efficient routines already exist for this case..
        """

        # Convert rots from full-space particle-hole excitations into AO pairs.
        def conv_to_aos(r):
            r = r.reshape((-1, self.base.nocc, self.base.nvir))
            return einsum("nia,pi,qa->npq", r, self.base.mo_coeff_occ, self.base.mo_coeff_vir)

        rota, rotb = rot[:, :self.ov_mf], rot[:, self.ov_mf:2 * self.ov_mf]

        if hasattr(self.base.mf, "with_df"):
            rota, rotb = conv_to_aos(rota), conv_to_aos(rotb)
            # Loop through cderis
            res = np.zeros((rot.shape[0], rot.shape[0]))
            for eri1 in self.mf.with_df.loop():
                l_ = einsum("npq,lpq->nl", pyscf.lib.unpack_tril(eri1), rota + rotb)

                res += dot(l_.T, l_)
            return res
        else:
            # This is painful to do for each fragment, but comes from working with 4-index eris.
            eris = self.base.get_eris_array(self.mf.mo_coeff)
            eris = eris[:self.base.nocc, self.base.nocc:, :self.base.nocc, self.base.nocc:].reshape(
                (self.ov_mf, self.ov_mf))
            return dot(rota + rotb, eris, rota.T + rotb.T)

    def get_xc_couplings(self, xc_kernel, rot):
        rota, rotb = rot[:, :self.ov_mf], rot[:, self.ov_mf:2 * self.ov_mf]

        # Convert rots from full-space particle-hole excitations into AO pairs.
        def conv_to_aos(r):
            r = r.reshape((-1, self.base.nocc, self.base.nvir))
            return einsum("nia,pi,qa->npq", r, self.base.mo_coeff_occ, self.base.mo_coeff_vir)

        rota, rotb = conv_to_aos(rota), conv_to_aos(rotb)
        if self.base.with_df:
            # Store low-rank expression for xc kernel.
            # Store alpha and beta-spin xc-kernel contributions separately, so need to treat separately.
            la_l = einsum("npq,lpq->nl", xc_kernel[0][0], rota) + einsum("npq,lpq->nl", xc_kernel[1][0], rotb)
            la_r = einsum("npq,lpq->nl", xc_kernel[0][1], rota) + einsum("npq,lpq->nl", xc_kernel[1][1], rotb)

            lb_l = einsum("npq,lpq->nl", xc_kernel[0][0], rota) + einsum("npq,lpq->nl", xc_kernel[1][0], rotb)
            lb_r = einsum("npq,lqp->nl", xc_kernel[0][1], rota) + einsum("npq,lqp->nl", xc_kernel[1][1], rotb)

            acontrib = dot(la_l.T, la_r)
            bcontrib = dot(lb_l.T, lb_r)
            apb = acontrib + bcontrib
            amb = acontrib - bcontrib
        else:
            # Have full-rank expression for xc kernel, but separate spin channels.
            acontrib = einsum("lpq,pqrs,mrs->lm", rota, xc_kernel[1], rotb)
            acontrib += acontrib.T + einsum("lpq,pqrs,mrs->lm", rota, xc_kernel[0], rota) + einsum("lpq,pqrs,mrs->lm",
                                                                                                   rotb, xc_kernel[2],
                                                                                                   rotb)

            bcontrib = einsum("lpq,pqrs,msr->lm", rota, xc_kernel[1], rotb)
            bcontrib += bcontrib.T + einsum("lpq,pqrs,msr->lm", rota, xc_kernel[0], rota) + einsum("lpq,pqrs,msr->lm",
                                                                                                   rotb, xc_kernel[2],
                                                                                                   rotb)

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
        solver_opts = self.get_solver_options(solver, chempot)

        solver_cls = get_solver_class(self.mf, solver)

        cluster_solver = solver_cls(self, self.cluster, **solver_opts)

        with log_time(self.log.info, ("Time for %s solver:" % solver) + " %s"):
            cluster_solver.kernel(eris=eris)

        results = self._results
        results.bno_threshold = bno_threshold
        results.n_active = self.cluster.norb_active
        # Need to rewrite EBFCI solver to expose this properly...
        results.converged = True

        if self.opts.make_rdm2:
            results.dm1, results.dm2 = cluster_solver.make_rdm12()
        elif self.opts.make_rdm1:
            results.dm1 = cluster_solver.make_rdm1()
        if self.opts.make_rdm_eb:
            results.dm_eb = cluster_solver.make_rdm_eb()
        if self.opts.make_dd_moments:
            ddmoms = cluster_solver.make_dd_moms(1, coeffs=dot(self.cluster.c_active.T, self.base.get_ovlp(),
                                                               self.c_frag))
            if self.opts.old_sc_condition:
                ddmoms[0] = [np.einsum("ppqq->pq", x) for x in ddmoms[0]]
                ddmoms[1] = [np.einsum("ppqq->pq", x) for x in ddmoms[1]]
            results.dd_mom0 = ddmoms[0]
            results.dd_mom1 = ddmoms[1]
        return results

    def get_solver_options(self, solver, chempot):
        solver_opts = {}
        solver_opts.update(self.opts.solver_options)
        pass_through = []
        for attr in pass_through:
            self.log.debugv("Passing fragment option %s to solver.", attr)
            solver_opts[attr] = getattr(self.opts, attr)

        solver_opts["v_ext"] = None if chempot is None else - chempot * self.get_fragment_projector(
            self.cluster.c_active)

        return solver_opts

    def get_edmet_energy_contrib(self):
        """Generate EDMET energy contribution, according to expression given in appendix of EDMET preprint"""
        e1, e2 = self.get_dmet_energy_contrib()
        c_act = self.cluster.c_active
        p_imp = self.get_fragment_projector(c_act)
        dm_eb = self._results.dm_eb
        # Have separate spin contributions.
        efb = 0.5 * (
                np.einsum("pr,npq,rqn", p_imp, self.couplings[0], dm_eb[0]) +
                np.einsum("qr,npq,prn", p_imp, self.couplings[0], dm_eb[0]) +
                np.einsum("pr,npq,rqn", p_imp, self.couplings[1], dm_eb[1]) +
                np.einsum("qr,npq,prn", p_imp, self.couplings[1], dm_eb[1])
        )
        return e1, e2, efb

    def construct_correlation_kernel_contrib(self, epsilon, m0_new, m1_new, eris=None):
        """
        Generate the contribution to the correlation kernel arising from this fragment, in terms of local degrees of
        freedom (ie cluster orbitals and bosons).
        """
        # Get the ApB, AmB and m0 for this cluster. Note that this is pre-boson decoupling, but we don't actually care
        # about that here and it shouldn't change our answer.
        apb_orig = self.apb
        amb_orig = self.amb
        m0_orig = self.eta0

        # m0_new = self.results.dd_mom0
        # m1_new = self.results.dd_mom1

        nocc_loc = self.cluster.nocc_active
        nvir_loc = self.cluster.nvir_active
        ov_loc = nocc_loc * nvir_loc

        # Now want to construct rotations defining which degrees of freedom contribute to two-point quantities.
        occ_frag_rot = np.linalg.multi_dot([self.c_frag.T, self.base.get_ovlp(), self.cluster.c_active_occ])
        vir_frag_rot = np.linalg.multi_dot([self.c_frag.T, self.base.get_ovlp(), self.cluster.c_active_vir])

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
            # proj_from_order = np.linalg.pinv(proj_to_order)
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

        r_occ, r_vir = self.get_overlap_m2c()
        # Given that our active orbitals are also canonical this should be diagonal, but calculating the whole
        # thing isn't prohibitive and might save pain.
        loc_eps = einsum("ia,ij,ab,ik,ac->jbkc", epsilon, r_occ, r_vir, r_occ, r_vir).reshape((ov_loc, ov_loc))
        # We want to actually consider the difference from the dRPA kernel. This is just the local eris in an OV basis.
        if eris is None:
            eris = self.base.get_eris_array(self.cluster.c_active)

        v = eris[:nocc_loc, nocc_loc:, :nocc_loc, nocc_loc:].reshape((ov_loc, ov_loc))

        fr_proj = self.get_fragment_projector(self.cluster.c_active)

        ncl = self.cluster.norb_active
        no = self.cluster.nocc_active

        def map_to_full(mat):
            """Given a matrix in only the occupied-virtual subspace, expand to full hf basis.
            Assumes and applies symmetry v_{pqrs} = v_{qpsr}."""
            fullmat = np.zeros((ncl, ncl, ncl, ncl))
            fullmat[:no, no:, :no, no:] = mat
            fullmat[no:, :no, no:, :no] = mat.transpose([1, 0, 3, 2])
            return fullmat

        def proj_all_indices(mat):
            """Obtains average over all possible projections of provided matrix, giving contribution to democratic
            partitioning from this cluster.
            """
            return (einsum("pqrs,pt->tqrs", mat, fr_proj) +
                    einsum("pqrs,qt->ptrs", mat, fr_proj) +
                    einsum("pqrs,rt->pqts", mat, fr_proj) +
                    einsum("pqrs,st->pqrt", mat, fr_proj)) / 4.0

        # Now calculate all spin components; could double check spin symmetry of ab terms if wanted.
        # This deducts the equivalent values at the level of dRPA, reshapes into fermionic indices, and performs
        # projection to only the fragment portions of all indices.
        newshape = (nocc_loc, nvir_loc, nocc_loc, nvir_loc)
        v_a_aa = map_to_full((new_a[:ov_loc, :ov_loc] - loc_eps - v).reshape(newshape))
        v_a_bb = map_to_full((new_a[ov_loc: 2 * ov_loc, ov_loc: 2 * ov_loc] - loc_eps - v).reshape(newshape))
        v_a_ab = map_to_full((new_a[:ov_loc:, ov_loc: 2 * ov_loc] - v).reshape(newshape))
        v_b_aa = map_to_full((new_b[:ov_loc, :ov_loc] - v).reshape(newshape)).transpose((0, 1, 3, 2))
        v_b_bb = map_to_full((new_b[ov_loc: 2 * ov_loc, ov_loc: 2 * ov_loc] - v).reshape(newshape)).transpose(
            (0, 1, 3, 2))
        v_b_ab = map_to_full((new_b[:ov_loc:, ov_loc: 2 * ov_loc] - v).reshape(newshape)).transpose((0, 1, 3, 2))

        v_aa = proj_all_indices(v_a_aa + v_b_aa)
        v_ab = proj_all_indices(v_a_ab + v_b_ab)
        v_bb = proj_all_indices(v_a_bb + v_b_bb)

        self.save = np.array((v_aa, v_ab, v_bb))

        if self.base.with_df:
            # If using RI we can now perform an svd to generate a low-rank representation in the cluster.
            def construct_low_rank_rep(vaa, vab, vbb):
                """Generates low-rank representation of kernel. Note that this will usually be non-PSD, so a real
                representation will be necessarily asymmetric. Once code is generalised for complex numbers can
                use symmetric decomposition..."""
                na, nb = vaa.shape[0], vbb.shape[0]
                vaa = vaa.reshape((na ** 2, na ** 2))
                vbb = vbb.reshape((nb ** 2, nb ** 2))
                vab = vab.reshape((na ** 2, nb ** 2))

                fullv = np.zeros((na ** 2 + nb ** 2, na ** 2 + nb ** 2))
                fullv[:na ** 2, :na ** 2] = vaa
                fullv[na ** 2:, na ** 2:] = vbb
                fullv[:na ** 2, na ** 2:] = vab
                fullv[na ** 2:, :na ** 2] = vab.T
                u, s, v = np.linalg.svd(fullv, full_matrices=False)
                want = s > 1e-8
                nwant = sum(want)
                self.log.info("Fragment %d gives rank %d xc-kernel contribution.", self.id, nwant)
                repr_l = einsum("n,np->np", s[:nwant] ** (0.5), v[:nwant])
                repr_r = einsum("n,pn->np", s[:nwant] ** (0.5), u[:, :nwant])

                repa = (repr_l[:, :na ** 2].reshape((nwant, na, na)), repr_r[:, :na ** 2].reshape((nwant, na, na)))
                repb = (repr_l[:, na ** 2:].reshape((nwant, nb, nb)), repr_r[:, na ** 2:].reshape((nwant, nb, nb)))
                return repa, repb

            return construct_low_rank_rep(v_aa, v_ab, v_bb)
        else:
            return v_aa, v_ab, v_bb

    def get_correlation_kernel_contrib(self, contrib):
        """Gets contribution to xc kernel in full space of system."""

        c = dot(self.base.get_ovlp(), self.cluster.c_active)

        if self.base.with_df:
            return [tuple([einsum("nij,pi,qj->npq", x, c, c) for x in y]) for y in contrib]
        else:
            v_aa, v_ab, v_bb = contrib
            v_aa = einsum("ijkl,pi,qj,rk,sl->pqrs", v_aa, c, c, c, c)
            v_ab = einsum("ijkl,pi,qj,rk,sl->pqrs", v_ab, c, c, c, c)
            v_bb = einsum("ijkl,pi,qj,rk,sl->pqrs", v_bb, c, c, c, c)
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
