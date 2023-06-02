import dataclasses
from timeit import default_timer as timer

import numpy as np
import pyscf.lib
import scipy.linalg

from vayesta.core.util import dot, einsum, log_time, time_string
from vayesta.dmet.fragment import DMETFragment
from vayesta.solver import check_solver_config
from vayesta.core.bath import helper


from pyscf import __config__

class EDMETFragmentExit(Exception):
    pass


@dataclasses.dataclass
class Options(DMETFragment.Options):
    make_dd_moments: bool = None
    old_sc_condition: bool = None
    max_bos: int = None
    occ_proj_kernel: bool = None
    boson_xc_kernel: bool = None
    bosonic_interaction: str = None


class EDMETFragment(DMETFragment):

    Options = Options

    @dataclasses.dataclass
    class Results(DMETFragment.Results):
        dm_eb: np.ndarray = None
        eb_couplings: np.ndarray = None
        boson_freqs: tuple = None
        dd_mom0: np.ndarray = None
        dd_mom1: np.ndarray = None
        e_fb: float = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prev_xc_contrib = None

    @property
    def ov_active(self):
        return self.cluster.nocc_active * self.cluster.nvir_active

    @property
    def ov_active_tot(self):
        return 2 * self.ov_active

    @property
    def ov_mf(self):
        return self.base.nocc * self.base.nvir

    @property
    def nbos(self):
        if self.sym_parent is not None:
            return self.sym_parent.nbos
        else:
            try:
                return self.r_bos.shape[0]
            except AttributeError:
                raise RuntimeError("Bosons are not yet defined!")

    @property
    def r_bos(self):
        if self.sym_parent is not None:
            raise RuntimeError("Symmetry transformation for EDMET bosons in particle-hole basis is not yet implemented."
                               )
        return self._r_bos

    @r_bos.setter
    def r_bos(self, value):
        if self.sym_parent is not None:
            raise RuntimeError("Cannot set attribute r_bos in symmetry derived fragment.")
        self._r_bos = value

    @property
    def r_bos_ao(self):
        # NB this is the definition of the bosons as a rotation of AO pair excitations.
        if self.sym_parent is None:
            # Need to convert bosonic definition from ov-excitations into ao pairs.
            r_bos = self.r_bos
            co = self.base.mo_coeff_occ
            cv = self.base.mo_coeff_vir
            r_bosa = r_bos[:, :self.ov_mf].reshape((self.nbos, self.base.nocc, self.base.nvir))
            r_bosb = r_bos[:, self.ov_mf:].reshape((self.nbos, self.base.nocc, self.base.nvir))

            return (einsum("nia,pi,qa->npq", r_bosa, co, cv), einsum("nia,pi,qa->npq", r_bosb, co, cv))

        else:
            r_bos_ao = self.sym_parent.r_bos_ao
            # Need to rotate to account for symmetry operations.
            r_bos_ao = tuple([self.sym_op(self.sym_op(x, axis=2), axis=1) for x in r_bos_ao])
        return r_bos_ao

    @property
    def r_ao_bos(self):
        # This is the rotation from the bosons into the AO basis.
        s = self.base.get_ovlp()
        return tuple([einsum("npq,pr,qs->nrs", x, s, s) for x in self.r_bos_ao])

    @property
    def energy_couplings(self):
        try:
            return self._ecouplings
        except AttributeError:
            return self.couplings

    @energy_couplings.setter
    def energy_couplings(self, value):
        self._ecouplings = value

    def check_solver(self, solver):
        is_uhf = np.ndim(self.base.mo_coeff[1]) == 2
        is_eb = True
        check_solver_config(is_uhf, is_eb, solver, self.log)

    def get_fock(self):
        f = self.base.get_fock()
        return np.array((f, f))

    def get_co_active(self):
        co = self.cluster.c_active_occ
        return co, co

    def get_cv_active(self):
        cv = self.cluster.c_active_vir
        return cv, cv

    def get_rot_to_mf_ov(self):
        ro = self.get_overlap('mo[occ]|cluster[occ]')
        rv = self.get_overlap('mo[vir]|cluster[vir]')
        spat_rot = einsum("iJ,aB->iaJB", ro, rv).reshape((self.ov_mf, self.ov_active)).T
        res = np.zeros((2 * self.ov_active, 2 * self.ov_mf))
        res[:self.ov_active, :self.ov_mf] = spat_rot
        res[self.ov_active:2 * self.ov_active, self.ov_mf:2 * self.ov_mf] = spat_rot
        return res

    def get_fragment_projector_ov(self, proj="o", inc_bosons=False):
        """In space of cluster p-h excitations, generate the projector to the impurity portion of the occupied index."""
        if not ("o" in proj or "v" in proj):
            raise ValueError("Must project the occupied and/or virtual index to the fragment. Please specify at least "
                             "one")

        nex = self.ov_active_tot
        if inc_bosons:
            nex += self.nbos

        def get_ov_projector(po, pv):
            p_ov_spat = einsum("ij,ab->iajb", po, pv).reshape((self.ov_active, self.ov_active))
            p_ov = np.zeros((nex, nex))
            p_ov[:self.ov_active, :self.ov_active] = p_ov_spat
            p_ov[self.ov_active:2 * self.ov_active, self.ov_active:2 * self.ov_active] = p_ov_spat
            return p_ov

        p_ov = np.zeros((nex, nex))
        if "o" in proj:
            po = self.get_fragment_projector(self.cluster.c_active_occ)
            pv = np.eye(self.cluster.nvir_active)
            p_ov += get_ov_projector(po, pv)
        if "v" in proj:
            po = np.eye(self.cluster.nocc_active)
            pv = self.get_fragment_projector(self.cluster.c_active_vir)
            p_ov += get_ov_projector(po, pv)
        return p_ov

    def set_up_fermionic_bath(self):
        """Set up the fermionic bath orbitals"""
        self.make_bath()
        cluster = self.make_cluster()
        self._c_active_occ = cluster.c_active_occ
        self._c_active_vir = cluster.c_active_vir
        # Want to return the rotation of the canonical HF orbitals which produce the cluster canonical orbitals.
        return self.get_rot_to_mf_ov()

    def define_bosons(self, rpa_mom, rot_ov=None, tol=1e-8):
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
        nbos = min(sum(want), self.opts.max_bos)
        if nbos < len(s):
            self.log.info("Zeroth moment matching generated %d cluster bosons.Largest discarded singular value: %4.2e.",
                          nbos, s[nbos:].max())
        else:
            self.log.info("Zeroth moment matching generated %d cluster bosons.", nbos)
        self.log.info("Fragment %s Quasiboson histogram", self.id_name)
        self.log.info("------------------------------%s", "-"*len(self.id_name))
        bins = np.hstack([-np.inf, np.logspace(0, -12, 13)[::-1], np.inf])
        self.log.info(helper.make_horizontal_histogram(s, bins=bins))
        # Calculate the relevant components of the zeroth moment- we don't want to recalculate these.
        self.r_bos = v[:nbos, :]
        self.eta0_ferm = np.dot(rpa_mom, rot_ov.T)
        self.eta0_coupling = np.dot(env_mom, self.r_bos.T)
        return self.r_bos

    def construct_boson_hamil(self, eta0_bos, eps, xc_kernel):
        """Given the zeroth moment coupling of our bosons to the remainder of the space, along with stored information,
        generate the components of our interacting electron-boson Hamiltonian.
        At the same time, calculate the local RPA correlation energy since this requires all the same information we
        already have to hand.
        """

        self.store_cluster_rpa(eta0_bos, eps, xc_kernel)

        if "qba" in self.opts.bosonic_interaction.lower():
            bosonic_exchange = "bos_ex" in self.opts.bosonic_interaction.lower()
            self.proj_hamil_qba(exchange_between_bos=bosonic_exchange)
        else:
            if self.opts.bosonic_interaction.lower() == "xc":
                couplings_aa, couplings_bb, a_bos, b_bos = self.save_wxc
            elif self.opts.bosonic_interaction.lower() == "direct":
                couplings_aa, couplings_bb, a_bos, b_bos = self.save_noxc
            else:
                self.log.critical("Unknown bosonic interaction kernel specified.")
                raise RuntimeError

            self.a_bos = a_bos
            if self.nbos >0:
                self.bos_freqs, x, y = bogoliubov_decouple(a_bos + b_bos, a_bos - b_bos)
                couplings_aa = einsum("npq,nm->mpq", couplings_aa, x) + einsum("npq,nm->mqp", couplings_aa, y)
                couplings_bb = np.einsum("npq,nm->mpq", couplings_bb, x) + np.einsum("npq,nm->mqp", couplings_bb, y)
            else:
                self.bos_freqs = np.zeros((0,))
            self.couplings = (couplings_aa, couplings_bb)

        # Will also want to save the effective local modification resulting from our local construction.
        self.log.info("Local correlation energy for fragment %d: %6.4e", self.id, self.loc_erpa)
        return self.loc_erpa

    def store_cluster_rpa(self, eta0_bos, eps, xc_kernel):
        """This function just stores all required information for the """
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
        eta0[:self.ov_active_tot, :self.ov_active_tot] = self.eta0_ferm
        eta0[:self.ov_active_tot, self.ov_active_tot:] = self.eta0_coupling
        eta0[self.ov_active_tot:, :self.ov_active_tot] = self.eta0_coupling.T
        eta0[self.ov_active_tot:, self.ov_active_tot:] = self.eta0_bos

        # Need to generate projector from our RPA excitation space to the local fragment degrees of freedom.
        fproj_ov = self.get_fragment_projector_ov()
        # loc_erpa = (einsum("pq,qr,rp->", fproj_ov, eta0[:self.ov_active_tot, :], apb[:, :self.ov_active_tot]) \
        #                - einsum("pq,qp->", fproj_ov, eps_loc[:self.ov_active_tot, :self.ov_active_tot]) \
        #                - einsum("pq,qp->", fproj_ov, eris[:self.ov_active_tot, :self.ov_active_tot])) / 2.0
        xc_b = (xc_apb - xc_amb) / 2.0
        self.loc_erpa = (einsum("pq,qr,rp->", fproj_ov, eta0[:self.ov_active_tot],
                                (apb - (xc_b / 2.0))[:, :self.ov_active_tot])
                         - einsum("pq,qp->", fproj_ov,
                                  ((apb + amb - xc_b) / 2)[:self.ov_active_tot, :self.ov_active_tot])
                         ) / 2.0

        # loc_erpa = (einsum("pq,qr,rp->", fproj_ov, eta0[:self.ov_active_tot], eris[:, :self.ov_active_tot])
        #            - einsum("pq,qp->", fproj_ov, eris[:self.ov_active_tot, :self.ov_active_tot])) / 4.0

        renorm_amb = dot(eta0, apb, eta0)
        self.amb_renorm_effect = renorm_amb - amb

        maxdev = abs(amb - renorm_amb)[:self.ov_active_tot, :self.ov_active_tot].max()
        if maxdev > 1e-6:
            self.log.error("Maximum deviation in irreducible polarisation propagator=%6.4e",
                           abs(amb - renorm_amb)[:self.ov_active_tot, :self.ov_active_tot].max())

        # If have xc kernel from previous iteration want to deduct contribution from this cluster; otherwise bosons
        # will contain a double-counted representation of the already captured correlation in the cluster.
        self.save_noxc = self._get_boson_hamil(apb - xc_apb, renorm_amb - xc_amb)

        if self.prev_xc_contrib is not None:
            dc_apb, dc_amb = self.get_xc_couplings(self.prev_xc_contrib, np.concatenate([ov_rot, self.r_bos], axis=0))
            apb -= dc_apb
            renorm_amb -= dc_amb

        self.save_wxc = self._get_boson_hamil(apb, renorm_amb)
        # These are the quantities before decoupling, since these in some sense represent the `physical' excitations of
        # the system. ie. each quasi-bosonic excitation operator is made up of only environmental excitations, rather
        # than also including deexcitations, making later manipulations more straightforward.
        self.apb = apb
        self.amb = renorm_amb
        self.eta0 = eta0

    def _get_boson_hamil(self, apb, amb):
        a = 0.5 * (apb + amb)
        b = 0.5 * (apb - amb)

        nactive_a = nactive_b = self.cluster.norb_active

        couplings_aa = np.zeros((self.nbos, nactive_a, nactive_a))
        couplings_bb = np.zeros((self.nbos, nactive_b, nactive_b))

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

        return couplings_aa, couplings_bb, a_bos, b_bos

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

    def conv_to_aos(self, ra, rb):
        # Convert rots from full-space particle-hole excitations into AO pairs.
        def conv_to_aos(r):
            r = r.reshape((-1, self.base.nocc, self.base.nvir))
            return einsum("nia,pi,qa->npq", r, self.base.mo_coeff_occ, self.base.mo_coeff_vir)

        return conv_to_aos(ra), conv_to_aos(rb)

    def get_xc_couplings(self, xc_kernel, rot):

        ov_mf = self.ov_mf
        if isinstance(ov_mf, int): ov_mf = (ov_mf, ov_mf)

        rota, rotb = rot[:, :ov_mf[0]], rot[:, ov_mf[0]:sum(ov_mf)]

        rota, rotb = self.conv_to_aos(rota, rotb)
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

    def proj_hamil_qba(self, exchange_between_bos=True):
        """Generate quasi-bosonic Hamiltonian via projection of appropriate Hamiltonian elements of full system.
        This represents the bosons as an explicit sum of environmental excitations, which we then approximate as bosonic
         degrees of freedom."""

        # Note that electron-boson couplings set here describe a Hamiltonian with terms:
        #       V[n,p,q] p^+ q b_n^+ + h.c.
        # This is arbitrary (and indeed different solvers have different definitions) but this is definitely the one
        # used here for all future reference (after much checking...).
        t0 = timer()
        c = self.cluster.c_active
        if not isinstance(c, tuple):
            ca = cb = c
        else:
            ca, cb = c
        # indexed as npq in atomic orbitals, but p is a projection of the occupied indices and q virtual indices.
        r_bos_aoa, r_bos_aob = self.r_bos_ao
        # Note that our o-v fock matrix blocks may be nonzero, however our environmental states are always constructed
        # from only particle-hole excitations.
        # If no correlation potential was used this can be calculated by eps.
        fa, fb = self.get_fock()

        coa, cob = self.get_co_active()
        cva, cvb = self.get_cv_active()

        noa, nva = coa.shape[1], cva.shape[1]
        nob, nvb = cob.shape[1], cvb.shape[1]

        ovlp = self.base.get_ovlp()
        t_fock_start = timer()
        # Can just use expressions for Hamiltonian elements between single excitations.
        # First, get fock contributions. All are N^3 or less.
        # This will be zero if at HF solution.
        # V_n <= C_{nia}f_{ia}
        bos_nonconserv = einsum("npq,pq->n", r_bos_aoa, fa) + einsum("npq,pq->n", r_bos_aob, fb)
        # \Omega_n <= C_{mia}C_{nib}f_{ab} - C_{mia}C_{nja}f_{ij}
        a_bos = einsum("npq,msr,qr,ps->nm", r_bos_aoa, r_bos_aoa, fa, ovlp) + \
                einsum("npq,msr,qr,ps->nm", r_bos_aob, r_bos_aob, fb, ovlp)
        a_bos -= einsum("npq,mrs,pr,qs->nm", r_bos_aoa, r_bos_aoa, fa, ovlp) + \
                 einsum("npq,mrs,pr,qs->nm", r_bos_aob, r_bos_aob, fb, ovlp)

        # Write this as a single function for both spin channels, to avoid chance of typos
        def get_fock_couplings_spin_channel(r_bos_ao, f, co, cv, no, nv):
            couplings = np.zeros((self.nbos,) + (no + nv,) * 2)
            # No o->v excitation fock contribution.
            # v->o excitation within active space.
            # V_{nai} <= C_{nic}f_{ac} - C_{nka}f_{ik}
            couplings[:, no:, :no] = einsum("npc,qc,pi,qa->nai", r_bos_ao, f, dot(ovlp, co), cv) - \
                                        einsum("nkq,pk,pi,qa->nai", r_bos_ao, f, co, dot(ovlp, cv))
            # o->o excitation within active space. Note that we're constructing the non-normal ordered parameterisation
            # here, so all signs are flipped for o-o component.
            # V_{nij} <= -(\delta_{ij}C_{nkc}f_{ck} - C_{njc}f_{ic})
            fac = einsum("nck,ck->n", r_bos_ao, f)
            couplings[:, :no, :no] = -einsum("pq,n->npq", np.eye(no), fac) + \
                                        einsum("npc,qc,qi,pj->nij", r_bos_ao, f, co, dot(ovlp, co))
            # v->v excitation within active space.
            # V_{nab} <= C_{nic}f_{ac} C_{nka}f_{ik}
            couplings[:, no:, no:] = einsum("pq,n->npq", np.eye(nv), fac) - \
                                        einsum("nkp,kq,pa,qb->nab", r_bos_ao, f, dot(ovlp, cv), cv)
            return couplings

        fcouplings_aa = get_fock_couplings_spin_channel(r_bos_aoa, fa, coa, cva, noa, nva)
        fcouplings_bb = get_fock_couplings_spin_channel(r_bos_aob, fb, cob, cvb, nob, nvb)

        t_fock = timer() - t_fock_start

        # Get coulombic contribution; for coupling this is just V_{npq} <= C_{nkc}<pk||qc>.
        ccouplings_aa = np.zeros_like(fcouplings_aa)
        ccouplings_bb = np.zeros_like(fcouplings_bb)

        t_coulomb = 0
        if exchange_between_bos:
            t_bos_exchange = 0

        if self.base.with_df:
            if exchange_between_bos:
                blk_prefactor = self.nbos * (self.mf.mol.nao ** 2)
            else:
                blk_prefactor = self.mf.mol.nao ** 2
            # Limit ourselves to only use quarter the maximum memory for the single largest array.
            blksize = max(1, int(__config__.MAX_MEMORY / (4 * 8.0 * blk_prefactor)))
            if blksize > self.mf.with_df.get_naoaux():
                blksize = None
            else:
                self.log.info("Using blksize of %d to generate Bosonic Hamiltonian.", blksize)

            for eri1 in self.mf.with_df.loop(blksize):
                # Here we've kept the old einsum expressions around just in case we need comparison later.
                l_ = pyscf.lib.unpack_tril(eri1)
                t_coulomb_start = timer()
                # First generate coulomb interactions effects. This all scales as N^3.
                # l_bos = einsum("npq,mpq->nm", l_, r_bos_aoa + r_bos_aob)  # N^3
                # la_ferm = einsum("npq,pi,qj->nij", l_, ca, ca)  # N^3
                # lb_ferm = einsum("npq,pi,qj->nij", l_, cb, cb)  # N^3

                l_bos = np.tensordot(l_, r_bos_aoa + r_bos_aob, ([1, 2], [1, 2]))
                la_ferm = np.tensordot(np.tensordot(l_, ca, ([1], [0])), ca, ([1], [0]))
                lb_ferm = np.tensordot(np.tensordot(l_, cb, ([1], [0])), cb, ([1], [0]))

                # print("!!1!!",
                #    abs(einsum("npq,mpq->nm", l_, r_bos_aoa + r_bos_aob) - l_bos).max(),
                #    abs(einsum("npq,pi,qj->nij", l_, ca, ca) - la_ferm).max(),
                #    abs(einsum("npq,pi,qj->nij", l_, cb, cb) - lb_ferm).max()
                # )

                # V_{npq} <= (pq|ia)C_{nia} = <pi|qa>C_{nia}
                # ccouplings_aa += einsum("nm,nij->mij", l_bos, la_ferm)  # N^3
                # ccouplings_bb += einsum("nm,nij->mij", l_bos, lb_ferm)  # N^3
                ccouplings_aa += np.tensordot(l_bos, la_ferm, ([0], [0]))
                ccouplings_bb += np.tensordot(l_bos, lb_ferm, ([0], [0]))

                # print("!!2!!",
                #      abs(einsum("nm,nij->mij", l_bos, la_ferm) - np.tensordot(l_bos, la_ferm, ([0], [0]))).max(),
                #      abs( einsum("nm,nij->mij", l_bos, lb_ferm) - np.tensordot(l_bos, lb_ferm, ([0], [0]))).max()
                #      )

                del la_ferm, lb_ferm
                # \Omega_n <= (ia|bj)C_{nia}C_{mjb} = <ib|aj>C_{nia}C_{mjb}
                # a_bos += einsum("nm,no->mo", l_bos, l_bos)  # N
                a_bos += np.tensordot(l_bos, l_bos, ([0], [0]))
                del l_bos
                # Now exchange contributions; those to the coupling are straightforward (N^3) to calculate.
                # la_singl = einsum("npq,pi->niq", l_, ca)  # N^3
                # lb_singl = einsum("npq,pi->niq", l_, cb)  # N^3
                la_singl = np.tensordot(l_, ca, ([1], [0])).transpose((0, 2, 1))  # N^3
                lb_singl = np.tensordot(l_, cb, ([1], [0])).transpose((0, 2, 1))  # N^3
                # print("!!3!!",
                #      abs(einsum("npq,pi->niq", l_, ca) - la_singl).max(),
                #      abs(einsum("npq,pi->niq", l_, cb) - lb_singl).max())
                # V_{npq} <= -(pa|iq)C_{nia} = -<pi|aq>C_{nia}
                # ccouplings_aa -= einsum("nip,njq,mpq->mji", la_singl, la_singl, r_bos_aoa)  # N^3
                # ccouplings_bb -= einsum("nip,njq,mpq->mji", lb_singl, lb_singl, r_bos_aob)  # N^3

                ccouplings_aa -= np.tensordot(
                    la_singl,
                    np.tensordot(la_singl, r_bos_aoa, ([2], [2])),  # njq,mpq->njmp
                    ([0, 2], [0, 3])
                ).transpose([2, 1, 0])  # nip,njmp->ijm->mji

                ccouplings_bb -= np.tensordot(
                    lb_singl,
                    np.tensordot(lb_singl, r_bos_aob, ([2], [2])),  # njq,mpq->njmp
                    ([0, 2], [0, 3])
                ).transpose([2, 1, 0])  # nip,njmp->ijm->mji

                # print("!!4!!",
                #      abs(einsum("nip,njq,mpq->mji", la_singl, la_singl, r_bos_aoa) - np.tensordot(
                #    la_singl,
                #    np.tensordot(la_singl, r_bos_aoa, ([2], [2])),  # njq,mpq->njmp
                #    ([0, 2], [0, 3])
                # ).transpose([2, 1, 0])).max(),
                #      abs(einsum("nip,njq,mpq->mji", lb_singl, lb_singl, r_bos_aob) - np.tensordot(
                #    lb_singl,
                #    np.tensordot(lb_singl, r_bos_aob, ([2], [2])),  # njq,mpq->njmp
                #    ([0, 2], [0, 3])
                # ).transpose([2, 1, 0])).max()
                #      )

                t_coulomb += timer() - t_coulomb_start
                del la_singl, lb_singl
                if exchange_between_bos:
                    t_bosex_start = timer()
                    # boson-boson interactions are N^4, so if have O(N) clusters this would push our scaling to N^5...
                    # Note we want both `occupied` indices of bosonic degrees of freedom to contract to same l, and the
                    # same for both `virtual` indices.
                    # Only same-spin so need to do different channels separately.
                    # \Omega_n <= (ab|ji)C_{nia}C_{mjb} = <aj|bi>C_{nia}C_{mjb}
                    # a_bos -= einsum("nqrm,nrql->ml",
                    #                einsum("npq,mpr->nqrm", l_, r_bos_aoa),
                    #                einsum("npq,lrq->nprl", l_, r_bos_aoa))
                    # a_bos -= einsum("nqrm,nrql->ml",
                    #                einsum("npq,mpr->nqrm", l_, r_bos_aob),
                    #                einsum("npq,lrq->nprl", l_, r_bos_aob))

                    a_bos -= np.tensordot(np.tensordot(l_, r_bos_aoa, ([1], [1])),  # npq,mpr->nqmr
                                          np.tensordot(l_, r_bos_aoa, ([2], [2])),  # npq,mrq->npmr
                                          ([0, 1, 3], [0, 3, 1]))  # nqmr,nrlq->ml
                    a_bos -= np.tensordot(np.tensordot(l_, r_bos_aob, ([1], [1])),  # npq,mpr->nqmr
                                          np.tensordot(l_, r_bos_aob, ([2], [2])),  # npq,mrq->npmr
                                          ([0, 1, 3], [0, 3, 1]))  # nqmr,nrlq->ml

                    t_bos_exchange += timer() - t_bosex_start
        else:
            raise NotImplementedError("Explicit QBA Hamiltonian construction is currently only implemented for use with"
                                      "density fitting.")

        couplings_aa = fcouplings_aa + ccouplings_aa
        couplings_bb = fcouplings_bb + ccouplings_bb

        nelec = self.cluster.nocc_active
        if not isinstance(nelec, int):
            nelec = sum(nelec)
        else:
            nelec *= 2
        shift = -einsum("npp->n", couplings_aa[:, :noa, :noa]) - einsum("npp->n", couplings_bb[:, :nob, :nob])

        bos_nonconserv += shift

        couplings_aa += einsum("n,pq->npq", bos_nonconserv / nelec, np.eye(noa + nva))
        couplings_bb += einsum("n,pq->npq", bos_nonconserv / nelec, np.eye(nob + nvb))

        # Decouple bosons here.
        self.bos_freqs, c = np.linalg.eigh(a_bos)
        self.couplings = (einsum("nm,npq->mqp", c, couplings_aa), einsum("nm,npq->mqp", c, couplings_bb))
        # ccouplings[n,p,q] = <pi||qa>C_{nia}; can use this for energy evaluation later.
        self.energy_couplings = (einsum("nm,npq->mqp", c, ccouplings_aa), einsum("nm,npq->mqp", c, ccouplings_bb))

        self.log.info("Time for Bosonic Hamiltonian Projection into fragment %d:  %s", self.id,
                      time_string(timer() - t0))
        if exchange_between_bos:
            self.log.info("         %s for fock components, %s for N^3 scaling coulombic components and %s for N^4 "
                          "bosonic exchange.", time_string(t_fock), time_string(t_coulomb),
                          time_string(t_bos_exchange))
        else:
            self.log.info("         %s for fock components, and %s for N^3 scaling coulombic components.",
                          time_string(t_fock), time_string(t_coulomb))

    def check_qba_approx(self, rdm1):
        """Given boson and cluster coefficient definitions, checks deviation from exact bosonic commutation relations
        within our cluster projected onto the ground state.
        This will hopefully tell us whether our bosons are likely to be a good approximation to the full system.
        We could take the L2 norm of the overall deviation, but given most of the resultant operators have essentially
        negligible expectation values with the ground state this is an unnecessarily pessimistic
        estimator.
        """

        r_bos_a, r_bos_b = self.get_rbos_split()
        r_o = self.get_overlap('mo[occ]|cluster[occ]')
        r_v = self.get_overlap('mo[vir]|cluster[vir]')
        if not self.base.is_uhf:
            r_o = (r_o, r_o)
            r_v = (r_v, r_v)
            rdm1 = (rdm1 / 2, rdm1 / 2)

        # Contributions to commutator [b_n, b_m^+]
        odev_a = einsum("nia,mja,ik,jl->nmkl", r_bos_a, r_bos_a, r_o[0], r_o[0])
        odev_b = einsum("nia,mja,ik,jl->nmkl", r_bos_b, r_bos_b, r_o[1], r_o[1])

        vdev_a = einsum("nia,mib,ac,bd->nmcd", r_bos_a, r_bos_a, r_v[0], r_v[0])
        vdev_b = einsum("nia,mib,ac,bd->nmcd", r_bos_b, r_bos_b, r_v[1], r_v[1])

        no_a, no_b = r_o[0].shape[1], r_o[1].shape[1]
        dev = einsum("nmij,ij->nm", odev_a, np.eye(no_a) - rdm1[0][:no_a, :no_a]) + \
              einsum("nmij,ij->nm", odev_b, np.eye(no_b) - rdm1[1][:no_b, :no_b]) + \
              einsum("nmab,ab->nm", vdev_a, rdm1[0][no_a:, no_a:]) + \
              einsum("nmab,ab->nm", vdev_b, rdm1[1][no_b:, no_b:])
        self.log.info("Maximum neglected local density fluctuation in quasi-boson commutation=%6.4e", abs(dev.max()))

    def get_rbos_split(self):
        r_bos_a = self.r_bos[:, :self.ov_mf]
        r_bos_b = self.r_bos[:, self.ov_mf:]
        return r_bos_a.reshape((self.nbos, self.base.nocc, self.base.nvir)), r_bos_b.reshape(
            (self.nbos, self.base.nocc, self.base.nvir))

    def kernel(self, solver=None, eris=None, construct_bath=False,
               chempot=None):
        """Solve the fragment with the specified solver and chemical potential."""
        solver = solver or self.solver

        # Create solver object
        t0 = timer()
        cluster_solver = self.get_solver(solver)
        # Chemical potential
        if chempot is not None:
            px =  self.get_fragment_projector(self.cluster.c_active)
            if isinstance(px, tuple):
                cluster_solver.v_ext = (-chempot*px[0], -chempot*px[1])
            else:
                cluster_solver.v_ext = -chempot*px

        with log_time(self.log.info, ("Time for %s solver:" % solver) + " %s"):
            cluster_solver.kernel()

        wf = cluster_solver.wf

        dm1 = wf.make_rdm1()
        dm2 = wf.make_rdm2()
        if self.nbos > 0:
            self.check_qba_approx(dm1)
        dm_eb = wf.make_rdmeb()
        self._results = results = self.Results(fid=self.id, n_active=self.cluster.norb_active,
                converged=True, wf=wf, dm1=dm1, dm2=dm2, dm_eb=dm_eb)
        results.e1, results.e2, results.e_fb = self.get_edmet_energy_contrib()

        if self.opts.make_dd_moments:
            r_o = self.get_overlap('cluster[occ]|frag')
            r_v = self.get_overlap('cluster[vir]|frag')
            if isinstance(r_o, tuple):
                r = tuple([np.concatenate([x, y], axis=0) for x, y in zip(r_o, r_v)])
            else:
                r = np.concatenate([r_o, r_v], axis=0)

            ddmoms = wf.make_dd_moms(1, coeffs=r)
            if self.opts.old_sc_condition:
                ddmoms[0] = [np.einsum("ppqq->pq", x) for x in ddmoms[0]]
                ddmoms[1] = [np.einsum("ppqq->pq", x) for x in ddmoms[1]]
            results.dd_mom0 = ddmoms[0]
            results.dd_mom1 = ddmoms[1]

        return results

    def get_solver_options(self, solver):
        solver_opts = {}
        solver_opts.update(self.opts.solver_options)
        pass_through = []
        for attr in pass_through:
            self.log.debugv("Passing fragment option %s to solver.", attr)
            solver_opts[attr] = getattr(self.opts, attr)

        return solver_opts

    def get_edmet_energy_contrib(self, hamil=None):
        """Generate EDMET energy contribution, according to expression given in appendix of EDMET preprint"""
        e1, e2 = self.get_dmet_energy_contrib(hamil)
        c_act = self.cluster.c_active
        p_imp = self.get_fragment_projector(c_act)
        if not isinstance(p_imp, tuple):
            p_imp = (p_imp, p_imp)
        dm_eb = self._results.dm_eb
        couplings = self.energy_couplings

        # Have separate spin contributions.
        if "qba" in self.opts.bosonic_interaction:
            # Already have exchange effects included in interactions, so can use straightforward contraction.
            # dm_eb -> <0|b^+ p^+ q|0> in P[p,q,b].
            # couplings -> <pi||qa>C_{nia} in couplings[n,p,q].
            # Want <pj||qb>C_{njb} ( <b_n^+ q^+ p> - <b_n q^+ p>) so our energy is:
            efb = 0.25 * (einsum("qr,npq,rpn", p_imp[0], couplings[0], dm_eb[0] - dm_eb[0].transpose(1, 0, 2)) +
                          einsum("qr,npq,rpn", p_imp[1], couplings[1], dm_eb[1] - dm_eb[1].transpose(1, 0, 2))
                          )
            self.delta = efb
        else:
            efb = 0.5 * (
                    np.einsum("pr,npq,rqn", p_imp[0], couplings[0], dm_eb[0]) +
                    np.einsum("qr,npq,prn", p_imp[0], couplings[0], dm_eb[0]) +
                    np.einsum("pr,npq,rqn", p_imp[1], couplings[1], dm_eb[1]) +
                    np.einsum("qr,npq,prn", p_imp[1], couplings[1], dm_eb[1])
            )
        return e1, e2, efb

    # From this point on have functionality to perform self-consistency.

    def construct_correlation_kernel_contrib(self, epsilon, m0_new, m1_new, eris=None, svdtol=1e-12):
        """
        Generate the contribution to the correlation kernel arising from this fragment, in terms of local degrees of
        freedom (ie cluster orbitals and bosons).
        """

        new_amb, new_apb = self.get_composite_moments(m0_new, m1_new)

        r_occ = self.get_overlap('mo[occ]|cluster[occ]')
        r_vir = self.get_overlap('mo[vir]|cluster[vir]')
        if not isinstance(r_occ, tuple): r_occ = (r_occ, r_occ)
        if not isinstance(r_vir, tuple): r_vir = (r_vir, r_vir)
        ov_a, ov_b = self.ov_active_ab
        no_a, no_b = self.nocc_ab
        nv_a, nv_b = self.nvir_ab
        ncl_a, ncl_b = self.nclus_ab
        # We want to actually consider the difference from the dRPA kernel.
        # Get irreducible polarisation propagator.
        ov_rot = self.get_rot_to_mf_ov()
        eps_loc = self.get_loc_eps(epsilon, np.concatenate([ov_rot, self.r_bos], axis=0))
        # Get eri couplings between all fermionic and boson degrees of freedom.
        eris = self.get_eri_couplings(np.concatenate([ov_rot, self.r_bos], axis=0))
        # Calculate just xc contribution.
        # For A+B need to deduct dRPA contribution.
        new_xc_apb = new_apb - eps_loc - 2 * eris
        # For A-B need to deduct effective dRPA contribution and the renormalisation of interactions introduced in
        # cluster construction.
        new_xc_amb = new_amb - eps_loc - self.amb_renorm_effect

        new_xc_a = 0.5 * (new_xc_apb + new_xc_amb)
        new_xc_b = 0.5 * (new_xc_apb - new_xc_amb)
        # Now just need to convert to ensure proper symmetries of couplings are imposed, and project each index in turn
        # into the fragment space.
        if self.opts.occ_proj_kernel:
            fr_proj = self.get_fragment_projector_ov(proj="o", inc_bosons=True)
            # Need to divide my the number of projectors actually applied; here it's just two.
            fac = 2.0
        else:
            fr_proj = self.get_fragment_projector_ov(proj="ov", inc_bosons=True)
            # Have sum of occupied and virtual projectors, so four total.
            fac = 4.0

        new_xc_a = (dot(fr_proj, new_xc_a) + dot(new_xc_a, fr_proj)) / fac
        new_xc_b = (dot(fr_proj, new_xc_b) + dot(new_xc_b, fr_proj)) / fac

        # Now need to combine A and B contributions, taking care of bosonic contributions.
        # Note that we currently won't have any boson-boson contributions, and all contributions are
        # symmetric.

        def get_fermionic_spat_contrib(acon, bcon, no_l, nv_l, no_r, nv_r):
            f_shape = (no_l, nv_l, no_r, nv_r)
            fermionic = np.zeros((no_l + nv_l,) * 2 + (no_r + nv_r,) * 2)
            fermionic[:no_l, no_l:, :no_r, no_r:] = acon.reshape(f_shape)
            fermionic[:no_l, no_l:, no_r:, :no_r] = bcon.reshape(f_shape).transpose((0, 1, 3, 2))
            fermionic = fermionic + fermionic.transpose((1, 0, 3, 2))
            return fermionic
        ferm_aa = get_fermionic_spat_contrib(new_xc_a[:ov_a, :ov_a], new_xc_b[:ov_a, :ov_a], no_a, nv_a, no_a, nv_a)
        ferm_ab = get_fermionic_spat_contrib(new_xc_a[:ov_a, ov_a:ov_a + ov_b], new_xc_b[:ov_a, ov_a:ov_a+ov_b],
                                             no_a, nv_a, no_b, nv_b)
        ferm_bb = get_fermionic_spat_contrib(new_xc_a[ov_a:ov_a+ov_b, ov_a:ov_a+ov_b],
                                             new_xc_b[ov_a:ov_a+ov_b, ov_a:ov_a+ov_b], no_b, nv_b, no_b, nv_b)

        def get_fb_spat_contrib(acon, bcon, no, nv):
            fb_shape = (no, nv, self.nbos)
            fermbos = np.zeros((no + nv,) * 2 + (self.nbos,))
            fermbos[:no, no:, :] = acon.reshape(fb_shape)
            fermbos[no:, :no, :] = bcon.reshape(fb_shape).transpose((1, 0, 2))
            return fermbos

        if self.opts.boson_xc_kernel:
            fb_a = get_fb_spat_contrib(new_xc_a[:ov_a, ov_a+ov_b:], new_xc_b[:ov_a, ov_a+ov_b:], no_a, nv_a)
            fb_b = get_fb_spat_contrib(new_xc_a[ov_a:ov_a+ov_b, ov_a+ov_b:], new_xc_b[ov_a:ov_a+ov_b, ov_a+ov_b:],
                                       no_b, nv_b)
        else:
            fb_a = np.zeros((no_a + nv_a,) * 2 + (0,))
            fb_b = np.zeros((no_b + nv_b,) * 2 + (0,))

        if self.base.with_df:
            # If using RI we can now perform an svd to generate a low-rank representation in the cluster.
            def construct_low_rank_rep(vaa, vab, vbb, v_fb_a, v_fb_b):
                """Generates low-rank representation of kernel. Note that this will usually be non-PSD, so a real
                representation will be necessarily asymmetric. Once code is generalised for complex numbers can
                use symmetric decomposition..."""
                na, nb = vaa.shape[0], vbb.shape[0]
                nbos = v_fb_a.shape[2]
                nferm_tot = na ** 2 + nb ** 2

                vaa = vaa.reshape((na ** 2, na ** 2))
                vbb = vbb.reshape((nb ** 2, nb ** 2))
                vab = vab.reshape((na ** 2, nb ** 2))

                v_fb_a_ex = v_fb_a.reshape((na ** 2, nbos))
                v_fb_b_ex = v_fb_b.reshape((nb ** 2, nbos))

                v_fb_a_dex = v_fb_a.transpose((1, 0, 2)).reshape((na ** 2, nbos))
                v_fb_b_dex = v_fb_b.transpose((1, 0, 2)).reshape((nb ** 2, nbos))

                fullv = np.zeros((na ** 2 + nb ** 2 + 2 * nbos,) * 2)
                fullv[:na ** 2, :na ** 2] = vaa
                fullv[na ** 2:nferm_tot, na ** 2:nferm_tot] = vbb
                fullv[:na ** 2, na ** 2:nferm_tot] = vab
                fullv[na ** 2:nferm_tot, :na ** 2] = vab.T

                # Component coupling to bosonic excitations.
                fullv[:na ** 2, nferm_tot:nferm_tot + nbos] = v_fb_a_ex
                fullv[na ** 2:nferm_tot, nferm_tot:nferm_tot + nbos] = v_fb_b_ex

                fullv[nferm_tot:nferm_tot + nbos, :na ** 2] = v_fb_a_ex.T
                fullv[nferm_tot:nferm_tot + nbos, na ** 2:nferm_tot] = v_fb_b_ex.T

                # Component coupling to bosonic excitations.
                fullv[:na ** 2, nferm_tot + nbos:] = v_fb_a_dex
                fullv[na ** 2:nferm_tot, nferm_tot + nbos:] = v_fb_b_dex

                fullv[nferm_tot + nbos:, :na ** 2] = v_fb_a_dex.T
                fullv[nferm_tot + nbos:, na ** 2:nferm_tot] = v_fb_b_dex.T

                u, s, v = np.linalg.svd(fullv, full_matrices=False)
                want = s > svdtol
                nwant = sum(want)
                self.log.info("Fragment %d gives rank %d xc-kernel contribution.", self.id, nwant)
                repr_l = einsum("n,np->np", s[:nwant] ** (0.5), v[:nwant])
                repr_r = einsum("n,pn->np", s[:nwant] ** (0.5), u[:, :nwant])

                repf_a = (repr_l[:, :na ** 2].reshape((nwant, na, na)),
                          repr_r[:, :na ** 2].reshape((nwant, na, na)))
                repf_b = (repr_l[:, na ** 2:nferm_tot].reshape((nwant, nb, nb)),
                          repr_r[:, na ** 2:nferm_tot].reshape((nwant, nb, nb)))
                repbos_ex = (repr_l[:, nferm_tot:nferm_tot + nbos], repr_r[:, nferm_tot:nferm_tot + nbos])
                repbos_dex = (repr_l[:, nferm_tot + nbos:], repr_r[:, nferm_tot + nbos:])
                return repf_a, repf_b, repbos_ex, repbos_dex

            return construct_low_rank_rep(ferm_aa, ferm_ab, ferm_bb, fb_a, fb_b)
        else:
            return ferm_aa, ferm_ab, ferm_bb, fb_a, fb_b

    def get_correlation_kernel_contrib(self, contrib):
        """Gets contribution to xc kernel in full space of system."""

        c = dot(self.base.get_ovlp(), self.cluster.c_active)

        if self.base.with_df:
            # First get the contribution from the fermionic degrees of freedom.
            res = [tuple([einsum("nij,pi,qj->npq", x, c, c) for x in y]) for y in contrib[:2]]
            if self.opts.boson_xc_kernel:
                repbos_ex, repbos_dex = contrib[2:]
                r_ao_bosa, r_ao_bosb = self.r_ao_bos

                bos_contrib = [
                    (einsum("nz,zpq->npq", repbos_ex[0], r_ao_bosa) + einsum("nz,zpq->nqp", repbos_dex[0], r_ao_bosa),
                     einsum("nz,zpq->npq", repbos_ex[1], r_ao_bosa) + einsum("nz,zpq->nqp", repbos_dex[1], r_ao_bosa)),
                    (einsum("nz,zpq->npq", repbos_ex[0], r_ao_bosb) + einsum("nz,zpq->nqp", repbos_dex[0], r_ao_bosb),
                     einsum("nz,zpq->npq", repbos_ex[1], r_ao_bosb) + einsum("nz,zpq->nqp", repbos_dex[1], r_ao_bosb))]
                res = [tuple([z1 + z2 for z1, z2 in zip(x, y)]) for x, y in zip(res, bos_contrib)]
            self.prev_xc_contrib = res
            return res
        else:
            v_aa, v_ab, v_bb, fb_a, fb_b = contrib
            v_aa = einsum("ijkl,pi,qj,rk,sl->pqrs", v_aa, c, c, c, c)
            v_ab = einsum("ijkl,pi,qj,rk,sl->pqrs", v_ab, c, c, c, c)
            v_bb = einsum("ijkl,pi,qj,rk,sl->pqrs", v_bb, c, c, c, c)

            if self.opts.boson_xc_kernel:
                r_bosa, r_bosb = self.r_bos_ao
                # First bosonic excitations, need to consider boson for both first and second index pair.
                bos_v_aa = einsum("ijn,pi,qj,nrs->pqrs", fb_a, c, c, r_bosa)
                bos_v_aa += einsum("pqrs->rspq", bos_v_aa)
                bos_v_bb = einsum("ijn,pi,qj,nrs->pqrs", fb_b, c, c, r_bosb)
                bos_v_bb += einsum("pqrs->rspq", bos_v_bb)
                bos_v_ab = einsum("ijn,pi,qj,nrs->pqrs", fb_a, c, c, r_bosb)
                bos_v_ab += einsum("ijn,pi,qj,nrs->rspq", fb_b, c, c, r_bosa)
                # Bosonic dexcitations contributions swap pqrs->qpsr.
                bos_v_aa += einsum("pqrs->qpsr", bos_v_aa)
                bos_v_ab += einsum("pqrs->qpsr", bos_v_ab)
                bos_v_bb += einsum("pqrs->qpsr", bos_v_bb)

                v_aa += bos_v_aa
                v_ab += bos_v_ab
                v_bb += bos_v_bb

            self.prev_xc_contrib = (v_aa, v_ab, v_bb)

            return v_aa, v_ab, v_bb

    def get_composite_moments(self, m0_new, m1_new):
        """Construct composite moments using the local solver dd moments and the lattice RPA moments"""
        # Get the ApB, AmB and m0 for this cluster. Note that this is pre-boson decoupling, but we don't actually care
        # about that here and it shouldn't change our answer.
        apb_orig = self.apb
        amb_orig = self.amb
        m0_orig = self.eta0
        # Now want to construct rotations defining which degrees of freedom contribute to two-point quantities.
        rot_ov_frag, rot_frag_ov, proj_to_order = self.get_rot_ov_frag()
        ov_a, ov_b = self.ov_active_ab
        # Now generate new moments in whatever space our self-consistency condition requires.
        m0_new = [dot(proj_to_order.T, x.reshape((proj_to_order.shape[0],) * 2), proj_to_order) for x in m0_new]
        m1_new = [dot(proj_to_order.T, x.reshape((proj_to_order.shape[0],) * 2), proj_to_order) for x in m1_new]

        def get_updated(orig, update, rot_ovf, rot_fov):
            """Given the original value of a block, the updated solver value, and rotations between appropriate spaces
            generate the updated value of the appropriate block."""
            if not isinstance(rot_ovf, tuple): rot_ovf = (rot_ovf, rot_ovf)
            if not isinstance(rot_fov, tuple): rot_fov = (rot_fov, rot_fov)
            # Generate difference in local, two-point excitation basis.
            diff = update - np.linalg.multi_dot([rot_ovf[0], orig, rot_ovf[1].T])
            return orig + np.linalg.multi_dot([rot_fov[0], diff, rot_fov[1].T])

        def get_updated_spincomponents(orig, update, rot_ov_frag, rot_frag_ov):
            newmat = orig.copy()

            newmat[:ov_a, :ov_a] = get_updated(newmat[:ov_a, :ov_a], update[0], rot_ov_frag[0], rot_frag_ov[0])
            newmat[:ov_a, ov_a:ov_a + ov_b] = get_updated(newmat[:ov_a, ov_a:ov_a + ov_b], update[1], rot_ov_frag,
                                                          rot_frag_ov)
            newmat[ov_a:ov_a + ov_b, :ov_a] = newmat[:ov_a, ov_a:ov_a + ov_b].T
            newmat[ov_a:ov_a + ov_b, ov_a:ov_a + ov_b] = get_updated(newmat[ov_a:ov_a + ov_b, ov_a:ov_a + ov_b],
                                                                     update[2],
                                                                     rot_ov_frag[1], rot_frag_ov[1])
            return newmat

        new_amb = get_updated_spincomponents(amb_orig, m1_new, rot_ov_frag, rot_frag_ov)
        new_m0 = get_updated_spincomponents(m0_orig, m0_new, rot_ov_frag, rot_frag_ov)
        new_m0_inv = np.linalg.inv(new_m0)
        new_apb = np.linalg.multi_dot([new_m0_inv, new_amb, new_m0_inv])

        return new_amb, new_apb

    def get_rot_ov_frag(self):
        """Get rotations between the relevant space for fragment two-point excitations and the cluster active occupied-
        virtual excitations."""

        occ_frag_rot = self.get_overlap('cluster[occ]|frag')
        vir_frag_rot = self.get_overlap('cluster[vir]|frag')
        ov_loc = self.ov_active
        if self.opts.old_sc_condition:
            # Then get projectors to local quantities in ov-basis. Note this needs to be stacked to apply to each spin
            # pairing separately.
            rot_ov_frag = np.einsum("ip,ap->pia", occ_frag_rot, vir_frag_rot).reshape((-1, ov_loc))
            # Get pseudo-inverse to map from frag to loc. Since occupied-virtual excitations aren't spanning this
            # isn't a simple transpose.
            rot_frag_ov = np.linalg.pinv(rot_ov_frag)
            proj_to_order = np.eye(rot_ov_frag.shape[0])
        else:
            # First, grab rotations from particle-hole excitations to fragment degrees of freedom, ignoring reordering
            rot_ov_frag = np.einsum("ip,aq->pqia", occ_frag_rot, vir_frag_rot).reshape((-1, ov_loc))
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
        # Return tuples so can can unified interface with UHF implementation.
        return (rot_ov_frag, rot_ov_frag), (rot_frag_ov, rot_frag_ov), proj_to_order

    def calc_exact_ac(self, eps, use_plasmon=True, deg=5):
        """Evaluate the exact local energy for RPA in this cluster via the Adiabatic Connection, with or without
        the plasmon formula. Note that although

        """
        ov_rot = self.get_rot_to_mf_ov()
        # Get couplings between all fermionic and boson degrees of freedom.
        eris = self.get_eri_couplings(np.concatenate([ov_rot, self.r_bos], axis=0))

        eps_loc = self.get_loc_eps(eps, np.concatenate([ov_rot, self.r_bos], axis=0))

        fproj_ov = self.get_fragment_projector_ov()

        xc_apb = self.apb - eps_loc - 2 * eris
        # NB for the sake of our local energy evaluation the renormalisation is just a component of the coulomb
        # interaction.
        xc_amb = self.amb - eps_loc - self.amb_renorm_effect

        def calc_eta0(alpha):
            amb_alpha = eps_loc + (self.amb - eps_loc) * alpha
            apb_alpha = eps_loc + (self.apb - eps_loc) * alpha
            # e, c = np.linalg.eig(dot(amb, apb))
            MPrt = scipy.linalg.sqrtm(dot(amb_alpha, apb_alpha))  # einsum("pn,n,qn->pq", c, e ** (0.5), c)
            eta0 = dot(MPrt, np.linalg.solve(apb_alpha, np.eye(apb_alpha.shape[0])))
            return eta0

        def calc_contrib_partialint(alpha):
            eta0 = calc_eta0(alpha)
            eta0inv = np.linalg.inv(eta0)

            return -(einsum("pq,qr,rp->", fproj_ov, eta0[:self.ov_active_tot], xc_apb[:, :self.ov_active_tot]) +
                     einsum("pq,qr,rp->", fproj_ov, eta0inv[:self.ov_active_tot],
                            xc_amb[:, :self.ov_active_tot])) / 4

        def calc_contrib_direct(alpha):
            eta0 = calc_eta0(alpha)
            eta0inv = np.linalg.inv(eta0)
            # This is just the contribution from the bare, standard coulomb interaction.
            e_bare = einsum("pq,pr,rq->", fproj_ov,
                            (eta0 - np.eye(self.ov_active_tot + self.nbos))[:self.ov_active_tot],
                            eris[:, :self.ov_active_tot]) / 2
            # Need to account for renormalisation of bosonic interactions, which is included in cluster coulomb kernel.
            renorm = self.amb_renorm_effect / 2
            e_renorm = einsum("pq,pr,rq->", fproj_ov,
                              (eta0inv - np.eye(self.ov_active_tot + self.nbos))[:self.ov_active_tot],
                              renorm[:, :self.ov_active_tot]) / 2

            return e_bare + e_renorm

        def run_ac_inter(func, deg=5):
            points, weights = np.polynomial.legendre.leggauss(deg)
            # Shift and reweight to interval of [0,1].
            points += 1
            points /= 2
            weights /= 2
            return sum([w * func(p) for w, p in zip(weights, points)])

        if use_plasmon:
            e_plasmon = (einsum("pq,qr,rp->", fproj_ov, self.eta0[:self.ov_active_tot],
                                self.apb[:, :self.ov_active_tot]) - (
                             einsum("pq,qp->", fproj_ov,
                                    (eps_loc + eris + self.amb_renorm_effect / 2)[:self.ov_active_tot,
                                    :self.ov_active_tot]))) / 2

            return e_plasmon + run_ac_inter(calc_contrib_partialint, deg)

        else:
            return run_ac_inter(calc_contrib_direct, deg)

    def test_total_rpa_energy(self, eps, use_plasmon=True, deg=5):
        """Evaluate the exact local energy for RPA in this cluster via the Adiabatic Connection, with or without
        the plasmon formula. Note that although

        """
        ov_rot = self.get_rot_to_mf_ov()
        # Get couplings between all fermionic and boson degrees of freedom.
        eris = self.get_eri_couplings(np.concatenate([ov_rot, self.r_bos], axis=0))

        eps_loc = self.get_loc_eps(eps, np.concatenate([ov_rot, self.r_bos], axis=0))
        xc_apb = self.apb - eps_loc - 2 * eris
        # NB for the sake of our local energy evaluation the renormalisation is just a component of the coulomb
        # interaction.
        xc_amb = self.amb - eps_loc - self.amb_renorm_effect

        def calc_eta0(alpha):
            amb_alpha = eps_loc + (self.amb - eps_loc) * alpha
            apb_alpha = eps_loc + (self.apb - eps_loc) * alpha
            # e, c = np.linalg.eig(dot(amb, apb))
            MPrt = scipy.linalg.sqrtm(dot(amb_alpha, apb_alpha))  # einsum("pn,n,qn->pq", c, e ** (0.5), c)
            eta0 = dot(MPrt, np.linalg.solve(apb_alpha, np.eye(apb_alpha.shape[0])))
            return eta0

        def calc_contrib_partialint(alpha):
            eta0 = calc_eta0(alpha)
            eta0inv = np.linalg.inv(eta0)

            return -(einsum("pq,qp->", eta0, xc_apb) +
                     einsum("pq,qp->", eta0inv, xc_amb)) / 4

        def calc_contrib_direct(alpha):
            eta0 = calc_eta0(alpha)
            eta0inv = np.linalg.inv(eta0)
            # This is just the contribution from the bare, standard coulomb interaction.
            e_bare = einsum("pq,qp->", (eta0 - np.eye(self.ov_active_tot + self.nbos)), eris) / 2
            # Need to account for renormalisation of bosonic interactions, which is included in cluster coulomb kernel.
            renorm = self.amb_renorm_effect / 2
            e_renorm = einsum("pq,qp->", (eta0inv - np.eye(self.ov_active_tot + self.nbos)), renorm) / 2

            return e_bare + e_renorm

        def run_ac_inter(func, deg=5):
            points, weights = np.polynomial.legendre.leggauss(deg)
            # Shift and reweight to interval of [0,1].
            points += 1
            points /= 2
            weights /= 2
            return sum([w * func(p) for w, p in zip(weights, points)])

        e_plasmon = (einsum("pq,qp->", self.eta0, self.apb) -
                     ((eps_loc + eris + self.amb_renorm_effect / 2).trace())) / 2

        e_plasmon = e_plasmon + run_ac_inter(calc_contrib_partialint, deg)

        e_direct = run_ac_inter(calc_contrib_direct, deg)

        self.log.info("Difference between plasmon and direct AC total correlation energies: %6.4e",
                      e_plasmon - e_direct)

        return e_plasmon, e_direct

    @property
    def ov_active_ab(self):
        ov = self.ov_active
        if isinstance(ov, int):
            return (ov, ov)
        else:
            return ov

    @property
    def nocc_ab(self):
        no = self.cluster.nocc_active
        if isinstance(no, int):
            return (no, no)
        else:
            return no

    @property
    def nclus_ab(self):
        ncl = self.cluster.norb_active
        if isinstance(ncl, int):
            return (ncl, ncl)
        else:
            return ncl

    @property
    def nvir_ab(self):
        return tuple([x - y for x, y in zip(self.nclus_ab, self.nocc_ab)])

    def split_ov_spin_components(self, mat):
        ov = self.ov_active
        if isinstance(ov, tuple):
            ova, ovb = ov
        else:
            ova = ovb = ov

        return mat[:ova, :ova], mat[:ova, ova:ova + ovb], mat[ova:ova + ovb, ova:ova + ovb]


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
