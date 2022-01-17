import numpy as np

from vayesta.core.util import *
from vayesta.core.qemb import UFragment
from vayesta.dmet.ufragment import UDMETFragment
from .fragment import EDMETFragment

import pyscf.lib


class UEDMETFragment(UDMETFragment, EDMETFragment):

    @property
    def ov_active(self):
        no_a, no_b = self.cluster.nocc_active
        nv_a, nv_b = self.cluster.nvir_active
        return no_a * nv_a, no_b * nv_b

    @property
    def ov_active_tot(self):
        return sum(self.ov_active)

    @property
    def ov_mf(self):
        no_a, no_b = self.base.nocc
        nv_a, nv_b = self.base.nvir
        return no_a * nv_a, no_b * nv_b

    def get_rot_to_mf_ov(self):
        r_o, r_v = self.get_overlap_m2c()
        spat_rota = einsum("iJ,aB->iaJB", r_o[0], r_v[0]).reshape((self.ov_mf[0], self.ov_active[0])).T
        spat_rotb = einsum("iJ,aB->iaJB", r_o[1], r_v[1]).reshape((self.ov_mf[1], self.ov_active[1])).T
        res = np.zeros((sum(self.ov_active), sum(self.ov_mf)))
        res[:self.ov_active[0], :self.ov_mf[0]] = spat_rota
        res[self.ov_active[0]:, self.ov_mf[0]:] = spat_rotb
        return res

    def get_fragment_projector_ov(self):
        """In space of cluster p-h excitations, generate the projector to the impurity portion of the occupied index."""
        poa, pob = self.get_fragment_projector(self.cluster.c_active_occ)
        pva, pvb = [np.eye(x) for x in self.cluster.nvir_active]
        p_ova = einsum("ij,ab->iajb", poa, pva).reshape((self.ov_active[0], self.ov_active[0]))
        p_ovb = einsum("ij,ab->iajb", pob, pvb).reshape((self.ov_active[1], self.ov_active[1]))
        p_ov = np.zeros((self.ov_active_tot, self.ov_active_tot))
        p_ov[:self.ov_active[0], :self.ov_active[0]] = p_ova
        p_ov[self.ov_active[0]:, self.ov_active[0]:] = p_ovb
        return p_ov

    def _get_boson_hamil(self, apb, amb):
        a = 0.5 * (apb + amb)
        b = 0.5 * (apb - amb)

        n_a, n_b = self.cluster.norb_active
        nocc_a, nocc_b = self.cluster.nocc_active
        nvir_a, nvir_b = self.cluster.nvir_active
        ov_a, ov_b = self.ov_active

        couplings_aa = np.zeros((self.nbos, n_a, n_a))
        couplings_bb = np.zeros((self.nbos, n_b, n_b))

        couplings_aa[:, :nocc_a, nocc_a:] = a[ov_a + ov_b:, :ov_a].reshape(self.nbos, nocc_a, nvir_a)
        couplings_aa[:, nocc_a:, :nocc_a] = b[ov_a + ov_b:, :ov_a].reshape(self.nbos, nocc_a, nvir_a).transpose(
            [0, 2, 1])

        couplings_bb[:, :nocc_b, nocc_b:] = a[ov_a + ov_b:, ov_a:ov_a + ov_b].reshape(self.nbos, nocc_b, nvir_b)
        couplings_bb[:, nocc_b:, :nocc_b] = b[ov_a + ov_b:, ov_a:ov_a + ov_b].reshape(self.nbos, nocc_b,
                                                                                      nvir_b).transpose(
            [0, 2, 1])

        a_bos = a[ov_a + ov_b:, ov_a + ov_b:]
        b_bos = b[ov_a + ov_b:, ov_a + ov_b:]

        return couplings_aa, couplings_bb, a_bos, b_bos

    def conv_to_aos(self, ra, rb):
        ra = ra.reshape((-1, self.base.nocc[0], self.base.nvir[0]))
        rb = rb.reshape((-1, self.base.nocc[1], self.base.nvir[1]))
        return einsum("nia,pi,qa->npq", ra, self.base.mo_coeff_occ[0], self.base.mo_coeff_vir[0]), \
               einsum("nia,pi,qa->npq", rb, self.base.mo_coeff_occ[1], self.base.mo_coeff_vir[1])

    def get_eri_couplings(self, rot):
        """Obtain eri in a space defined by an arbitrary rotation of the mean-field particle-hole excitations of our
        systems. Note that this should only really be used in the case that such a rotation cannot be described by a
        rotation of the underlying single-particle basis, since highly efficient routines already exist for this case..
        """

        # Convert rots from full-space particle-hole excitations into AO pairs.

        rota, rotb = rot[:, :self.ov_mf[0]], rot[:, self.ov_mf[0]:sum(self.ov_mf)]

        if hasattr(self.base.mf, "with_df"):
            rota, rotb = self.conv_to_aos(rota, rotb)
            # Loop through cderis
            res = np.zeros((rot.shape[0], rot.shape[0]))
            for eri1 in self.mf.with_df.loop():
                l_ = einsum("npq,lpq->nl", pyscf.lib.unpack_tril(eri1), rota + rotb)

                res += dot(l_.T, l_)
            return res
        else:
            # This is painful to do for each fragment, but comes from working with 4-index eris.
            eris_aa, eris_ab, eris_bb = self.base.get_eris_array(self.mf.mo_coeff)
            eris_aa = eris_aa[:self.base.nocc[0], self.base.nocc[0]:, :self.base.nocc[0], self.base.nocc[0]:].reshape(
                (self.ov_mf[0], self.ov_mf[0]))
            eris_ab = eris_ab[:self.base.nocc[0], self.base.nocc[0]:, :self.base.nocc[1], self.base.nocc[1]:].reshape(
                (self.ov_mf[0], self.ov_mf[1]))
            eris_bb = eris_bb[:self.base.nocc[1], self.base.nocc[1]:, :self.base.nocc[1], self.base.nocc[1]:].reshape(
                (self.ov_mf[1], self.ov_mf[1]))

            return dot(rota, eris_aa, rota.T) + dot(rota, eris_ab, rotb.T) + \
                   dot(rotb, eris_ab.T, rota.T) + dot(rotb, eris_bb, rotb.T)

    def get_edmet_energy_contrib(self):
        """Generate EDMET energy contribution, according to expression given in appendix of EDMET preprint"""
        e1, e2 = self.get_dmet_energy_contrib()
        c_act = self.cluster.c_active
        p_imp = self.get_fragment_projector(c_act)
        dm_eb = self._results.dm_eb
        # Have separate spin contributions.
        efb = 0.5 * (
                np.einsum("pr,npq,rqn", p_imp[0], self.couplings[0], dm_eb[0]) +
                np.einsum("qr,npq,prn", p_imp[0], self.couplings[0], dm_eb[0]) +
                np.einsum("pr,npq,rqn", p_imp[1], self.couplings[1], dm_eb[1]) +
                np.einsum("qr,npq,prn", p_imp[1], self.couplings[1], dm_eb[1])
        )
        return e1, e2, efb

    def get_rot_ov_frag(self):
        """Get rotations between the relevant space for fragment two-point excitations and the cluster active occupied-
        virtual excitations."""

        occ_frag_rot, vir_frag_rot = self.get_overlap_c2f()
        if self.opts.old_sc_condition:
            # Then get projectors to local quantities in ov-basis. Note this needs to be stacked to apply to each spin
            # pairing separately.
            rot_ov_frag = tuple([np.einsum("ip,ap->pia", x, y).reshape((-1, ov)) for x, y, ov in
                                 zip(occ_frag_rot, vir_frag_rot, self.ov_active)])
            # Get pseudo-inverse to map from frag to loc. Since occupied-virtual excitations aren't spanning this
            # isn't a simple transpose.
            proj_to_order = np.eye(rot_ov_frag[0].shape[0])
        else:
            # First, grab rotations from particle-hole excitations to fragment degrees of freedom, ignoring reordering
            rot_ov_frag = tuple([np.einsum("ip,aq->pqia", x, y).reshape((-1, ov)) for x, y, ov in
                                 zip(occ_frag_rot, vir_frag_rot, self.ov_active)])
            # Set up matrix to map down to only a single index ordering.
            # Note that we current assume the number of alpha and beta orbitals is equal
            nf = self.n_frag[0]
            proj_to_order = np.zeros((nf,) * 4)
            for p in range(nf):
                for q in range(p + 1):
                    proj_to_order[p, q, p, q] = proj_to_order[q, p, p, q] = 1.0
            proj_to_order = proj_to_order.reshape((nf ** 2, nf, nf))
            # Now restrict to triangular portion of array
            proj_to_order = pyscf.lib.pack_tril(proj_to_order)
            # proj_from_order = np.linalg.pinv(proj_to_order)
            # Now have rotation between single fragment ordering, and fragment particle-hole excits.
            rot_ov_frag = tuple([dot(proj_to_order.T, x) for x in rot_ov_frag])
            # Get pseudo-inverse to map from frag to loc. Since occupied-virtual excitations aren't spanning this
            # isn't a simple transpose.
        rot_frag_ov = tuple([np.linalg.pinv(x) for x in rot_ov_frag])

        # Return tuples so can can unified interface with UHF implementation.
        return rot_ov_frag, rot_frag_ov, proj_to_order

    def get_correlation_kernel_contrib(self, contrib):
        """Gets contribution to xc kernel in full space of system."""

        ca = dot(self.base.get_ovlp(), self.cluster.c_active[0])
        cb = dot(self.base.get_ovlp(), self.cluster.c_active[1])

        if self.base.with_df:
            return [tuple([einsum("nij,pi,qj->npq", x, c, c) for x in y]) for y,c in zip(contrib, [ca, cb])]
        else:
            v_aa, v_ab, v_bb = contrib
            v_aa = einsum("ijkl,pi,qj,rk,sl->pqrs", v_aa, ca, ca, ca, ca)
            v_ab = einsum("ijkl,pi,qj,rk,sl->pqrs", v_ab, ca, ca, cb, cb)
            v_bb = einsum("ijkl,pi,qj,rk,sl->pqrs", v_bb, cb, cb, cb, cb)
            return v_aa, v_ab, v_bb
