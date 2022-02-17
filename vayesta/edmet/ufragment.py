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

    @property
    def r_bos_ao(self):
        if self.sym_parent is None:
            # Need to convert bosonic definition from ov-excitations into ao pairs.
            r_bos = self.r_bos
            co_a, co_b = self.base.mo_coeff_occ
            cv_a, cv_b = self.base.mo_coeff_vir

            r_bosa = r_bos[:, :self.ov_mf[0]].reshape((self.nbos, self.base.nocc[0], self.base.nvir[0]))
            r_bosb = r_bos[:, self.ov_mf[0]:].reshape((self.nbos, self.base.nocc[1], self.base.nvir[1]))
            return (einsum("nia,pi,qa->npq", r_bosa, co_a, cv_a), einsum("nia,pi,qa->npq", r_bosb, co_b, cv_b))
        else:
            r_bos_ao = self.sym_parent.r_bos_ao
            # Need to rotate to account for symmetry operations.
            r_bos_ao = tuple([self.sym_op(self.sym_op(x, axis=2), axis=1) for x in r_bos_ao])
        return r_bos_ao

    def get_fock(self):
        return self.base.get_fock()

    def get_co_active(self):
        return self.cluster.c_active_occ

    def get_cv_active(self):
        return self.cluster.c_active_vir

    def get_rot_to_mf_ov(self):
        r_o, r_v = self.get_overlap_m2c()
        spat_rota = einsum("iJ,aB->iaJB", r_o[0], r_v[0]).reshape((self.ov_mf[0], self.ov_active[0])).T
        spat_rotb = einsum("iJ,aB->iaJB", r_o[1], r_v[1]).reshape((self.ov_mf[1], self.ov_active[1])).T
        res = np.zeros((sum(self.ov_active), sum(self.ov_mf)))
        res[:self.ov_active[0], :self.ov_mf[0]] = spat_rota
        res[self.ov_active[0]:, self.ov_mf[0]:] = spat_rotb
        return res

    def get_fragment_projector_ov(self, proj="o", inc_bosons=False):
        """In space of cluster p-h excitations, generate the projector to the ."""
        if not ("o" in proj or "v" in proj):
            raise ValueError("Must project the occupied and/or virtual index to the fragment. Please specify at least "
                             "one")

        nex = self.ov_active_tot
        if inc_bosons:
            nex += self.nbos

        def get_ov_projector(poa, pob, pva, pvb):
            p_ova = einsum("ij,ab->iajb", poa, pva).reshape((self.ov_active[0], self.ov_active[0]))
            p_ovb = einsum("ij,ab->iajb", pob, pvb).reshape((self.ov_active[1], self.ov_active[1]))
            p_ov = np.zeros((nex, nex))
            p_ov[:self.ov_active[0], :self.ov_active[0]] = p_ova
            p_ov[self.ov_active[0]:self.ov_active_tot, self.ov_active[0]:self.ov_active_tot] = p_ovb
            return p_ov

        p_ov = np.zeros((nex, nex))
        if "o" in proj:
            poa, pob = self.get_fragment_projector(self.cluster.c_active_occ)
            pva, pvb = [np.eye(x) for x in self.cluster.nvir_active]
            p_ov += get_ov_projector(poa, pob, pva, pvb)
        if "v" in proj:
            poa, pob = [np.eye(x) for x in self.cluster.nocc_active]
            pva, pvb = self.get_fragment_projector(self.cluster.c_active_vir)
            p_ov += get_ov_projector(poa, pob, pva, pvb)
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
            # First get the contribution from the fermionic degrees of freedom.
            res = [tuple([einsum("nij,pi,qj->npq", x, c, c) for x in y]) for y, c in zip(contrib[:2], [ca, cb])]
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
            v_aa = einsum("ijkl,pi,qj,rk,sl->pqrs", v_aa, ca, ca, ca, ca)
            v_ab = einsum("ijkl,pi,qj,rk,sl->pqrs", v_ab, ca, ca, cb, cb)
            v_bb = einsum("ijkl,pi,qj,rk,sl->pqrs", v_bb, cb, cb, cb, cb)

            if self.opts.boson_xc_kernel:
                r_bosa, r_bosb = self.r_bos_ao
                # First bosonic excitations
                bos_v_aa = einsum("ijn,pi,qj,nrs->pqrs", fb_a, ca, ca, r_bosa)
                bos_v_aa += einsum("pqrs->rspq", bos_v_aa)
                bos_v_bb = einsum("ijn,pi,qj,nrs->pqrs", fb_b, cb, cb, r_bosb)
                bos_v_bb += einsum("pqrs->rspq", bos_v_bb)
                bos_v_ab = einsum("ijn,pi,qj,nrs->pqrs", fb_a, ca, ca, r_bosb)
                bos_v_ab += einsum("ijn,pi,qj,nrs->rspq", fb_b, cb, cb, r_bosa)
                # Bosonic dexcitations contributions swap pqrs->qpsr.
                bos_v_aa += einsum("pqrs->qpsr", bos_v_aa)
                bos_v_ab += einsum("pqrs->qpsr", bos_v_ab)
                bos_v_bb += einsum("pqrs->qpsr", bos_v_bb)

                v_aa += bos_v_aa
                v_ab += bos_v_ab
                v_bb += bos_v_bb

            self.prev_xc_contrib = (v_aa, v_ab, v_bb)

            return v_aa, v_ab, v_bb
