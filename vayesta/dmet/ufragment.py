import numpy as np

from vayesta.core.util import dot, einsum, log_time, timer
from vayesta.core.qemb import UFragment
from vayesta.dmet.fragment import DMETFragment

class UDMETFragment(UFragment, DMETFragment):

    def set_cas(self, *args, **kwargs):
        raise NotImplementedError()

    def get_frag_hl_dm(self):
        ca = dot(self.c_frag[0].T, self.mf.get_ovlp(), self.cluster.c_active[0])
        cb = dot(self.c_frag[1].T, self.mf.get_ovlp(), self.cluster.c_active[1])

        return dot(ca, self.results.dm1[0], ca.T), dot(cb, self.results.dm1[1], cb.T)

    def get_nelectron_hl(self):
        dma, dmb = self.get_frag_hl_dm()
        return dma.trace() + dmb.trace()

    def get_dmet_energy_contrib(self, hamil=None):
        """Calculate the contribution of this fragment to the overall DMET energy."""
        # Projector to the impurity in the active basis.
        p_imp_a, p_imp_b = self.get_fragment_projector(self.cluster.c_active)

        c_active = self.cluster.c_active
        t0 = timer()
        if hamil is None:
            hamil = self.hamil
        eris = hamil.get_eris_bare()

        fock = self.base.get_fock()
        fa = dot(c_active[0].T, fock[0], c_active[0])
        fb = dot(c_active[1].T, fock[1], c_active[1])
        oa = np.s_[:self.cluster.nocc_active[0]]
        ob = np.s_[:self.cluster.nocc_active[1]]
        gaa, gab, gbb = eris
        va = (einsum('iipq->pq', gaa[oa,oa]) + einsum('pqii->pq', gab[:,:,ob,ob])   # Coulomb
            - einsum('ipqi->pq', gaa[oa,:,:,oa]))                                   # Exchange
        vb = (einsum('iipq->pq', gbb[ob,ob]) + einsum('iipq->pq', gab[oa,oa])       # Coulomb
            - einsum('ipqi->pq', gbb[ob,:,:,ob]))                                   # Exchange
        h_eff = (fa-va, fb-vb)

        h_bare = tuple([dot(c.T, self.base.get_hcore(), c) for c in c_active])

        e1 = 0.5 * (dot(p_imp_a, h_bare[0] + h_eff[0], self.results.dm1[0]).trace() +
                    dot(p_imp_b, h_bare[1] + h_eff[1], self.results.dm1[1]).trace())

        e2 = 0.5 * (einsum("tp,pqrs,tqrs->", p_imp_a, eris[0], self.results.dm2[0]) +
                    einsum("tp,pqrs,tqrs->", p_imp_a, eris[1], self.results.dm2[1]) +
                    einsum("tr,pqrs,pqts->", p_imp_b, eris[1], self.results.dm2[1]) +
                    einsum("tp,pqrs,tqrs->", p_imp_b, eris[2], self.results.dm2[2]))

        # Code to generate the HF energy contribution for testing purposes.
        # mf_dm1 = np.linalg.multi_dot((c_act.T, self.base.get_ovlp(), self.mf.make_rdm1(),\
        #                               self.base.get_ovlp(), c_act))
        # e_hf = np.linalg.multi_dot((P_imp, 0.5 * (h_bare + f_act), mf_dm1)).trace()
        return e1, e2
