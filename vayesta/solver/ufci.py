import dataclasses

import numpy as np

import pyscf
import pyscf.ci
import pyscf.fci

from vayesta.core.util import *
from .fci2 import FCI_Solver


class UFCI_Solver(FCI_Solver):
    """FCI with UHF orbitals."""

    @dataclasses.dataclass
    class Options(FCI_Solver.Options):
        fix_spin: float = None

    #@property
    #def cas(self):
    #    cas = self.cluster.get_cas_size()
    #    if cas[1][0] != cas[1][1]:
    #        raise NotImplementedError()
    #    return (cas[0], cas[1][0])

    @property
    def ncas(self):
        ncas = self.cluster.norb_active
        if ncas[0] != ncas[1]:
            raise NotImplementedError()
        return ncas[0]

    @property
    def nelec(self):
        return self.cluster.nocc_active

    def get_solver(self):
        return pyscf.fci.direct_uhf.FCISolver(self.mol)

    def get_c2e(self):
        """C2 in intermediate normalization."""
        #c2aa = self.c2[0]
        #c2bb = self.c2[2]
        ca, cb = self.c1
        caa = self.c2[0] + einsum('ia,jb->ijab', ca, ca) - einsum('ib,ja->ijab', ca, ca)
        cbb = self.c2[2] + einsum('ia,jb->ijab', cb, cb) - einsum('ib,ja->ijab', cb, cb)
        cab = self.c2[1]
        return (caa, cab, cbb)

    def get_heff(self, eris, with_vext=True):
        c_active = self.cluster.c_active
        fock = self.base.get_fock()
        fa = dot(c_active[0].T, fock[0], c_active[0])
        fb = dot(c_active[1].T, fock[1], c_active[1])
        oa = np.s_[:self.cluster.nocc_active[0]]
        ob = np.s_[:self.cluster.nocc_active[1]]
        va = (einsum('iipq->pq', eris[0][oa,oa])        # Coulomb  a-a
            + einsum('pqii->pq', eris[1][:,:,ob,ob])    # Coulomb  a-b
            - einsum('ipqi->pq', eris[0][oa,:,:,oa]))   # Exchange a-a
        vb = (einsum('iipq->pq', eris[2][ob,ob])        # Coulomb  a-a
            + einsum('iipq->pq', eris[1][oa,oa])        # Coulomb  a-b
            - einsum('ipqi->pq', eris[2][ob,:,:,ob]))   # Exchange b-b
        h_eff = (fa-va, fb-vb)
        # TEST
        #hcore = self.base.get_hcore()
        #hcore2 = self.mf.get_hcore()
        #print(np.linalg.norm(hcore - hcore2))
        #assert np.allclose(hcore, hcore2)
        #h_eff = (dot(c_active[0].T, hcore, c_active[0]),
        #         dot(c_active[1].T, hcore, c_active[1]))
        #if with_vext and self.opts.v_ext is not None:
        #    h_eff[0] += self.opts.v_ext[0]
        #    h_eff[1] += self.opts.v_ext[1]
        #self.base.debug_h_eff = h_eff
        return h_eff

    def get_cisd_amps(self, civec):
        cisdvec = pyscf.ci.ucisd.from_fcivec(civec, self.ncas, self.nelec)
        c0, c1, c2 = pyscf.ci.ucisd.cisdvec_to_amplitudes(cisdvec, 2*[self.ncas], self.nelec)
        c1 = c1/c0
        c2 = c2/c0
        return c0, c1, c2
