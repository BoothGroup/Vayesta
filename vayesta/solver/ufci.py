import dataclasses

import numpy as np

import pyscf
import pyscf.ci
import pyscf.cc
import pyscf.fci

from vayesta.core.util import *
from .fci2 import FCI_Solver


class UFCI_Solver(FCI_Solver):
    """FCI with UHF orbitals."""

    @dataclasses.dataclass
    class Options(FCI_Solver.Options):
        fix_spin: float = None

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

    def get_heff(self, eris, with_vext=True):
        c_active = self.cluster.c_active
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

    #def get_cisd_amps(self, civec):
    #    cisdvec = pyscf.ci.ucisd.from_fcivec(civec, self.ncas, self.nelec)
    #    c0, (c1a, c1b), (c2aa, c2ab, c2bb) = pyscf.ci.ucisd.cisdvec_to_amplitudes(cisdvec, 2*[self.ncas], self.nelec)
    #    c1a = c1a/c0
    #    c1b = c1b/c0
    #    c2aa = c2aa/c0
    #    c2ab = c2ab/c0
    #    c2bb = c2bb/c0

    #    return c0, (c1a, c1b), (c2aa, c2ab, c2bb)

    def get_cisd_amps(self, civec):
        norba, norbb = self.cluster.norb_active
        nocca, noccb = self.cluster.nocc_active
        nvira, nvirb = self.cluster.nvir_active

        t1addra, t1signa = pyscf.ci.cisd.tn_addrs_signs(norba, nocca, 1)
        t1addrb, t1signb = pyscf.ci.cisd.tn_addrs_signs(norbb, noccb, 1)
        t2addra, t2signa = pyscf.ci.cisd.tn_addrs_signs(norba, nocca, 2)
        t2addrb, t2signb = pyscf.ci.cisd.tn_addrs_signs(norbb, noccb, 2)
        na = pyscf.fci.cistring.num_strings(norba, nocca)
        nb = pyscf.fci.cistring.num_strings(norbb, noccb)

        civec = civec.reshape(na,nb)
        c0 = civec[0,0]
        c1a = (civec[t1addra,0] * t1signa).reshape(nocca,nvira)
        c1b = (civec[0,t1addrb] * t1signb).reshape(noccb,nvirb)

        nocca_comp = nocca*(nocca-1)//2
        noccb_comp = noccb*(noccb-1)//2
        nvira_comp = nvira*(nvira-1)//2
        nvirb_comp = nvirb*(nvirb-1)//2
        c2aa = (civec[t2addra,0] * t2signa).reshape(nocca_comp, nvira_comp)
        c2bb = (civec[0,t2addrb] * t2signb).reshape(noccb_comp, nvirb_comp)
        c2aa = pyscf.cc.ccsd._unpack_4fold(c2aa, nocca, nvira)
        c2bb = pyscf.cc.ccsd._unpack_4fold(c2bb, noccb, nvirb)
        c2ab = einsum('i,j,ij->ij', t1signa, t1signb, civec[t1addra[:,None],t1addrb])
        c2ab = c2ab.reshape(nocca,nvira,noccb,nvirb).transpose(0,2,1,3)

        # C1 and C2 in intermediate normalization:
        c1a = c1a/c0
        c1b = c1b/c0
        c2aa = c2aa/c0
        c2ab = c2ab/c0
        c2bb = c2bb/c0
        return c0, (c1a, c1b), (c2aa, c2ab, c2bb)

    def make_rdm1(self, civec=None):
        if civec is None: civec = self.civec
        self.dm1 = self.solver.make_rdm1s(civec, self.ncas, self.nelec)
        return self.dm1

    def make_rdm12(self, civec=None):
        if civec is None: civec = self.civec
        self.dm1, self.dm2 = self.solver.make_rdm12s(civec, self.ncas, self.nelec)
        return self.dm1, self.dm2
