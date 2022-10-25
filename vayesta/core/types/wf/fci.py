import numpy as np
import pyscf
import pyscf.fci
import vayesta
from vayesta.core.util import *
from vayesta.core.types import wf as wf_types


def FCI_WaveFunction(mo, ci, **kwargs):
    if mo.nspin == 1:
        cls = RFCI_WaveFunction
    elif mo.nspin == 2:
        cls = UFCI_WaveFunction
    return cls(mo, ci, **kwargs)


class RFCI_WaveFunction(wf_types.WaveFunction):

    def __init__(self, mo, ci, projector=None):
        super().__init__(mo, projector=projector)
        self.ci = ci

    def make_rdm1(self, ao_basis=False, with_mf=True):
        dm1 = pyscf.fci.direct_spin1.make_rdm1(self.ci, self.norb, self.nelec)
        if not with_mf:
            dm1[np.diag_indices(self.nocc)] -= 2
        if not ao_basis:
            return dm1
        return dot(self.mo.coeff, dm1, self.mo.coeff.T)

    def make_rdm2(self, ao_basis=False, with_dm1=True, approx_cumulant=True):
        dm1, dm2 = pyscf.fci.direct_spin1.make_rdm12(self.ci, self.norb, self.nelec)
        if not with_dm1:
            if not approx_cumulant:
                dm2 -= (einsum('ij,kl->ijkl', dm1, dm1) - einsum('ij,kl->iklj', dm1, dm1)/2)
            elif (approx_cumulant in (1, True)):
                dm1[np.diag_indices(self.nocc)] -= 1
                for i in range(self.nocc):
                    dm2[i,i,:,:] -= 2*dm1
                    dm2[:,:,i,i] -= 2*dm1
                    dm2[:,i,i,:] += dm1
                    dm2[i,:,:,i] += dm1
            elif (approx_cumulant == 2):
                raise NotImplementedError
            else:
                raise ValueError
        if not ao_basis:
            return dm2
        return einsum('ijkl,ai,bj,ck,dl->abcd', dm2, *(4*[self.mo.coeff]))

    def project(self, projector, inplace=False):
        raise NotImplementedError

    def restore(self, projector=None, inplace=False):
        raise NotImplementedError

    @property
    def c0(self):
        return self.ci[0,0]

    def as_unrestricted(self):
        mo = self.mo.to_spin_orbitals()
        return UFCI_WaveFunction(mo, self.ci)

    def as_mp2(self):
        raise self.as_cisd().as_mp2()

    def as_cisd(self, c0=None):
        if self.projector is not None:
            raise NotImplementedError
        norb, nocc, nvir = self.norb, self.nocc, self.nvir
        t1addr, t1sign = pyscf.ci.cisd.t1strs(norb, nocc)
        c1 = self.ci[0,t1addr] * t1sign
        c2 = einsum('i,j,ij->ij', t1sign, t1sign, self.ci[t1addr[:,None],t1addr])
        c1 = c1.reshape(nocc,nvir)
        c2 = c2.reshape(nocc,nvir,nocc,nvir).transpose(0,2,1,3)
        if c0 is None:
            c0 = self.c0
        else:
            c1 *= c0/self.c0
            c2 *= c0/self.c0
        return wf_types.RCISD_WaveFunction(self.mo, c0, c1, c2, projector=self.projector)

    def as_ccsd(self):
        return self.as_cisd().as_ccsd()

    def as_fci(self):
        return self


class UFCI_WaveFunction(RFCI_WaveFunction):

    def make_rdm1(self, ao_basis=False, with_mf=True):
        assert (self.norb[0] == self.norb[1])
        dm1 = pyscf.fci.direct_spin1.make_rdm1s(self.ci, self.norb[0], self.nelec)
        if not with_mf:
            dm1[0][np.diag_indices(self.nocc[0])] -= 1
            dm1[1][np.diag_indices(self.nocc[1])] -= 1
        if not ao_basis:
            return dm1
        return (dot(self.mo.coeff[0], dm1[0], self.mo.coeff[0].T),
                dot(self.mo.coeff[1], dm1[1], self.mo.coeff[1].T))

    def make_rdm2(self, ao_basis=False, with_dm1=True, approx_cumulant=True):
        assert (self.norb[0] == self.norb[1])
        dm1, dm2 = pyscf.fci.direct_spin1.make_rdm12s(self.ci, self.norb[0], self.nelec)
        if not with_dm1:
            dm1a, dm1b = dm1
            dm2aa, dm2ab, dm2bb = dm2
            if not approx_cumulant:
                dm2aa -= (einsum('ij,kl->ijkl', dm1a, dm1a) - einsum('ij,kl->iklj', dm1a, dm1a))
                dm2bb -= (einsum('ij,kl->ijkl', dm1b, dm1b) - einsum('ij,kl->iklj', dm1b, dm1b))
                dm2ab -= einsum('ij,kl->ijkl', dm1a, dm1b)
            elif (approx_cumulant in (1, True)):
                dm1a[np.diag_indices(self.nocca)] -= 0.5
                dm1b[np.diag_indices(self.noccb)] -= 0.5
                for i in range(self.nocca):
                    dm2aa[i,i,:,:] -= dm1a
                    dm2aa[:,:,i,i] -= dm1a
                    dm2aa[:,i,i,:] += dm1a
                    dm2aa[i,:,:,i] += dm1a
                    dm2ab[i,i,:,:] -= dm1b
                for i in range(self.noccb):
                    dm2bb[i,i,:,:] -= dm1b
                    dm2bb[:,:,i,i] -= dm1b
                    dm2bb[:,i,i,:] += dm1b
                    dm2bb[i,:,:,i] += dm1b
                    dm2ab[:,:,i,i] -= dm1a
            elif (approx_cumulant == 2):
                raise NotImplementedError
            else:
                raise ValueError
        if not ao_basis:
            return dm2
        moa, mob = self.mo.coeff
        return (einsum('ijkl,ai,bj,ck,dl->abcd', dm2[0], *(4*[moa])),
                einsum('ijkl,ai,bj,ck,dl->abcd', dm2[1], *[moa, moa, mob, mob]),
                einsum('ijkl,ai,bj,ck,dl->abcd', dm2[2], *(4*[mob])))

    def as_cisd(self, c0=None):
        if self.projector is not None:
            raise NotImplementedError
        norba, norbb = self.norb
        nocca, noccb = self.nocc
        nvira, nvirb = self.nvir

        t1addra, t1signa = pyscf.ci.cisd.tn_addrs_signs(norba, nocca, 1)
        t1addrb, t1signb = pyscf.ci.cisd.tn_addrs_signs(norbb, noccb, 1)
        t2addra, t2signa = pyscf.ci.cisd.tn_addrs_signs(norba, nocca, 2)
        t2addrb, t2signb = pyscf.ci.cisd.tn_addrs_signs(norbb, noccb, 2)
        na = pyscf.fci.cistring.num_strings(norba, nocca)
        nb = pyscf.fci.cistring.num_strings(norbb, noccb)

        ci = self.ci.reshape(na,nb)
        c1a = (self.ci[t1addra,0] * t1signa).reshape(nocca,nvira)
        c1b = (self.ci[0,t1addrb] * t1signb).reshape(noccb,nvirb)

        nocca_comp = nocca*(nocca-1)//2
        noccb_comp = noccb*(noccb-1)//2
        nvira_comp = nvira*(nvira-1)//2
        nvirb_comp = nvirb*(nvirb-1)//2
        c2aa = (self.ci[t2addra,0] * t2signa).reshape(nocca_comp, nvira_comp)
        c2bb = (self.ci[0,t2addrb] * t2signb).reshape(noccb_comp, nvirb_comp)
        c2aa = pyscf.cc.ccsd._unpack_4fold(c2aa, nocca, nvira)
        c2bb = pyscf.cc.ccsd._unpack_4fold(c2bb, noccb, nvirb)
        c2ab = einsum('i,j,ij->ij', t1signa, t1signb, self.ci[t1addra[:,None],t1addrb])
        c2ab = c2ab.reshape(nocca,nvira,noccb,nvirb).transpose(0,2,1,3)
        if c0 is None:
            c0 = self.c0
        else:
            c1a *= c0/self.c0
            c1b *= c0/self.c0
            c2aa *= c0/self.c0
            c2ab *= c0/self.c0
            c2bb *= c0/self.c0
        c1 = (c1a, c1b)
        c2 = (c2aa, c2ab, c2bb)
        return wf_types.UCISD_WaveFunction(self.mo, c0, c1, c2, projector=self.projector)
