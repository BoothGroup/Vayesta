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

    def as_cisdtq(self, c0=None):
        if self.projector is not None:
            raise NotImplementedError
        norb, nocc, nvir = self.norb, self.nocc, self.nvir
        # For packed 2D arrays
        ij_pairs = int(nocc * (nocc - 1) / 2)
        ab_pairs = int(nvir * (nvir - 1) / 2)
        ooidx = np.tril_indices(nocc, -1) # second index lower than first
        vvidx = np.tril_indices(nvir, -1) # second index lower than first
        # For packed 3D arrays
        oooidx = tril_indices_ndim(nocc, 3) # i > j > k
        vvvidx = tril_indices_ndim(nvir, 3) # a > b > c
        ijk_pairs = int(nocc * (nocc - 1) * (nocc - 2) / 6)
        abc_pairs = int(nvir * (nvir - 1) * (nvir - 2) / 6)

        # === C1 amplitudes ===
        # These functions extract out the indicies and signs of 
        # the *same spin* excitations of a given rank from the FCI vector
        t1addr, t1sign = pyscf.ci.cisd.tn_addrs_signs(norb, nocc, 1)
        # C1 are taken to be the beta -> beta excitations (which should be 
        # the same as alpha -> alpha), by taking the first (alpha) index to be doubly occupied.
        c1 = self.ci[0,t1addr] * t1sign
        c1 = c1.reshape((nocc, nvir))

        # Longhand check (to be put into a test)
        c1_ = np.zeros(t1addr.shape[0])
        c1_full = np.zeros_like(c1)
        for s_cnt, sing_ind in enumerate(t1addr):
            c1_[s_cnt] = self.ci[0, sing_ind] * t1sign[s_cnt]
            i = int(s_cnt / nvir)
            a = s_cnt % nvir
            c1_full[i,a] = c1_[s_cnt]
        assert(np.allclose(c1, c1_full))

        # === C2 amplitudes ===
        # For RHF, we want the (alpha, beta) -> (alpha, beta) excitation amplitudes.
        # Therefore, we can just take single excitations of alpha and 
        # combine with the single excitations of beta.
        c2 = np.einsum('i,j,ij->ij', t1sign, t1sign, self.ci[t1addr[:,None],t1addr])
        # Reorder occupied indices to the front
        c2 = c2.reshape((nocc, nvir, nocc, nvir)).transpose(0,2,1,3)

        # === C3 amplitudes ===
        # For the C3 amplitudes, we want to find the ijk -> abc amplitudes of 
        # spin signature (alpha, beta, alpha) -> (alpha, beta, alpha)

        # t2addr, t2sign is the index and sign of the packed (alpha, alpha) -> (alpha, alpha) 
        # excitations in the FCI array. To get the orbital indices that they correspond to,
        # use ooidx and vvidx
        t2addr, t2sign = pyscf.ci.cisd.tn_addrs_signs(norb, nocc, 2)
        assert(t2addr.shape[0] == ij_pairs * ab_pairs)

        # First find the ijk -> abc excitations, where ijab are alpha, and kc are beta
        c3_comp = np.zeros((ij_pairs * ab_pairs, nocc * nvir))
        c3 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir))
        for d_cnt, doub_ind in enumerate(t2addr):
            ij = int(d_cnt / ab_pairs)
            ab = d_cnt % ab_pairs
            i, j = ooidx[0][ij], ooidx[1][ij] # j ind < i ind
            a, b = vvidx[0][ab], vvidx[1][ab] # b ind < a ind
            for s_cnt, sing_ind in enumerate(t1addr):
                # First index of c3_comp is a compound index of ijab (alpha, alpha) excitations, 
                # with the second index being the kc (beta, beta) single excitation.
                c3_comp[d_cnt, s_cnt] = self.ci[doub_ind, sing_ind] * t2sign[d_cnt] * t1sign[s_cnt]

                k = int(s_cnt / nvir)
                c = s_cnt % nvir
                # Note, we want aba -> aba spin signature, not aab -> aab, which is what we have.
                # We can therefore swap (jk) and (bc). This does not cause an overall sign change.
                # We then also want to fill up the contributions between permutations 
                # amongst the alpha electrons and alpha holes.
                # If only one is permuted, then this will indeed cause a sign change.
                assert(i != j)
                assert(a != b)
                c3[i,k,j,a,c,b] = c3_comp[d_cnt, s_cnt]
                c3[j,k,i,a,c,b] = -c3_comp[d_cnt, s_cnt]
                c3[i,k,j,b,c,a] = -c3_comp[d_cnt, s_cnt]
                c3[j,k,i,b,c,a] = c3_comp[d_cnt, s_cnt]
        assert(np.allclose(c3_comp, np.einsum('i,j,ij->ij', \
            t2sign, t1sign, self.ci[t2addr[:,None], t1addr])))
        del c3_comp

        # === C4 amplitudes ===
        # For the C4 amplitudes, ijkl -> abcd, we are going to store two different spin
        # signatures: 
        # (alpha, beta, alpha, beta)  -> (alpha, beta, alpha, beta) and
        # (alpha, beta, alpha, alpha) -> (alpha, beta, alpha, alpha)
        # TODO: Can we store the information as a single combined spatial orbital representation?

        # Start with abab. We will first get this as aabb -> aabb, via a product of
        # alpha-alpha double excitations and beta-beta double excitations and then reorder.
        c4_abab = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir, nvir, nvir))
        c4_comp = np.zeros((ij_pairs * ab_pairs, ij_pairs * ab_pairs))
        for d_cnt_a, doub_ind_a in enumerate(t2addr):
            ij_alpha = int(d_cnt_a / ab_pairs)
            ab_alpha = d_cnt_a % ab_pairs
            i, j = ooidx[0][ij_alpha], ooidx[1][ij_alpha]
            a, b = vvidx[0][ab_alpha], vvidx[1][ab_alpha]
            for d_cnt_b, doub_ind_b in enumerate(t2addr):
                ij_beta = int(d_cnt_b / ab_pairs)
                ab_beta = d_cnt_b % ab_pairs
                I, J = ooidx[0][ij_beta], ooidx[1][ij_beta]
                A, B = vvidx[0][ab_beta], vvidx[1][ab_beta]
                
                # Swap aabb -> abab spin signature. No sign change required.
                c4_comp[d_cnt_a, d_cnt_b] = self.ci[doub_ind_a, doub_ind_b] \
                        * t2sign[d_cnt_a] * t2sign[d_cnt_b]
                # Consider all possible (antisymmetric) permutations of (i_alpha, j_alpha), 
                # (i_beta, j_beta), (a_alpha, b_alpha), (a_beta, b_beta). 16 options.
                c4_abab[i, I, j, J, a, A, b, B] =  c4_comp[d_cnt_a, d_cnt_b]
                c4_abab[i, I, j, J, a, B, b, A] = -c4_comp[d_cnt_a, d_cnt_b]
                c4_abab[i, I, j, J, b, A, a, B] = -c4_comp[d_cnt_a, d_cnt_b]
                c4_abab[i, I, j, J, b, B, a, A] =  c4_comp[d_cnt_a, d_cnt_b]
                c4_abab[i, J, j, I, a, A, b, B] = -c4_comp[d_cnt_a, d_cnt_b]
                c4_abab[i, J, j, I, a, B, b, A] =  c4_comp[d_cnt_a, d_cnt_b]
                c4_abab[i, J, j, I, b, A, a, B] =  c4_comp[d_cnt_a, d_cnt_b]
                c4_abab[i, J, j, I, b, B, a, A] = -c4_comp[d_cnt_a, d_cnt_b]
                c4_abab[j, I, i, J, a, A, b, B] = -c4_comp[d_cnt_a, d_cnt_b]
                c4_abab[j, I, i, J, a, B, b, A] =  c4_comp[d_cnt_a, d_cnt_b]
                c4_abab[j, I, i, J, b, A, a, B] =  c4_comp[d_cnt_a, d_cnt_b]
                c4_abab[j, I, i, J, b, B, a, A] = -c4_comp[d_cnt_a, d_cnt_b]
                c4_abab[j, J, i, I, a, A, b, B] =  c4_comp[d_cnt_a, d_cnt_b]
                c4_abab[j, J, i, I, a, B, b, A] = -c4_comp[d_cnt_a, d_cnt_b]
                c4_abab[j, J, i, I, b, A, a, B] = -c4_comp[d_cnt_a, d_cnt_b]
                c4_abab[j, J, i, I, b, B, a, A] =  c4_comp[d_cnt_a, d_cnt_b]
        assert(np.allclose(c4_comp, np.einsum('i,j,ij->ij', t2sign, t2sign, \
                self.ci[t2addr[:,None], t2addr])))
        del c4_comp

        # abaa spin signature. Get this from the aaab->aaab excitations.
        # This requires the index of the (alpha, alpha, alpha) -> (alpha, alpha, alpha) excits.
        t3addr, t3sign = pyscf.ci.cisd.tn_addrs_signs(norb, nocc, 3)
        assert(t3addr.shape[0] == ijk_pairs * abc_pairs)

        c4_abaa = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir, nvir, nvir))
        c4_comp = np.zeros((ijk_pairs * abc_pairs, nocc * nvir))
        for t_cnt_a, trip_ind_a in enumerate(t3addr):
            # Find alpha i,j,k -> a,b,c indices
            ijk_alpha = int(t_cnt_a / abc_pairs)
            abc_alpha = t_cnt_a % abc_pairs
            i, j, k = oooidx[0][ijk_alpha], oooidx[1][ijk_alpha], oooidx[2][ijk_alpha]
            a, b, c = vvvidx[0][abc_alpha], vvvidx[1][abc_alpha], vvvidx[2][abc_alpha]
            for s_cnt_b, sing_ind_b in enumerate(t1addr):
                c4_comp[t_cnt_a, s_cnt_b] = self.ci[trip_ind_a, sing_ind_b] * \
                        t3sign[t_cnt_a] * t1sign[s_cnt_b]

                # Beta singles values
                I = int(s_cnt / nvir)
                A = s_cnt % nvir

                # Swap aaab -> abaa spin signature. No sign change required.
                c4_abaa[i, I, j, k, a, A, b, c] = c4_comp[t_cnt_a, s_cnt_b]
                # All antisym permutations of (ijk) x (abc) amongst alpha orbitals.
                # Six permutations each, making 36 overall
                # just rearrange occupied
                c4_abaa[i, I, j, k, a, A, b, c] =  c4_comp[t_cnt_a, s_cnt_b]
                c4_abaa[i, I, k, j, a, A, b, c] = -c4_comp[t_cnt_a, s_cnt_b]
                c4_abaa[k, I, j, i, a, A, b, c] = -c4_comp[t_cnt_a, s_cnt_b]
                c4_abaa[j, I, i, k, a, A, b, c] = -c4_comp[t_cnt_a, s_cnt_b]
                c4_abaa[j, I, k, i, a, A, b, c] =  c4_comp[t_cnt_a, s_cnt_b]
                c4_abaa[k, I, i, j, a, A, b, c] =  c4_comp[t_cnt_a, s_cnt_b]
                # swap ac
                c4_abaa[i, I, j, k, c, A, b, a] = -c4_comp[t_cnt_a, s_cnt_b]
                c4_abaa[i, I, k, j, c, A, b, a] =  c4_comp[t_cnt_a, s_cnt_b]
                c4_abaa[k, I, j, i, c, A, b, a] =  c4_comp[t_cnt_a, s_cnt_b]
                c4_abaa[j, I, i, k, c, A, b, a] =  c4_comp[t_cnt_a, s_cnt_b]
                c4_abaa[j, I, k, i, c, A, b, a] = -c4_comp[t_cnt_a, s_cnt_b]
                c4_abaa[k, I, i, j, c, A, b, a] = -c4_comp[t_cnt_a, s_cnt_b]
                # swap ab
                c4_abaa[i, I, j, k, b, A, a, c] = -c4_comp[t_cnt_a, s_cnt_b]
                c4_abaa[i, I, k, j, b, A, a, c] =  c4_comp[t_cnt_a, s_cnt_b]
                c4_abaa[k, I, j, i, b, A, a, c] =  c4_comp[t_cnt_a, s_cnt_b]
                c4_abaa[j, I, i, k, b, A, a, c] =  c4_comp[t_cnt_a, s_cnt_b]
                c4_abaa[j, I, k, i, b, A, a, c] = -c4_comp[t_cnt_a, s_cnt_b]
                c4_abaa[k, I, i, j, b, A, a, c] = -c4_comp[t_cnt_a, s_cnt_b]
                # swap bc
                c4_abaa[i, I, j, k, a, A, c, b] = -c4_comp[t_cnt_a, s_cnt_b]
                c4_abaa[i, I, k, j, a, A, c, b] =  c4_comp[t_cnt_a, s_cnt_b]
                c4_abaa[k, I, j, i, a, A, c, b] =  c4_comp[t_cnt_a, s_cnt_b]
                c4_abaa[j, I, i, k, a, A, c, b] =  c4_comp[t_cnt_a, s_cnt_b]
                c4_abaa[j, I, k, i, a, A, c, b] = -c4_comp[t_cnt_a, s_cnt_b]
                c4_abaa[k, I, i, j, a, A, c, b] = -c4_comp[t_cnt_a, s_cnt_b]
                # swap abc -> cab
                c4_abaa[i, I, j, k, c, A, a, b] =  c4_comp[t_cnt_a, s_cnt_b]
                c4_abaa[i, I, k, j, c, A, a, b] = -c4_comp[t_cnt_a, s_cnt_b]
                c4_abaa[k, I, j, i, c, A, a, b] = -c4_comp[t_cnt_a, s_cnt_b]
                c4_abaa[j, I, i, k, c, A, a, b] = -c4_comp[t_cnt_a, s_cnt_b]
                c4_abaa[j, I, k, i, c, A, a, b] =  c4_comp[t_cnt_a, s_cnt_b]
                c4_abaa[k, I, i, j, c, A, a, b] =  c4_comp[t_cnt_a, s_cnt_b]
                # swap abc -> bca
                c4_abaa[i, I, j, k, b, A, c, a] =  c4_comp[t_cnt_a, s_cnt_b]
                c4_abaa[i, I, k, j, b, A, c, a] = -c4_comp[t_cnt_a, s_cnt_b]
                c4_abaa[k, I, j, i, b, A, c, a] = -c4_comp[t_cnt_a, s_cnt_b]
                c4_abaa[j, I, i, k, b, A, c, a] = -c4_comp[t_cnt_a, s_cnt_b]
                c4_abaa[j, I, k, i, b, A, c, a] =  c4_comp[t_cnt_a, s_cnt_b]
                c4_abaa[k, I, i, j, b, A, c, a] =  c4_comp[t_cnt_a, s_cnt_b]

        assert(np.allclose(c4_comp, np.einsum('i,j,ij->ij', t3sign, t1sign, \
                self.ci[t3addr[:,None], t1addr])))

        if c0 is None:
            c0 = self.c0
        else:
            c1 *= c0/self.c0
            c2 *= c0/self.c0
            c3 *= c0/self.c0
            c4_abab *= c0/self.c0
            c4_abaa *= c0/self.c0

        return wf_types.RCISDTQ_WaveFunction(self.mo, c0, c1, c2, c3, (c4_abaa, c4_abab))

    def as_ccsd(self):
        return self.as_cisd().as_ccsd()

    def as_ccsdtq(self):
        return self.as_cisdtq().as_ccsdtq()

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

    def as_cisdtq(self, c0=None):
        if self.projector is not None:
            raise NotImplementedError
        norba, norbb = self.norb
        nocca, noccb = self.nocc
        nvira, nvirb = self.nvir

        t1addra, t1signa = pyscf.ci.cisd.tn_addrs_signs(norba, nocca, 1)
        t1addrb, t1signb = pyscf.ci.cisd.tn_addrs_signs(norbb, noccb, 1)
        t2addra, t2signa = pyscf.ci.cisd.tn_addrs_signs(norba, nocca, 2)
        t2addrb, t2signb = pyscf.ci.cisd.tn_addrs_signs(norbb, noccb, 2)
        t3addra, t3signa = pyscf.ci.cisd.tn_addrs_signs(norba, nocca, 3)
        t3addrb, t3signb = pyscf.ci.cisd.tn_addrs_signs(norbb, noccb, 3)
        t4addra, t4signa = pyscf.ci.cisd.tn_addrs_signs(norba, nocca, 4)
        t4addrb, t4signb = pyscf.ci.cisd.tn_addrs_signs(norbb, noccb, 4)

        na = pyscf.fci.cistring.num_strings(norba, nocca)
        nb = pyscf.fci.cistring.num_strings(norbb, noccb)

        ci = self.ci.reshape(na,nb)
        # C1
        c1a = (self.ci[t1addra,0] * t1signa).reshape(nocca,nvira)
        c1b = (self.ci[0,t1addrb] * t1signb).reshape(noccb,nvirb)
        # C2
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
        # C3
        raise NotImplementedError
        # C4
        raise NotImplementedError

        if c0 is None:
            c0 = self.c0
        else:
            c1a *= c0/self.c0
            c1b *= c0/self.c0
            c2aa *= c0/self.c0
            c2ab *= c0/self.c0
            c2bb *= c0/self.c0
            # TODO
            c3aaa *= c0/self.c0
        c1 = (c1a, c1b)
        c2 = (c2aa, c2ab, c2bb)
        # TODO
        #c3 = (c3aaa, ...
        #c4 = (c4aaaa, ...
        return wf_types.UCISDTQ_WaveFunction(self.mo, c0, c1, c2, c3, c4, projector=self.projector)
