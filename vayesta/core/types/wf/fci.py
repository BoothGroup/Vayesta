import numpy as np
import pyscf
import pyscf.fci
from vayesta.core.util import decompress_axes, dot, einsum, tril_indices_ndim, replace_attr
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

        # Change to arrays, in case of empty slice
        t1addr = np.asarray(t1addr, dtype=int)

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

        t1addr, t1sign = pyscf.ci.cisd.tn_addrs_signs(norb, nocc, 1)
        t2addr, t2sign = pyscf.ci.cisd.tn_addrs_signs(norb, nocc, 2)
        t3addr, t3sign = pyscf.ci.cisd.tn_addrs_signs(norb, nocc, 3)
        t4addr, t4sign = pyscf.ci.cisd.tn_addrs_signs(norb, nocc, 4)

        t1addr = np.asarray(t1addr, dtype=int)
        t2addr = np.asarray(t2addr, dtype=int)
        t3addr = np.asarray(t3addr, dtype=int)
        t4addr = np.asarray(t4addr, dtype=int)

        # === C1 amplitudes ===
        # These functions extract out the indicies and signs of 
        # the *same spin* excitations of a given rank from the FCI vector
        # C1 are taken to be the beta -> beta excitations (which should be 
        # the same as alpha -> alpha), by taking the first (alpha) index to be doubly occupied.
        c1 = self.ci[0,t1addr] * t1sign
        c1 = c1.reshape((nocc, nvir))

        # === C2 amplitudes ===
        # For RHF, we want the (alpha, beta) -> (alpha, beta) excitation amplitudes.
        # Therefore, we can just take single excitations of alpha and 
        # combine with the single excitations of beta.
        c2 = np.einsum('i,j,ij->ij', t1sign, t1sign, self.ci[t1addr[:, None], t1addr])
        c2 = c2.reshape((nocc, nvir, nocc, nvir))
        c2 = c2.transpose(0, 2, 1, 3)

        # === C3 amplitudes ===
        # For the C3 amplitudes, we want to find the ijk -> abc amplitudes of 
        # spin signature (alpha, beta, alpha) -> (alpha, beta, alpha)
        c3 = np.einsum('i,j,ij->ij', t2sign, t1sign, self.ci[t2addr[:, None], t1addr])
        c3 = decompress_axes("iiaajb", c3, shape=(nocc, nocc, nvir, nvir, nocc, nvir))
        c3 = c3.transpose(0, 4, 1, 2, 5, 3)

        # === C4 amplitudes ===
        # For the C4 amplitudes, ijkl -> abcd, we are going to store two different spin
        # signatures: 
        # (alpha, beta, alpha, beta)  -> (alpha, beta, alpha, beta) and
        # (alpha, beta, alpha, alpha) -> (alpha, beta, alpha, alpha)
        c4_abaa = np.einsum('i,j,ij->ij', t3sign, t1sign, self.ci[t3addr[:, None], t1addr])
        c4_abaa = decompress_axes("iiiaaajb", c4_abaa, shape=(nocc, nocc, nocc, nvir, nvir, nvir, nocc, nvir))
        c4_abaa = c4_abaa.transpose(0, 6, 2, 1, 3, 7, 5, 4)
        c4_abab = np.einsum('i,j,ij->ij', t2sign, t2sign, self.ci[t2addr[:, None], t2addr])
        c4_abab = decompress_axes("iiaajjbb", c4_abab, shape=(nocc, nocc, nvir, nvir, nocc, nocc, nvir, nvir))
        c4_abab = c4_abab.transpose(0, 4, 1, 5, 2, 6, 3, 7)

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

        # Change to arrays, in case of empty slice
        t1addra = np.asarray(t1addra, dtype=int)
        t1addrb = np.asarray(t1addrb, dtype=int)

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
        
        ij_pairs_a = int(nocca * (nocca - 1) / 2)
        ab_pairs_a = int(nvira * (nvira - 1) / 2)
        ij_pairs_b = int(noccb * (noccb - 1) / 2)
        ab_pairs_b = int(nvirb * (nvirb - 1) / 2)
        ooidx_a = np.tril_indices(nocca, -1) # second index lower than first
        vvidx_a = np.tril_indices(nvira, -1) # second index lower than first
        ooidx_b = np.tril_indices(noccb, -1) # second index lower than first
        vvidx_b = np.tril_indices(nvirb, -1) # second index lower than first
        # For packed 3D arrays
        oooidx_a = tril_indices_ndim(nocca, 3) # i > j > k
        vvvidx_a = tril_indices_ndim(nvira, 3) # a > b > c
        ijk_pairs_a = int(nocca * (nocca - 1) * (nocca - 2) / 6)
        abc_pairs_a = int(nvira * (nvira - 1) * (nvira - 2) / 6)
        oooidx_b = tril_indices_ndim(noccb, 3) # i > j > k
        vvvidx_b = tril_indices_ndim(nvirb, 3) # a > b > c
        ijk_pairs_b = int(noccb * (noccb - 1) * (noccb - 2) / 6)
        abc_pairs_b = int(nvirb * (nvirb - 1) * (nvirb - 2) / 6)

        ijkl_pairs_a = int(nocca * (nocca - 1) * (nocca - 2) * (nocca - 3) / 24)
        abcd_pairs_a = int(nvira * (nvira - 1) * (nvira - 2) * (nvira - 3) / 24)
        ijkl_pairs_b = int(noccb * (noccb - 1) * (noccb - 2) * (noccb - 3) / 24)
        abcd_pairs_b = int(nvirb * (nvirb - 1) * (nvirb - 2) * (nvirb - 3) / 24)

        t1addra, t1signa = pyscf.ci.cisd.tn_addrs_signs(norba, nocca, 1)
        t1addrb, t1signb = pyscf.ci.cisd.tn_addrs_signs(norbb, noccb, 1)
        t2addra, t2signa = pyscf.ci.cisd.tn_addrs_signs(norba, nocca, 2)
        t2addrb, t2signb = pyscf.ci.cisd.tn_addrs_signs(norbb, noccb, 2)
        t3addra, t3signa = pyscf.ci.cisd.tn_addrs_signs(norba, nocca, 3)
        t3addrb, t3signb = pyscf.ci.cisd.tn_addrs_signs(norbb, noccb, 3)
        t4addra, t4signa = pyscf.ci.cisd.tn_addrs_signs(norba, nocca, 4)
        t4addrb, t4signb = pyscf.ci.cisd.tn_addrs_signs(norbb, noccb, 4)

        # Change to arrays, in case of empty slice
        t1addra = np.asarray(t1addra, dtype=int)
        t1addrb = np.asarray(t1addrb, dtype=int)
        t2addra = np.asarray(t2addra, dtype=int)
        t2addrb = np.asarray(t2addrb, dtype=int)
        t3addra = np.asarray(t3addra, dtype=int)
        t3addrb = np.asarray(t3addrb, dtype=int)
        t4addra = np.asarray(t4addra, dtype=int)
        t4addrb = np.asarray(t4addrb, dtype=int)

        na = pyscf.fci.cistring.num_strings(norba, nocca)
        nb = pyscf.fci.cistring.num_strings(norbb, noccb)

        # C1
        c1_a = (self.ci[t1addra,0] * t1signa).reshape(nocca,nvira)
        c1_b = (self.ci[0,t1addrb] * t1signb).reshape(noccb,nvirb)

        # C2
        c2_aa = (self.ci[t2addra,0] * t2signa).reshape(ij_pairs_a, ab_pairs_a)
        c2_aa = pyscf.cc.ccsd._unpack_4fold(c2_aa, nocca, nvira)

        c2_bb = (self.ci[0,t2addrb] * t2signb).reshape(ij_pairs_b, ab_pairs_b)
        c2_bb = pyscf.cc.ccsd._unpack_4fold(c2_bb, noccb, nvirb)

        c2_ab = einsum('i,j,ij->ij', t1signa, t1signb, self.ci[t1addra[:, None], t1addrb])
        c2_ab = c2_ab.reshape(nocca, nvira, noccb, nvirb)
        c2_ab = c2_ab.transpose(0, 2, 1, 3)

        # C3
        c3_aaa = (self.ci[t3addra,0] * t3signa).reshape(ijk_pairs_a, abc_pairs_a)
        c3_aaa = decompress_axes("iiiaaa", c3_aaa, shape=(nocca, nocca, nocca, nvira, nvira, nvira))

        c3_bbb = (self.ci[0,t3addrb] * t3signb).reshape(ijk_pairs_b, abc_pairs_b)
        c3_bbb = decompress_axes("iiiaaa", c3_bbb, shape=(noccb, noccb, noccb, nvirb, nvirb, nvirb))

        c3_aba = np.einsum('i,j,ij->ij', t2signa, t1signb, self.ci[t2addra[:, None], t1addrb])
        c3_aba = decompress_axes("iiaajb", c3_aba, shape=(nocca, nocca, nvira, nvira, noccb, nvirb))
        c3_aba = c3_aba.transpose(0, 4, 1, 2, 5, 3)

        c3_bab = np.einsum('i,j,ij->ij', t1signa, t2signb, self.ci[t1addra[:, None], t2addrb])
        c3_bab = decompress_axes("iajjbb", c3_bab, shape=(nocca, nvira, noccb, noccb, nvirb, nvirb))
        c3_bab = c3_bab.transpose(2, 0, 3, 4, 1, 5)

        # C4
        c4_aaaa = (self.ci[t4addra,0] * t4signa).reshape(ijkl_pairs_a, abcd_pairs_a)
        c4_aaaa = decompress_axes("iiiiaaaa", c4_aaaa, shape=(nocca, nocca, nocca, nocca, nvira, nvira, nvira, nvira))

        c4_bbbb = (self.ci[0,t4addrb] * t4signb).reshape(ijkl_pairs_b, abcd_pairs_b)
        c4_bbbb = decompress_axes("iiiiaaaa", c4_bbbb, shape=(noccb, noccb, noccb, noccb, nvirb, nvirb, nvirb, nvirb))

        c4_aaab = np.einsum('i,j,ij->ij', t3signa, t1signb, self.ci[t3addra[:,None], t1addrb])
        c4_aaab = decompress_axes("iiiaaajb", c4_aaab, shape=(nocca, nocca, nocca, nvira, nvira, nvira, noccb, nvirb))
        c4_aaab = c4_aaab.transpose(0, 1, 2, 6, 3, 4, 5, 7)

        c4_abab = np.einsum('i,j,ij->ij', t2signa, t2signb, self.ci[t2addra[:,None], t2addrb])
        c4_abab = decompress_axes("iiaajjbb", c4_abab, shape=(nocca, nocca, nvira, nvira, noccb, noccb, nvirb, nvirb))
        c4_abab = c4_abab.transpose(0, 4, 1, 5, 2, 6, 3, 7)

        c4_abbb = np.einsum('i,j,ij->ij', t1signa, t3signb, self.ci[t1addra[:,None], t3addrb])
        c4_abbb = decompress_axes("iajjjbbb", c4_abbb, shape=(nocca, nvira, noccb, noccb, noccb, nvirb, nvirb, nvirb))
        c4_abbb = c4_abbb.transpose(0, 2, 3, 4, 1, 5, 6, 7)

        c1 = (c1_a, c1_b)
        c2 = (c2_aa, c2_ab, c2_bb)
        c3 = (c3_aaa, c3_aba, c3_bab, c3_bbb)
        c4 = (c4_aaaa, c4_aaab, c4_abab, c4_abbb, c4_bbbb)

        if c0 is None:
            c0 = self.c0
        else:
            fac = c0 / self.c0
            c1 = tuple(c * fac for c in c1)
            c2 = tuple(c * fac for c in c2)
            c3 = tuple(c * fac for c in c3)
            c4 = tuple(c * fac for c in c4)

        return wf_types.UCISDTQ_WaveFunction(self.mo, c0, c1, c2, c3, c4)


class UFCI_WaveFunction_w_dummy(UFCI_WaveFunction):
    """Class to allow use of dummy orbitals to balance alpha and beta spin channels.
    This is done by introducing a dummy `SpinOrbitals` object during calculation of properties in orbital basis, then
    removal of dummy indices from these quantities.
    We currently choose to only introduce virtual orbitals.

    TODO check all quantities removed are negligible.
    """

    def __init__(self, mo, ci, dummy_orbs, projector=None):
        super().__init__(mo, ci, projector)
        self.dummy_orbs = dummy_orbs

        if len(dummy_orbs[0]) > 0:
            dummy_occ = min(dummy_orbs[0]) < self.nocca
        else:
            dummy_occ = min(dummy_orbs[1]) < self.noccb
        if dummy_occ:
            raise NotImplementedError("Only dummy virtual orbitals are supported.")
        norb = np.array(self.ndummy) + np.array(self.norb)
        if norb[0] != norb[1]:
            raise RuntimeError("Including padded orbitals doesn't match the number of orbitals in each spin channel!"
                               " %d != %d (%d + %d != %d + %d)" % (norb[0], norb[1], self.ndummy[0], self.norb[0],
                                                                   self.ndummy[1], self.norb[1]))

    @property
    def ndummy(self):
        return tuple([len(x) for x in self.dummy_orbs])

    @property
    def dummy_mo(self):
        # Dummy orbital object to impersonate correct number of orbitals for pyscf routines.
        coeff = self.mo.coeff
        norb = np.array(self.ndummy) + np.array(self.norb)
        nao = coeff[0].shape[0]
        # Generate coefficients of correct dimension, but with zero contribution.

        coeff_w_dummy = [np.zeros((norb[0], nao)), np.zeros((norb[0], nao))]
        sa, sb = self._phys_ind_orbs()

        coeff_w_dummy[0][sa] = coeff[0].T
        coeff_w_dummy[1][sb] = coeff[1].T

        coeff_w_dummy = [x.T for x in coeff_w_dummy]
        return type(self.mo)(coeff_w_dummy, occ=self.mo.nocc)

    def _phys_ind_orbs(self):
        return [np.array([i for i in range(y) if i not in x]) for x, y in zip(self.dummy_orbs, self.norb)]

    def _phys_ind_vir(self):
        return [np.array([i for i in range(y) if i + z not in x]) for x, y, z in zip(self.dummy_orbs, self.nvir, self.nocc)]

    def make_rdm1(self, ao_basis=False, *args, **kwargs):
        with replace_attr(self, mo=self.dummy_mo):
            dm1 = super().make_rdm1(*args, ao_basis=ao_basis, **kwargs)
        if ao_basis:
            return dm1
        sa, sb = self._phys_ind_orbs()
        return (dm1[0][np.ix_(sa, sa)], dm1[1][np.ix_(sb, sb)])

    def make_rdm2(self, ao_basis=False, *args, **kwargs):
        with replace_attr(self, mo=self.dummy_mo):
            dm2 = super().make_rdm2(*args, ao_basis=ao_basis, **kwargs)
        if ao_basis:
            return dm2
        sa, sb = self._phys_ind_orbs()
        return (dm2[0][np.ix_(sa, sa, sa, sa)], dm2[1][np.ix_(sa, sa, sb, sb)], dm2[2][np.ix_(sb, sb, sb, sb)])

    def as_cisd(self, *args, **kwargs):
        self.check_norb()
        with replace_attr(self, mo=self.dummy_mo):
            wf_cisd = super().as_cisd(*args, **kwargs)
            va, vb = self._phys_ind_vir()

        c1a, c1b = wf_cisd.c1
        c2aa, c2ab, c2bb = wf_cisd.c2

        # Define function to apply slices to virtual orbitals only
        def vsl(a, sl):
            # Swap slicelist order as well for consistency
            return a.transpose()[np.ix_(*sl[::-1])].transpose()

        c1 = (vsl(c1a, [va]), vsl(c1b, [vb]))
        c2 = (vsl(c2aa, [va, va]), vsl(c2ab, [va, vb]), vsl(c2bb, [vb, vb]))
        return wf_types.UCISD_WaveFunction(self.mo, wf_cisd.c0, c1, c2, projector=self.projector)

    def as_cisdtq(self, *args, **kwargs):
        with replace_attr(self, mo=self.dummy_mo):
            wf_cisdtq = super().as_cisdtq(*args, **kwargs)
            va, vb = self._phys_ind_vir()

        # Have wavefunction, but with dummy indices.
        c1a, c1b = wf_cisdtq.c1
        c2aa, c2ab, c2bb = wf_cisdtq.c2
        c3aaa, c3aba, c3bab, c3bbb = wf_cisdtq.c3
        c4aaaa, c4aaab, c4abab, c4abbb, c4bbbb = wf_cisdtq.c4

        # Define function to apply slices to virtual orbitals only
        def vsl(a, sl):
            # Swap slicelist order as well for consistency
            return a.transpose()[np.ix_(*sl[::-1])].transpose()

        c1 = (vsl(c1a, [va]), vsl(c1b, [vb]))
        c2 = (vsl(c2aa, [va, va]), vsl(c2ab, [va, vb]), vsl(c2bb, [vb, vb]))

        c3 = (vsl(c3aaa, [va, va, va]), vsl(c3aba, [va, vb, va]), vsl(c3bab, [vb, va, vb]), vsl(c3bbb, [vb, vb, vb]))

        c4 = (vsl(c4aaaa, [va, va, va, va]), vsl(c4aaab, [va, va, va, vb]), vsl(c4abab, [va, vb, va, vb]),
              vsl(c4abbb, [va, vb, vb, vb]), vsl(c4bbbb, [vb, vb, vb, vb]))

        return wf_types.UCISDTQ_WaveFunction(self.mo, wf_cisdtq.c0, c1, c2, c3, c4)
