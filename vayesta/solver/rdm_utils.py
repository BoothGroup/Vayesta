'''
06 March 2020
robert.anderson@kcl.ac.uk
'''

import numpy as np
from itertools import combinations, product, permutations
import pyscf, string

'''
flat spin-orbital [0, 2*nspat) index -> spatial orbital [0, nspat) and spin [0, 1] (aka [a, b])
two mappings: 
            | 0 1 2 3 4 5 6 7 8 9
spin-minor  | a b a b a b a b a b
            | 0 0 1 1 2 2 3 3 4 4

            | 0 1 2 3 4 5 6 7 8 9
spin-major  | a a a a a b b b b b
            | 0 1 2 3 4 0 1 2 3 4

spin-major requires nspat, whereas spin-minor does not. spin-major mode is therefore selected
in the following methods by passing something other than None as nspat
'''

def flat(spatorb, spin, nspat):
    return spatorb*2+spin if nspat is None else spatorb+nspat*spin
def spat(spinorb, nspat):
    return spinorb//2 if nspat is None else spinorb%nspat
def spats(spinorbs, nspat):
    return tuple(spat(i, nspat) for i in spinorbs)
def spin(spinorb, nspat):
    return spinorb%2 if nspat is None else spinorb//nspat
def spins(spinorbs, nspat):
    return tuple(spin(i, nspat) for i in spinorbs)
def total_spin(spinorbs, nspat):
    return sum(spins(spinorbs, nspat))
def kramers_op(spinorb, nspat):
    return tuple(flat(spat(i, nspat), int(not spin(i, nspat)), nspat) for i in spinorb)

'''
tests for common spins in adjacent (cre/des ops) rdm indices
'''
def rdm_paired_spins(spinorbs, nspat):
    return all(spin(spinorbs[2*i], nspat)==spin(spinorbs[2*i+1], nspat) for i in range(len(spinorbs)//2))

def parity(tup):
    work = list(tup)
    t = 0
    for i in range(len(tup)):
        if work[i]==i: continue
        for j in range(i+1, len(tup)):
            if work[j]==i:
                work[i], work[j] = work[j], work[i]
                t+=1
    assert work==list(range(len(tup)))
    return 1-2*(t%2)

assert parity((0,))==1

assert parity((0,1))==1
assert parity((1,0))==-1

assert parity((0,1,2))==1
assert parity((0,2,1))==-1
assert parity((2,0,1))==1
assert parity((2,1,0))==-1
assert parity((1,2,0))==1
assert parity((1,0,2))==-1

def interleave(v1, v2):
    return tuple(reduce(lambda a,b:a+b, zip(v1, v2)))

def make_all_perms(rank):
    all_perms = []
    for order in permutations(range(rank)):
        all_perms.append([order, parity(order)])
    return all_perms

def put_permutations(rdm, perms, creinds, desinds, element):
    for creorder, creparity in perms:
        creperm = tuple(creinds[i] for i in creorder)
        for desorder, desparity in perms:
            desperm = tuple(desinds[i] for i in desorder)
            inds = interleave(creperm, desperm)
            if abs(rdm[inds])>1e-12: 
                assert abs(rdm[inds]-element*creparity*desparity)<1e-12, "permutationally symmetric element already set to a different value"
            rdm[inds] = element*creparity*desparity
            
'''
assume that only the ascending-ordered tuples of creation and annihilation spin orbital indices are non-zero in the nord_rdm,
an fill the other elements that are related by permutational symmetry
'''
def restore_perm_syms(nord_rdm, spin_major, time_reversal_sym, tol=1e-12):
    norb = nord_rdm.shape[0]//2
    rank = len(nord_rdm.shape)//2
    sm = norb if spin_major else None
    nspinorb = norb*2
    rdm = np.zeros((nspinorb,)*(rank*2))
    perms = make_all_perms(rank)
    for icre, creinds in enumerate(combinations(range(nspinorb), rank)):
        crespins = spins(creinds, sm)
        for ides, desinds in enumerate(combinations(range(nspinorb), rank)):
            if ides > icre: continue # hermiticity (T)
            desspins = spins(desinds, sm)
            if sum(crespins)!=sum(desspins): continue # spin conservation
            if time_reversal_sym and desspins[0]: continue # time reversal symmetry (K)
            element = nord_rdm[interleave(creinds, desinds)]
            if abs(element)<tol: continue

            put_permutations(rdm, perms, creinds, desinds, element)
            put_permutations(rdm, perms, desinds, creinds, element) # T
            if time_reversal_sym:
                put_permutations(rdm, perms, kramers_op(creinds, sm), kramers_op(desinds, sm), element) # K
                put_permutations(rdm, perms, kramers_op(desinds, sm), kramers_op(creinds, sm), element) # T and K
    return rdm

def make_spin_resolved_rdm(norb, rank, get_expval, spin_major, time_reversal_sym, tol=1e-12):
    sm = norb if spin_major else None
    nspinorb = norb*2
    rdm = np.zeros((nspinorb,)*(rank*2))
    perms = make_all_perms(rank)
    for icre, creinds in enumerate(combinations(range(nspinorb), rank)):
        crespins = spins(creinds, sm)
        for ides, desinds in enumerate(combinations(range(nspinorb), rank)):
            if ides > icre: continue # hermiticity (T)
            desspins = spins(desinds, sm)
            if sum(crespins)!=sum(desspins): continue # spin conservation
            if time_reversal_sym and desspins[0]: continue # time reversal symmetry (K)
            element = get_expval(creinds, desinds)
            if abs(element)<tol: continue

            put_permutations(rdm, perms, creinds, desinds, element)
            put_permutations(rdm, perms, desinds, creinds, element) # T
            if time_reversal_sym:
                put_permutations(rdm, perms, kramers_op(creinds, sm), kramers_op(desinds, sm), element) # K
                put_permutations(rdm, perms, kramers_op(desinds, sm), kramers_op(creinds, sm), element) # T and K
    return rdm

def spin_resolved_to_spinfree(spin_resolved_rdm, spin_major):
    rank = len(spin_resolved_rdm.shape)//2
    nspatorb = spin_resolved_rdm.shape[0]//2
    sm = nspatorb if spin_major else None

    rdm = np.zeros((nspatorb,)*(rank*2))
    for inds in product(*((range(nspatorb*2),)*(2*rank))):
        if rdm_paired_spins(inds, sm):
            rdm[spats(inds, sm)] += spin_resolved_rdm[inds]
    return rdm

def make_spinfree_nord_rdm(nelec, norb, rank, get_expval, spin_major, time_reversal_sym, tol=1e-12):
    rdm = spin_resolved_to_spinfree(
            make_spin_resolved_rdm(norb, rank, get_expval, spin_major, time_reversal_sym, tol=tol), 
            spin_major)
    einsum_string = ''.join(tuple(string.ascii_lowercase[i]*2 for i in range(rank)))+'->'
    trace = np.einsum(einsum_string, rdm)
    for i in range(rank): trace/=(np.sum(nelec)-i)
    rdm/=trace
    return rdm


'''
partial tracing of NORD RDMs
'''
def one_from_two_rdm(two_rdm, nelec):
    # Last two indices refer to middle two second quantized operators in the 2RDM
    one_rdm = np.einsum('ikjj->ik', two_rdm)
    one_rdm /= (np.sum(nelec)-1)
    return one_rdm

def two_from_three_rdm(three_rdm, nelec):
    # Last two indices refer to middle two second quantized operators in the 3RDM
    two_rdm = np.einsum('ikjlpp->ikjl', three_rdm)
    two_rdm /= (np.sum(nelec)-2)
    return two_rdm

def three_from_four_rdm(four_rdm, nelec):
    # Last two indices refer to middle two second quantized operators in the 4RDM
    three_rdm = np.einsum('ikjlmnpp->ijklmn', four_rdm)
    three_rdm /= (np.sum(nelec)-3)
    return three_rdm

reorder_rdm12 = pyscf.fci.rdm.reorder_dm12
reorder_rdm123 = pyscf.fci.rdm.reorder_dm123
reorder_rdm1234 = pyscf.fci.rdm.reorder_dm1234

'''
"unreorder" methods do the opposite of the "reorder" methods in pyscf/fci/rdm.py,
that is to say, they convert normal-ordered RDMs to product-of-single-excitation RDMs.
'''
def unreorder_rdm12(rdm1, rdm2, inplace=True):
    nmo = rdm1.shape[0]
    if not inplace:
        rdm2 = rdm2.copy()
    for k in range(nmo):
        rdm2[:,k,k,:] += rdm1
    return rdm1, rdm2

def unreorder_rdm123(rdm1, rdm2, rdm3, inplace=True):
    if not inplace:
        rdm3 = rdm3.copy()
    norb = rdm1.shape[0]
    for q in range(norb):
        rdm3[:,q,q,:,:,:] += rdm2
        rdm3[:,:,:,q,q,:] += rdm2
        rdm3[:,q,:,:,q,:] += rdm2.transpose(0,2,3,1)
        for s in range(norb):
            rdm3[:,q,q,s,s,:] += rdm1
    rdm1, rdm2 = unreorder_rdm12(rdm1, rdm2, inplace)
    return rdm1, rdm2, rdm3

def unreorder_rdm1234(rdm1, rdm2, rdm3, rdm4, inplace=True):
    if not inplace:
        rdm4 = rdm4.copy()
    norb = rdm1.shape[0]
    for q in range(norb):
        rdm4[:,q,:,:,:,:,q,:] += rdm3.transpose(0,2,3,4,5,1)
        rdm4[:,:,:,q,:,:,q,:] += rdm3.transpose(0,1,2,4,5,3)
        rdm4[:,:,:,:,:,q,q,:] += rdm3
        rdm4[:,q,:,:,q,:,:,:] += rdm3.transpose(0,2,3,1,4,5)
        rdm4[:,:,:,q,q,:,:,:] += rdm3
        rdm4[:,q,q,:,:,:,:,:] += rdm3
        for s in range(norb):
            rdm4[:,q,q,s,:,:,s,:] += rdm2.transpose(0,2,3,1)
            rdm4[:,q,q,:,:,s,s,:] += rdm2
            rdm4[:,q,:,:,q,s,s,:] += rdm2.transpose(0,2,3,1)
            rdm4[:,q,:,s,q,:,s,:] += rdm2.transpose(0,2,1,3)
            rdm4[:,q,:,s,s,:,q,:] += rdm2.transpose(0,2,3,1)
            rdm4[:,:,:,s,s,q,q,:] += rdm2
            rdm4[:,q,q,s,s,:,:,:] += rdm2
            for u in range(norb):
                rdm4[:,q,q,s,s,u,u,:] += rdm1
    rdm1, rdm2, rdm3 = unreorder_rdm123(rdm1, rdm2, rdm3, inplace)
    return rdm1, rdm2, rdm3, rdm4


def spinfree_nord_to_pose(rdm, nelec):
    rank = len(rdm.shape)/2
    rdms = [rdm]
    if rank>3:
        rdms.insert(0, three_from_four_rdm(rdms[0], nelec))
    if rank>2:
        rdms.insert(0, two_from_three_rdm(rdms[0], nelec))
    if rank>1:
        rdms.insert(0, one_from_two_rdm(rdms[0], nelec))

    if rank==2:
        unreorder_rdm12(*rdms)
    elif rank==3:
        unreorder_rdm123(*rdms)
    elif rank==4:
        unreorder_rdm1234(*rdms)

    return rdms

def make_spinfree_pose_rdms(nelec, norb, rank, get_expval, spin_major, time_reversal_sym, tol=1e-12):
    return spinfree_nord_to_pose(
            make_spinfree_nord_rdm(nelec, norb, rank, get_expval, spin_major, time_reversal_sym, tol=1e-12), nelec)


def fact(n):
    out = 1.0;
    for i in range(1, n+1): out*=i
    return out

def ncomb(n, r):
    assert n >= r
    if r == 0 or n == 1 or r == n: return 1
    return fact(n)/(fact(r)*fact(n-r))

import h5py
from functools import reduce
def load_spin_resolved_rdm(fname, rank):
    archive = h5py.File(fname, 'r')['archive']
    nspinorb = int(archive['propagator']['nsite'][()])*2
    nelec = int(archive['propagator']['nelec'][()])
    nord_rdm = np.zeros((nspinorb,)*(rank*2))
    data = archive['rdms'][str(rank)*2+'00']
    assert rank*2 == data['indices'][:,:].shape[1]
    ndata = data['indices'][:,:].shape[0]
    trace = 0.0
    for idata in range(ndata):
        creinds = data['indices'][idata,:rank]
        anninds = data['indices'][idata,rank:]
        assert all(creinds==sorted(creinds))
        assert all(anninds==sorted(anninds))
        inds = interleave(creinds, anninds)
        nord_rdm[inds] = data['values'][idata]
        # enforce hermiticity:
        #inds = interleave(anninds, creinds)
        #nord_rdm[inds] = data['values'][idata]
        if all(creinds==anninds): trace+= data['values'][idata]
    nord_rdm *= ncomb(nelec, rank)/trace
    return nord_rdm

def load_spinfree_1rdm_from_m7(h5_fname):
    rdm1 = load_spin_resolved_rdm(h5_fname, 1)
    rdm1_restored = restore_perm_syms(rdm1, True, False)
    rdm1_sf = spin_resolved_to_spinfree(rdm1_restored, True)
    return rdm1_sf

def load_spinfree_1_2rdm_from_m7(h5_fname, nelec=None):
    if nelec is None:
        archive = h5py.File(fname, 'r')['archive']
        nelec = int(archive['propagator']['nelec'][()])
    rdm2 = load_spin_resolved_rdm(h5_fname, 2)
    rdm2_restored = restore_perm_syms(rdm2, True, False)
    rdm2_sf = spin_resolved_to_spinfree(rdm2_restored, True)
    rdm1_sf = one_from_two_rdm(rdm2_sf, nelec)
    return unreorder_rdm12(rdm1_sf, rdm2_sf, False)

def load_spinfree_ladder_rdm_from_m7(fname, cre):
    archive = h5py.File(fname, 'r')['archive']
    nsite = int(archive['propagator']['nsite'])
    norm = float(archive = h5py.File(fname, 'r')['archive']['rdm']['norm'][()])

    label = '1110' if cre else '1101'

    rdm = np.zeros((nspinorb,)*(rank*2))
    data = archive['rdms'][label]
    ndata = data['indices'][:,:].shape[0]
    for idata in range(ndata):
        imode, ispinorb, jspinorb = data['indices'][idata, :]
        assert (ispinorb < nsite) == (jspinorb < nsite), "Sz non-conservation is incompatible with spin averaging"
        isite = ispinorb%nsite
        jsite = jspinorb%nsite
        rdm[imode, isite, jsite] = data['values'][idata]
    rdm /= norm
    return rdm


if __name__=='__main__':
    '''
    testing the PySCF interface
    uses spin-minor ordering
    '''

    def pyscf_spin_resolved_element(civec, nelec, ncas, creinds, desinds):
        '''
        make_rdm methods should never call this method with spin-symmetry
        non-conserving orbital indices
        '''
        assert total_spin(creinds, None)==total_spin(desinds, None)
        ciket = civec.copy()
        fcre = (pyscf.fci.addons.cre_a, pyscf.fci.addons.cre_b)
        fdes = (pyscf.fci.addons.des_a, pyscf.fci.addons.des_b)
        count_elec = list(nelec)
        for desind in desinds:
            ciket = fdes[spin(desind, None)](ciket, ncas, count_elec, spat(desind, None))
            count_elec[spin(desind, None)]-=1
            if count_elec[spin(desind, None)]<0: return 0
        for creind in reversed(creinds):
            ciket = fcre[spin(creind, None)](ciket, ncas, count_elec, spat(creind, None))
            count_elec[spin(creind, None)]+=1
        assert tuple(count_elec)==nelec
        return float(np.dot(np.conj(civec.flatten()), ciket.flatten()))

    def pyscf_casci_spin_resolved_element(casci, creinds, desinds):
        return pyscf_spin_resolved_element(casci.ci, casci.nelecas, casci.ncas, creinds, desinds)

    def casci_example_N2_singlet():
        mol = pyscf.M(
            atom = 'N 0 0 0; N 0 0 1.2',
            basis = 'ccpvdz',
            spin = 0)
        myhf = mol.RHF().run()
        return myhf.CASCI(6, 6).run()

    def casci_example_O2_triplet():
        mol = pyscf.M(
            atom = 'O 0 0 0; O 0 0 1.2',
            basis = 'ccpvdz',
            spin = 2)
        myhf = mol.RHF().run()
        return myhf.CASCI(6, 8).run()


    def pyscf_rdm1234(casci):
        '''
        Spin-traced, product-of-single-excitation ordered RDMs of rank 1, 2, 3, and 4.
        '''
        return pyscf.fci.rdm.make_dm1234('FCI4pdm_kern_sf', casci.ci, casci.ci, casci.ncas, casci.nelecas)


    def test(casci, get_expval, max_rank, time_reversal_sym):
        constructed_rdms = make_spinfree_pose_rdms(casci.nelecas, casci.ncas, max_rank, get_expval, False, time_reversal_sym)
        pyscf_pose_rdms = pyscf_rdm1234(casci)
        '''
        alternatively, the pyscf RDMs can be brought to normal order for comparison
        pyscf_nord_rdms = pyscf.fci.rdm.reorder_dm1234(*pyscf_pose_rdms)
        '''
        for irdm, (constructed_rdm, pyscf_pose_rdm) in enumerate(zip(constructed_rdms, pyscf_pose_rdms)):
            assert np.allclose(constructed_rdm, pyscf_pose_rdm), 'RDMs of rank {} do not match.'.format(irdm+1)
    '''
    run tests
    '''
    max_rank = 3
    casci = casci_example_N2_singlet()
    def get_expval(creinds, desinds): return pyscf_casci_spin_resolved_element(casci, creinds, desinds)
    test(casci, get_expval, max_rank, False)
    print ("test passed: ms = 0 ignoring time reversal symmetry")
    test(casci, get_expval, max_rank, True)
    print ("test passed: ms = 0 exploiting time reversal symmetry")

    casci = casci_example_O2_triplet()
    def get_expval(creinds, desinds): return pyscf_casci_spin_resolved_element(casci, creinds, desinds)
    test(casci, get_expval, max_rank, False)
    print ("test passed: ms = 2")
