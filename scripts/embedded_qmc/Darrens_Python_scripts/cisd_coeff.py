from pyscf import ao2mo, tools, fci
import numpy
from pyscf.ci.cisd import tn_addrs_signs
import h5py
import pickle as pkl
from copy import deepcopy

class Hamiltonian:
    h0 = None
    h1e = None
    eri = None
    nelec = None
    
    def from_arrays(self, h0, h1e, eri, nelec):
        self.h0, self.h1e, self.eri, self.nelec = h0, h1e, eri, nelec

    def to_pickle(self, fname):
        with open(fname, 'wb') as f: pkl.dump([self.h0, self.h1e, self.eri, self.nelec], f)

    def from_pickle(self, fname):
        with open(fname, 'rb') as f:
            self.h0, self.h1e, self.eri, self.nelec = pkl.load(f)

    def write_fcidump(self, fname='FCIDUMP'):
        '''
        writes the provided integrals to a file in a standard format for FCI programs
        '''
        nsite = self.h1e.shape[0]
        if len(self.eri.shape)!=1:
            # ERIs must have 8-fold symmetry restored
            eri = ao2mo.restore(8, self.eri, nsite)
        else: eri = self.eri
        tools.fcidump.from_integrals(fname, self.h1e, eri, nsite, self.nelec, self.h0, 0, [1,]*nsite)
    
    def get_fci_energy(self):
        nsite = self.h1e.shape[0]
        return fci.direct_spin1.kernel(self.h1e, self.eri, nsite, self.nelec, verbose=6, ecore=self.ham.h0)[0]

'''
Symmetry-restored CISD coefficients
'''
class RestoredCisdCoeffs:
    c0 = None
    c1a = None
    c1b = None
    c2aa = None
    c2bb = None
    c2ab = None
    ham = None
    fock = None
    
    def __init__(self, ham):
        self.ham = deepcopy(ham)
        self.nmo = ham.h1e.shape[0]
        self.nelec = ham.nelec
        self.nocc = ham.nelec//2
        self.t1addrs, self.t1signs = tn_addrs_signs(self.nmo, self.nocc, 1)
        self.t2addrs, self.t2signs = tn_addrs_signs(self.nmo, self.nocc, 2)
        self.npair_occ = self.nocc*(self.nocc-1)//2
        self.nvrt = self.nmo-self.nocc
        self.npair_vrt = self.nvrt*(self.nvrt-1)//2
        o, v = slice(0, self.nocc), slice(self.nocc, self.nmo)
        if len(ham.eri.shape)!=4:
            self.ham.eri = ao2mo.restore(1, ham.eri, self.nmo)
        else: self.ham.eri = ham.eri.copy()
        self.fock = self.ham.h1e.copy()
        # add the Hartree-Fock effective potential to the core H to obtain the Fock matrix in the MO basis
        # F = h + J - K
        self.fock += 2*numpy.einsum('pqkk->pq', self.ham.eri[:,:,o,o])-numpy.einsum('pkkq->pq', self.ham.eri[:,o,o,:])

    def is_setup(self):
        if self.ham is None: return False
        if self.c0 is None: return False
        if self.c1a is None: return False
        if self.c1b is None: return False
        if self.c2aa is None: return False
        if self.c2bb is None: return False
        if self.c2ab is None: return False
        return True

    def from_fci_coeffs(self, c0, c1a, c1b, c2aa, c2bb, c2ab):
        # Physical vacuum ref to HF ref
        c1a *= self.t1signs
        c1b *= self.t1signs
        c2aa *= self.t2signs
        c2bb *= self.t2signs
        c2ab = numpy.einsum('ij,i,j->ij', c2ab, self.t1signs, self.t1signs)
        
        shape1 = (self.nocc, self.nvrt)
        shape2 = (self.nocc, self.nvrt, self.nocc, self.nvrt)
        self.c0 = c0
        assert len(c1a)==self.nocc*self.nvrt
        self.c1a = c1a.reshape(shape1)
        assert len(c1b)==self.nocc*self.nvrt
        self.c1b = c1b.reshape(shape1)
        assert len(c2aa)==self.npair_occ*self.npair_vrt
        assert len(c2bb)==self.npair_occ*self.npair_vrt
        assert len(c2ab[0])==self.nocc*self.nvrt
        assert len(c2ab[1])==self.nocc*self.nvrt
        self.c2ab = c2ab.reshape(shape2)
        '''
        restore antisymmetric coefficients
        '''
        self.c2aa = numpy.zeros(shape2)
        self.c2bb = numpy.zeros(shape2)
        iflat = 0
        for j in range(self.nocc):
            for i in range(j):
                for b in range(self.nvrt):
                    for a in range(b):
                        self.c2aa[i,a,j,b] = c2aa[iflat]
                        self.c2aa[j,a,i,b] = -c2aa[iflat]
                        self.c2bb[i,a,j,b] = c2bb[iflat]
                        self.c2bb[j,a,i,b] = -c2bb[iflat]
                        iflat+=1

    def from_fcivec(self, fcivec):
        c1a = fcivec[self.t1addrs, 0]
        c1b = fcivec[0, self.t1addrs]
        c2aa = fcivec[self.t2addrs, 0]
        c2bb = fcivec[0, self.t2addrs]
        c2ab = fcivec[self.t1addrs[:,None], self.t1addrs]

        self.from_fci_coeffs(fcivec[0,0], c1a, c1b, c2aa, c2bb, c2ab)

    def from_m7(self, fname):
        f = h5py.File(fname, 'r')['archive']
        
        assert self.nmo == f['propagator/nsite'][()]
        assert self.nelec == f['propagator/nelec'][()]
        
        f = f['ref_excits']
        
        # brackets at the end are to retrive the scalar
        c0 = f['0000'][()]  
        
        # spin-major orbital indices 
        c1_indx = f['1100/indices'][:]
        #c1_indx1 = f['1/spin_orbs_ann'][:]
        #c1_indx2 = f['1/spin_orbs_cre'][:]        

        # amplitudes
        c1_amp = f['1100/values'][:]
        
        # spin-major orbital indices 
        c2_indx = f['2200/indices'][:]
        #c2_indx1 = f['2/spin_orbs_ann'][:]
        #c2_indx2 = f['2/spin_orbs_cre'][:]

        # amplitudes
        c2_amp = f['2200/values'][:]
        
        shape1 = (self.nocc, self.nvrt)
        
        c1a = numpy.zeros(shape1)
        c1b = numpy.zeros(shape1)
        
        for i in range(len(c1_indx)):
            # splitting into the alpha and beta channels
            if c1_indx[i][1] >= self.nmo: 
                assert (c1_indx[i][0] >= self.nmo), "Non-conservative spin system?"
                # beta channel conversion to spatial orbital indices
                c1_indx[i][1] -= self.nmo
                c1_indx[i][0] -= self.nmo
                # Physical vac to HF ref conversion
                c1_indx[i][0] -= self.nocc
                # indx1 is ann so it goes to occ since this is HF reference
                c1b[c1_indx[i][1], c1_indx[i][0]-self.nocc] = c1_amp[i][0]
            else:
                assert (c1_indx[i][0] < self.nmo), "Non-conservative spin system?"
                # alpha channel
                # Physical vac to HF ref conversion
                c1_indx[i][0] -= self.nocc
                # indx1 is ann so it goes to occ since this is HF reference
                c1a[c1_indx[i][1], c1_indx[i][0]-self.nocc] = c1_amp[i][0]

        '''     
        for i in range(len(c1_indx1)):
            # splitting into the alpha and beta channels
            if c1_indx1[i][0] >= self.nmo: 
                assert (c1_indx2[i][0] >= self.nmo), "Non-conservative spin system?"
                # beta channel conversion to spatial orbital indices
                c1_indx1[i][0] -= self.nmo
                c1_indx2[i][0] -= self.nmo
                # Physical vac to HF ref conversion
                c1_indx2[i][0] -= self.nocc
                # indx1 is ann so it goes to occ since this is HF reference
                c1b[c1_indx1[i][0], c1_indx2[i][0]-self.nocc] = c1_amp[i][0]
            else:
                assert (c1_indx2[i][0] < self.nmo), "Non-conservative spin system?"
                # alpha channel
                # Physical vac to HF ref conversion
                c1_indx2[i][0] -= self.nocc
                # indx1 is ann so it goes to occ since this is HF reference
                c1a[c1_indx1[i][0], c1_indx2[i][0]-self.nocc] = c1_amp[i][0]
        '''
        c1a = numpy.ndarray.flatten(c1a)
        c1b = numpy.ndarray.flatten(c1b)
        
        shape2 = (self.nocc, self.nvrt, self.nocc, self.nvrt)
        c2aa = numpy.zeros(shape2)
        c2bb = numpy.zeros(shape2)
        c2ab = numpy.zeros(shape2)
        
        for i in range(len(c2_indx)):
            # splitting into aa, bb and ab channels
            channel = 0
            # left b
            if c2_indx[i][2] >= self.nmo:
                assert (c2_indx[i][0] >= self.nmo), "Non-conservative spin system?"
                # beta channel conversion to spatial orbital indices
                c2_indx[i][2] -= self.nmo
                c2_indx[i][0] -= self.nmo
                channel += 1
            # right b
            if c2_indx[i][3] >= self.nmo:
                assert (c2_indx[i][1] >= self.nmo), "Non-conservative spin system?"
                # beta channel conversion to spatial orbital indices
                c2_indx[i][3] -= self.nmo
                c2_indx[i][1] -= self.nmo
                channel += 1
            
            # Physical vac to HF ref conversion
            c2_indx[i][0] -= self.nocc
            c2_indx[i][1] -= self.nocc
            
            if channel == 0:
                # aa
                c2aa[c2_indx[i][2], c2_indx[i][0], c2_indx[i][3], \
                          c2_indx[i][1]] = c2_amp[i][0]
            elif channel == 1:
                # ab 
                c2ab[c2_indx[i][2], c2_indx[i][0], c2_indx[i][3], \
                          c2_indx[i][1]] = c2_amp[i][0]
            else:
                # bb
                c2bb[c2_indx[i][2], c2_indx[i][0], c2_indx[i][3], \
                          c2_indx[i][1]] = c2_amp[i][0]
        '''
        for i in range(len(c2_indx1)):
            # splitting into aa, bb and ab channels
            channel = 0
            # left b
            if c2_indx1[i][0] >= self.nmo:
                assert (c2_indx2[i][0] >= self.nmo), "Non-conservative spin system?"
                # beta channel conversion to spatial orbital indices
                c2_indx1[i][0] -= self.nmo
                c2_indx2[i][0] -= self.nmo
                channel += 1
            # right b
            if c2_indx1[i][1] >= self.nmo:
                assert (c2_indx2[i][1] >= self.nmo), "Non-conservative spin system?"
                # beta channel conversion to spatial orbital indices
                c2_indx1[i][1] -= self.nmo
                c2_indx2[i][1] -= self.nmo
                channel += 1
            
            # Physical vac to HF ref conversion
            c2_indx2[i][0] -= self.nocc
            c2_indx2[i][1] -= self.nocc
            
            if channel == 0:
                # aa
                c2aa[c2_indx1[i][0], c2_indx2[i][0], c2_indx1[i][1], \
                          c2_indx2[i][1]] = c2_amp[i][0]
            elif channel == 1:
                # ab 
                c2ab[c2_indx1[i][0], c2_indx2[i][0], c2_indx1[i][1], \
                          c2_indx2[i][1]] = c2_amp[i][0]
            else:
                # bb
                c2bb[c2_indx1[i][0], c2_indx2[i][0], c2_indx1[i][1], \
                          c2_indx2[i][1]] = c2_amp[i][0]
        '''
        c2aa_flat = []
        c2bb_flat = []
        
        for j in range(self.nocc):
            for i in range(j):
                for b in range(self.nvrt):
                    for a in range(b):
                        c2aa_flat.append(c2aa[i,a,j,b])
                        c2bb_flat.append(c2bb[i,a,j,b])
        
        c2aa = numpy.array(c2aa_flat)
        c2bb = numpy.array(c2bb_flat)
        
        c2ab = c2ab.reshape((self.nocc*self.nvrt, self.nocc*self.nvrt))
                
        self.from_fci_coeffs(c0, c1a, c1b, c2aa, c2bb, c2ab)


    def ref_energy(self):
        o = slice(0, self.nocc)
        e = self.ham.h0
        e += numpy.einsum('ii->', self.fock[o,o])*2
        '''
        subtract 0.5 * (J - K) to account for double-counting
        J = 4 * eris due to sum over spin channel combinations aaaa bbbb abab baba
        K = 2 * eris due to sum over spin channel combinations aaaa bbbb
        then factor of 0.5 gives the following correct factors
        '''
        e -= numpy.einsum('iijj->', self.ham.eri[o,o,o,o])*2
        e += numpy.einsum('ijji->', self.ham.eri[o,o,o,o])
        return e

    def energy(self):
        assert self.is_setup(), "need coeffs to be defined, call one of the from_ methods first"
        o, v = slice(0, self.nocc), slice(self.nocc, self.nmo)
        fock_ov = self.fock[o,v]
        eri_ovov = self.ham.eri[o,v,o,v]
        eri_oovv = self.ham.eri[o,o,v,v]
        
        e = self.ref_energy() * self.c0
        e+=numpy.einsum('ia,ia', self.c1a, fock_ov)
        e+=numpy.einsum('ia,ia', self.c1b, fock_ov)

        e+=numpy.einsum('iajb,iajb', self.c2aa, eri_ovov)
        e-=numpy.einsum('iajb,ijab', self.c2aa, eri_oovv)

        e+=numpy.einsum('iajb,iajb', self.c2bb, eri_ovov)
        e-=numpy.einsum('iajb,ijab', self.c2bb, eri_oovv)

        e+=numpy.einsum('iajb,iajb', self.c2ab, eri_ovov)

        return e/self.c0
    
    def normalise(self):
        self.c1a /= self.c0
        self.c1b /= self.c0
        self.c2aa /= self.c0
        self.c2bb /= self.c0
        self.c2ab /= self.c0
        self.c0 /= self.c0
        
    
    def flatten(self):
        '''
        returns c0, c1 and c2 coefficients combined in a 1d array
        '''
        assert self.is_setup(), "need coeffs to be defined, call one of the from_ methods first"
        c1_flat = numpy.ndarray.flatten(numpy.hstack((self.c1a, self.c1b)))
        c2_flat = numpy.ndarray.flatten(numpy.hstack((self.c2aa, self.c2bb, \
                                                      self.c2ab)))
        
        return numpy.hstack((c1_flat, c2_flat))
    
    def get_fci(self):
        '''
        compute the FCI energy and wavefunction exactly
        '''
        nsite = self.ham.h1e.shape[0]
        return fci.direct_spin1.kernel(self.ham.h1e, self.ham.eri, nsite, \
                                       self.ham.nelec, verbose=6, ecore=self.ham.h0)

    def to_pickle(self, fname):
        with open(fname, 'wb') as f: pkl.dump([self.c0, self.c1a, self.c1b, self.c2aa, self.c2bb, self.c2ab], f)

    def from_pickle(self, fname):
        with open(fname, 'rb') as f:
            self.c0, self.c1a, self.c1b, self.c2aa, self.c2bb, self.c2ab = pkl.load(f)

def compare_two_cisd(cisd1, cisd2):
    '''
    returns the absolute energy difference and mean absolute difference per
    coefficient element
    '''
    return abs(cisd1.energy() - cisd2.energy()), \
            1- numpy.dot(cisd1.flatten(), cisd2.flatten())/\
            numpy.sqrt(numpy.dot(cisd1.flatten(), cisd1.flatten())*numpy.dot(cisd2.flatten(), cisd2.flatten()))
            
            


