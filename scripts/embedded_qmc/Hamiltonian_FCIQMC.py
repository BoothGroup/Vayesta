import numpy as np

import vayesta
import vayesta.ewf
import vayesta.lattmod
import pickle as pkl
import matplotlib.pyplot as plt

from pyscf import gto, scf, ao2mo
from pyscf import cc
from pyscf import fci
from scipy.interpolate import interp1d

import vayesta

import vayesta.lattmod
import vayesta.ewf

from pyscf import tools, ao2mo


params = {
   'axes.labelsize': 40,
   'font.size': 40,
   'legend.fontsize': 40,
   'lines.linewidth' : 5,
   'lines.markersize' : 15,
   'xtick.labelsize': 40,
   'ytick.labelsize': 40,
   'figure.figsize': [40, 15]
   }
plt.rcParams.update(params)


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
        return fci.direct_spin1.kernel(self.h1e, self.eri, nsite, self.nelec, verbose=6, ecore=self.h0)[0]
    
def get_energy(mf, frags):
    # DMET energy
    e = np.asarray([get_e_dmet(mf, f) for f in frags])
    with open('Fragment_energies.txt', 'a') as f:
        f.write(str(mf.mol.atom)+'\n'+str(e)+'\n')
    print(e)
    #assert abs(e.min()-e.max()) < 1e-6
    e = np.sum(e)
    return e

def get_e_dmet(mf,f):
    """DMET (projected DM) energy of fragment f."""
    c = f.c_active
    p = f.get_fragment_projector(c)
    #print(c.shape)
    #print(p.shape)
    #print(f.results.dm1.shape)
    #print(f.results.dm2.shape)
    #print(mf._eri.shape)
    
    n = mf.mol.nelectron
    eri = ao2mo.addons.restore('s1', mf._eri, n).reshape(n, n, n, n)
    print(eri.shape)
    pdm1 = np.einsum('ix,xj,ai,bj->ab', p, f.results.dm1, c, c, optimize=True)
    pdm2 = np.einsum('ix,xjkl,ai,bj,ck,dl->abcd', p, f.results.dm2, c, c, c, c, optimize=True)
    e1 = np.einsum('ij,ij->', mf.get_hcore(), pdm1, optimize=True)
    e2 = np.einsum('ijkl,ijkl->', eri, pdm2,optimize=True)/2
    return e1 + e2

def write_fcidump(h0, h1e, eri, nelec, fname='FCIDUMP'):
    '''
    writes the provided integrals to a file in a standard format for FCI programs
    '''
    nsite = h1e.shape[0]
    if len(eri.shape)!=1:
        # ERIs must have 8-fold symmetry restored
        eri = ao2mo.restore(8, eri, nsite)
    tools.fcidump.from_integrals(fname, h1e, eri, nsite, nelec, h0, 0, [1,]*nsite)


def construct_FCIQMC_Hamiltonian(nsite, nelectron, nimp, hubbard_u, mean_field, fragment, out_filename='M7_Hamiltonian.txt'):
    '''
    For a mean-field determinant and a given fragment construct fragment basis Hamiltonians
    '''
    mf = mean_field
    mf.kernel()
    
    # Get site basis singles/doubles Hamiltonaian
    h0 = mf.e_tot # Hartree-Fock energy
    h1 = mf.get_hcore()
    h2 = mf._eri  
    
    print(h0.shape)
    print(h1.shape)
    print(h2.shape)
    
    
    # Transform into cluster basis for the given fragment
    c = fragment.c_active
    print(c.shape)
    h1 = np.einsum('ij,ia,jb->ab', h1, c, c)
    h2 = np.einsum('ijkl,ia,jb,kc,ld->abcd', h2, c, c, c, c)/2
    
    # Write result in file
    with open(out_filename, 'a') as f:
        f.write('%f %f %f %f \n'% (nsite, nelectron, nimp, hubbard_u))
        f.write('%f \n' % h0)
        f.write(str(h1)+'\n')
        f.write(str(h2)+'\n')
        
    return h0, h1, h2

# Input parameters

nsite = 10
nelectron = 10
hubbard_u = 2.0
nimp = 2 # Fragment size
 
filling = nelectron//nsite # Electron filling
nimps = nsite // nimp     # No. fragments


# Create Hubbard Hamiltonian using Vayesta
mol = vayesta.lattmod.Hubbard1D(nsite, nelectron=nelectron, hubbard_u=hubbard_u, output='pyscf.out', verbose=10)
mol.build()

# Run mean field calculation
mf = vayesta.lattmod.LatticeMF(mol)
mf.kernel()

#Create embedding solver
oneshot = vayesta.ewf.EWF(mf, solver='FCI', bno_threshold=np.inf, fragment_type='Site', make_rdm1=True, make_rdm2=True)

for site in range(0, nsite, nimp):
    f = oneshot.make_atom_fragment(list(range(site, site+nimp)))
    f.kernel()

# Ensure FCI convergence
assert len(oneshot.fragments) == nsite//nimp
for f in oneshot.fragments:
    assert f.results.converged

# Obtain reference embedding energies
e_tot_amp_ewf = oneshot.e_tot / nelectron
e_tot_rdm_ewf = get_energy(mf, oneshot.fragments) / nelectron
    
    
# Store projected Hamiltonians for each fragment as an array
h0s = []
h1s = []
eris = []

# Loop through fragments and write out Hamiltonians to FCIQMC readable format
for i in range(nsite//nimp):
    h0, h1, eri = construct_FCIQMC_Hamiltonian(nsite, nelectron, nimp, hubbard_u, mf, oneshot.fragments[i])
    h0s.append(h0)
    h1s.append(h1)
    eris.append(eri)

    Hubbard_Hamiltonian = Hamiltonian()
    Hubbard_Hamiltonian.from_arrays(h0, h1, eri, int(2*nimp*filling))
    Hubbard_Hamiltonian.to_pickle('Hubbard_Hamiltonian%d.pkl'%i)

    write_fcidump(h0, h1, eri, int(2*nimp*filling), fname='FCIDUMP_frag%d'%i)

# 1. Run FCIQMC for each fragment Hamiltonian and collect FCIQMC outputs -- Darren

# 2. Read in FCIQMC output c0, c1, c2 + RDMs for each fragment -- Darren

# 3. Calculate amplitude/rdm energy projection from FCIQMC for each fragment -- Marcell
#   Basis transformation from fragment basis to site basis
#   3a. Assert that each fragment energy matches
#   3b. Combine fragment energies

# 4. Comparison with FCI-embedded energies -- Marcell
