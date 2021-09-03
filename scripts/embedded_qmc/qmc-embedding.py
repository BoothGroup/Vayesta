import numpy as np

import vayesta
import vayesta.ewf
import vayesta.lattmod
import pickle as pkl
import matplotlib.pyplot as plt

from pyscf import gto, scf, ao2mo, fci
from pyscf import cc
from pyscf import fci
from scipy.interpolate import interp1d


import vayesta

import vayesta.lattmod
import vayesta.ewf

from pyscf import tools, ao2mo

import vayesta.solver
import vayesta.solver.solver_qmc as embqmc

#import vayesta.solver_qmc
'''
Simple interface to use M7 FCIQMC for a Vayesta embedding solver for Hubbard model
'''
# Hamiltonian class for dealing with FCIQMC format files

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

# General helper functions to perform DMET estimation
# for Vayesta embedding calculations
def get_energy(mf, frags):
    # DMET energy estimation
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

    n = mf.mol.nelectron
    eri = ao2mo.addons.restore('s1', mf._eri, n).reshape(n, n, n, n)
    print(eri.shape)
    pdm1 = np.einsum('ix,xj,ai,bj->ab', p, f.results.dm1, c, c, optimize=True)
    pdm2 = np.einsum('ix,xjkl,ai,bj,ck,dl->abcd', p, f.results.dm2, c, c, c, c, optimize=True)
    e1 = np.einsum('ij,ij->', mf.get_hcore(), pdm1, optimize=True)
    e2 = np.einsum('ijkl,ijkl->', eri, pdm2,optimize=True)/2
    return e1 + e2


# Initial parameters for 1D Hubbard ring

nsite = 10
nimp = 2
doping = 0
nelectron = nsite
hubbard_u = 2.0

filling = nelectron//nsite
nfrags = nsite // nimp

# Initialise Hubbard Hamiltonian using Vayesta
mol = vayesta.lattmod.Hubbard1D(nsite, nelectron=nelectron, hubbard_u=hubbard_u, output='pyscf.out', verbose=10)
mol.build()

# Run mean field calculation
mf = vayesta.lattmod.LatticeMF(mol)
mf.kernel()


#print(mf.get_hcore().shape, mf.get_hcore())
#print(mf._eri.shape, mf._eri)

#Create embedding solver (no self-consistency implemented)
oneshot = vayesta.ewf.EWF(mf, solver='FCI', bno_threshold=np.inf, fragment_type='Site', make_rdm1=True, make_rdm2=True)

# Carry out chain fragmentation and solve fragment + bath cluster for each
# ! Active space projectors only after invoking fragment.kernel()
# ! Issue for large fragments
for site in range(0, nsite, nimp):
    f = oneshot.make_atom_fragment(list(range(site, site+nimp)))
    f.kernel()
    
# Obtain reference EWF embedding energies
e_tot_amp_ewf = oneshot.get_e_tot() / nelectron
e_tot_rdm_ewf = get_energy(mf, oneshot.fragments) / nelectron

# Run FCI reference calculation using Vayesta (treat system as one
# fragment with no bath orbitals)

fci_solver = vayesta.ewf.EWF(mf, solver='FCI', make_rdm1=True, make_rdm2=True, bath_type=None, fragment_type='Site')
f = fci_solver.make_atom_fragment(list(range(nsite)))
fci_solver.kernel()
e_tot_fci_1 = fci_solver.e_tot / nelectron

# Run FCI reference calculation using PySCF
e_tot_fci_2 = \
fci.direct_spin1.kernel(h1e=mf.get_hcore(), eri=mf.mol.get_eri(), nelec=nelectron,norb=nelectron, verbose=6)[0] / nelectron

# Prepare FCIQMC Hamiltonians for each fragment:

out_filename='M7_Hamiltonian'


qmc_solver = vayesta.ewf.EWF(mf, solver="FCIQMC", bno_threshold=np.inf, fragment_type='Site', make_rdm1=True, make_rdm2=True)

frag_index = 0.0
for site in range(0, nsite, nimp):
    f_qmc = qmc_solver.make_atom_fragment(list(range(site, site+nimp)))
    f_qmc.kernel() # Automatically creates FCIDUMP/PKL for FCIQMC
    #fragment_H.to_pickle('Hubbard_Hamiltonian%d.pkl'%frag_index)
    #fragment_H.write_fcidump( fname='FCIDUMP_frag%d'%frag_index)
    

    frag_index += 1

# Here the code assumes M7 has solved the above Hamiltonians outside of
# script and returned FCI amplitudes in "init_fname" .pkl files
# Perform energy calculation for each fragment


# Combine fragment energies
qmc_amp_energy = (qmc_solver.get_e_tot()) /nelectron
qmc_rdm_energy = (get_energy(mf, qmc_solver.fragments))/nelectron
# Comparison of energy per electron
print('Energy per e- comparison')
print('------------------------')
print('QMC energy [t] %1.12f (Vayesta + FCIQMC / Amplitude)'% qmc_amp_energy)
print('EWF energy [t] %1.12f (Vayesta + FCI / Amplitude)'% e_tot_amp_ewf)
print()
print('QMC energy [t] %1.12f (Vayesta + FCIQMC / Redduced DM)'% qmc_rdm_energy)
print('RDM energy [t] %1.12f (Vayesta + FCI / Reduced DM)'% e_tot_rdm_ewf)
print()
print('FCI energy [t] %1.12f (Vayesta/EWF) '% e_tot_fci_1)
print('FCI energy [t] %1.12f (PySCF/direct_spin1) '% e_tot_fci_2)
    
