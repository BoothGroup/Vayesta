#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import numpy

import matplotlib.pyplot as plt
from pyscf import gto, scf, ao2mo
from pyscf import cc
from pyscf import fci
from scipy.interpolate import interp1d
'''
Customizing Hamiltonian for SCF module.

Three steps to define Hamiltonian for SCF:
1. Specify the number of electrons. (Note mole object must be "built" before doing this step)
2. Overwrite three attributes of scf object
    .get_hcore
    .get_ovlp
    ._eri
3. Specify initial guess (to overwrite the default atomic density initial guess)

Note you will see warning message on the screen:

        Overwritten attributes  get_ovlp get_hcore  of <class 'pyscf.scf.hf.RHF'>

'''

def read_exact(file):
    with open(file, 'r') as file_object:
        U = []
        E_tot = []
        for line in file_object:
            parts = line.split()
            U.append(float(parts[0]))
            E_tot.append(float(parts[1]))
    print(U)
    print(E_tot)
    return U, E_tot
    

mol = gto.M()
n = 102
mol.nelectron = n
mol.output = '1D_Hubbard.out'
mol.verbose = 10
mol.build()

resolution = 25
# Hubbard strengths

# Plot total exact total energy per electron for comparison
U_ex, E_tot_exact = read_exact('n102_hubbard_exact.txt')
U_bethe, E_tot_bethe = read_exact('hubbard1d-bethe.txt')
f = interp1d(U_ex, E_tot_exact)
f_bethe = interp1d(U_bethe, E_tot_bethe)

print(U_ex)
t = 1.0
U = 4.0
U_range = numpy.linspace(0.0, (U_ex)[0], resolution)



MF_energies = []
FCI_corr_energies = []
CCSD_corr_energies = []
MP2_corr_energies = []
CCSD_tot_energies = []
MP2_tot_energies = []


GS_weight = []



for iteration in range(0, resolution):

    mf = scf.RHF(mol)
    h1 = numpy.zeros((n,n))

    
    # Tight binding Hamiltonian with diagonal and near diagonal elements

    for i in range(n-1):
        h1[i,i+1] = h1[i+1,i] = -1.0*t
    h1[n-1,0] = h1[0,n-1] = -1.0*t  # PBC
    #h1[n-1,0] = h1[0,n-1] = 1.0*t  # APBC
    # ONsite interaction ('Electron repulsion interaction')
    eri = numpy.zeros((n,n,n,n))
    for i in range(n):
        eri[i,i,i,i] = U_range[iteration]
        
    mf.get_hcore = lambda *args: h1
    mf.get_ovlp = lambda *args: numpy.eye(n)
    # ao2mo.restore(8, eri, n) to get 8-fold permutation symmetry of the integrals
    # ._eri only supports the two-electron integrals in 4-fold or 8-fold symmetry.
    mf._eri = ao2mo.restore(8, eri, n)
    
    mf_e = mf.kernel()
    MF_energies.append(mf_e)
    
    mf.run()
    mp2_ecorr = mf.MP2().run().e_corr
    mp2_etot = mf.MP2().run().e_tot
    MP2_corr_energies.append(mp2_ecorr)
    MP2_tot_energies.append(mp2_etot)
    print('MP2 correlation energy ', mp2_ecorr)
    print('MP2 total energy ', mp2_etot)
    # If you need to run post-HF calculations based on the customized Hamiltonian,
    # setting incore_anyway=True to ensure the customized Hamiltonian (the _eri
    # attribute) to be used.  Without this parameter, some post-HF method
    # (particularly in the MO integral transformation) may ignore the customized
    # Hamiltonian if memory is not enough.
    '''
    mycc = cc.CCSD(mf)
    cc.diis = False
    cc.iterative_damping = 0.1
    cc.max_cycle = 300
    mycc.kernel()
    print('CCSD total energy ', mycc.e_tot)
    print('CCSD correlation energy ', mycc.e_corr)
    
    CCSD_corr_energies.append(mycc.e_corr)
    CCSD_tot_energies.append(mycc.e_tot)
    
    cisolver = fci.FCI(mf)
    fci_energy, c0 = cisolver.kernel()
    print(fci_energy)
    print('FCI Energy: E(FCI) = %.12f' % fci_energy)
    print('Mean Field Energy E(MF) = %.12f' % mf.kernel())
    #print('FICE Energy: E_corr(FCI) = %.12f' % fci_solution.e_corr)
    
    FCI_corr_energies.append(fci_energy - mf_e)
    GS_weight.append(abs(c0[0, 0]))

    mol.incore_anyway = True
    '''
    print()
    print('ITERATION FINISHED')
    print('U/t = ', U_range[iteration])
    print()
        
    
    
print('Energy iterations')
print(MF_energies)
print(CCSD_corr_energies)
print(FCI_corr_energies)

print('U range')
print(U_range)

print('CCSD Total energies')
print(CCSD_tot_energies)

print('MP2 Total energies')
print(MP2_tot_energies)

# Plotting

params = {
   'axes.labelsize': 40,
   'font.size': 40,
   'legend.fontsize': 40,
   'xtick.labelsize': 40,
   'ytick.labelsize': 40,
   'figure.figsize': [40, 15]
   }
plt.rcParams.update(params)

plt.title(str(n)+' electron 1D Hubbard model ' )
#plt.plot(U_range, numpy.array(CCSD_corr_energies)/n, 'red', label='CCSD')
#plt.plot(U_range, numpy.array(FCI_corr_energies)/n, 'blue', label='FCI')
plt.plot(U_range, numpy.array(MP2_corr_energies)/n, 'green', label='MP2')

plt.xlabel('U/t')
plt.ylabel('Correlation energy per electron [t]')
plt.legend()
plt.grid()
plt.savefig('1D_Hubbard_E_corr.jpeg')
plt.close()



plt.title(str(n)+' electron 1D Hubbard model')
#plt.plot(U_range, numpy.array(CCSD_tot_energies)/n, 'red', label='CCSD')
plt.plot(U_range, numpy.array(MP2_tot_energies)/n, 'orange', label='MP2')
plt.plot(U_range, f_bethe(U_range), 'blue', label='Bethe Ref')
#plt.plot(U_range, (f(U_range)), 'green', label='Reference')


plt.xlabel('U/t')
plt.ylabel('Total energy per electron [t]')
plt.legend()
plt.grid()
plt.savefig('1D_Hubbard_E_tot.jpeg')
plt.close()
'''
plt.title('10 electron 1D Hubbard model/Mean-field weight')
plt.plot(U_range, GS_weight, 'blue', label='FCI')
plt.xlabel('U/t')
plt.ylabel('Weight of MF determinannt')
plt.legend()
plt.grid()
plt.savefig('1D_Hubbard_Ground_State.jpeg')
'''
