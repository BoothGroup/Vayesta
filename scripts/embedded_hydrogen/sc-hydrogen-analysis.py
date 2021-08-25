import numpy as np
from numpy import einsum
import vayesta
import vayesta.ewf
import vayesta.lattmod
import pyscf.tools
import pyscf.tools.ring
from functools import reduce
import numpy
from pyscf import gto, scf, ao2mo, fci

from scmf import SelfConsistentMF

import numpy as np
from scipy.interpolate import interp1d

import vayesta
import vayesta.ewf
import vayesta.lattmod


from scmf import SelfConsistentMF

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
params = {
   'axes.labelsize': 40,
   'font.size': 40,
   'legend.fontsize': 40,
   'lines.linewidth' : 2,
   'lines.markersize' : 23,
   'xtick.labelsize': 40,
   'ytick.labelsize': 40,
   'figure.figsize': [40, 20]
   }
plt.rcParams.update(params)

def rel_error(a, b):
    '''
    Returns abs relative error (as a difference) for two arrays
    where b[i] values estimate a[i] reference values
    '''
    assert np.array(a).shape == np.array(b).shape
    
    result = []
    for i in range(len(a)):
        if (a[i] == b[i]):
            result.append(0.0)
        elif (b[i] == 0.0):
            result.append(0.0)
        else:
            result.append(abs((a[i]-b[i])))
            
    return result

nsite = 12
nimp = 2

R_range = []

e_tot_hf = []
e_tot_fci = []

e_tot_ewf_oneshot = []
e_tot_ewf_pdmet = []
e_tot_ewf_brueck = []
e_tot_ewf_dmet = []

with open('H-energies-ewf.txt', 'r') as f:
    iteration = 0
    for line in f.read().splitlines():
        segments = line.split()
        if (line.split()[0] == '#'):
            if iteration == 1:
                # Read initial conditions
                print(line.split())
                char, nsite, nimp = (line.split())
                #print(nsite, nimp, doping)
            iteration += 1
            continue
        R_range.append(float(line.split()[0]))
        e_tot_hf.append(float(line.split()[1]))
        e_tot_fci.append(float(line.split()[2]))
        e_tot_ewf_oneshot.append(float(line.split()[3]))
        e_tot_ewf_pdmet.append(float(line.split()[4]))
        e_tot_ewf_brueck.append(float(line.split()[5]))
        e_tot_ewf_dmet.append(float(line.split()[6]))

        iteration += 1
    
plt.title(str(nsite)+' nsite '+str(nimp)+' nimp Hydrogen ring\nAmplitude projection')

plt.plot(R_range, e_tot_hf, '.', color='black', label='HF')
plt.plot(R_range, e_tot_hf, color='black')


plt.plot(R_range, e_tot_fci, '.', color='green', label='FCI')
plt.plot(R_range, e_tot_fci, color='green')


plt.plot(R_range, e_tot_ewf_oneshot, '.', color='blue', label='Oneshot')
plt.plot(R_range, e_tot_ewf_oneshot, color='blue')

#plt.plot(R_range, e_tot_ewf_pdmet, '.', color='red', label='P-DMET')
#plt.plot(R_range, e_tot_ewf_pdmet, color='red')

#plt.plot(R_range, e_tot_ewf_brueck, '.', color='orange', label='Brueckner')
#plt.plot(R_range, e_tot_ewf_brueck, color='orange')

plt.plot(R_range, e_tot_ewf_dmet, '.', color='purple', label='DMET-RDM proj.')
plt.plot(R_range, e_tot_ewf_dmet, color='purple')

plt.legend()
plt.grid()

plt.xlabel('Bond length [A]')
plt.ylabel('Energy per electron [Hartree]')

plt.savefig('Hydrogen_e_tot_ewf.jpeg')
plt.close()

plt.title(str(nsite)+' nsite '+str(nimp)+' nimp Hydrogen ring\nAmplitude projection')

#plt.plot(R_range, rel_error(e_tot_fci, e_tot_hf), '.', color='black', label='HF')
#plt.plot(R_range, rel_error(e_tot_fci, e_tot_hf), color='black')


plt.plot(R_range, rel_error(e_tot_fci, e_tot_fci), '.', color='green', label='FCI')
plt.plot(R_range, rel_error(e_tot_fci, e_tot_fci), color='green')


plt.plot(R_range, rel_error(e_tot_fci, e_tot_ewf_oneshot), '.', color='blue', label='Oneshot')
plt.plot(R_range, rel_error(e_tot_fci, e_tot_ewf_oneshot), color='blue')

#plt.plot(R_range, rel_error(e_tot_fci, e_tot_ewf_pdmet), '.', color='red', label='P-DMET')
#plt.plot(R_range, rel_error(e_tot_fci, e_tot_ewf_pdmet), color='red')

#plt.plot(R_range, rel_error(e_tot_fci, e_tot_ewf_brueck), '.', color='orange', label='Brueckner')
#plt.plot(R_range, rel_error(e_tot_fci, e_tot_ewf_brueck), color='orange')

plt.plot(R_range, rel_error(e_tot_fci, e_tot_ewf_dmet), '.', color='purple', label='DMET-RDM proj.')
plt.plot(R_range, rel_error(e_tot_fci, e_tot_ewf_dmet), color='purple')

plt.semilogy()
plt.legend()
plt.grid()

plt.xlabel('Bond length [A]')
plt.ylabel('Energy per electron FCI residual [Hartrees]')

plt.savefig('Hydrogen_e_tot_ewf_err.jpeg')
plt.close()

R_range = []
e_tot_hf = []
e_tot_fci = []

e_tot_rdm_oneshot = []
e_tot_rdm_pdmet = []
e_tot_rdm_brueck = []
e_tot_rdm_dmet = []

with open('H-energies-rdm.txt', 'r') as f:
    iteration = 0
    for line in f.read().splitlines():
        segments = line.split()
        if (line.split()[0] == '#'):
            if iteration == 1:
                # Read initial conditions
                print(line.split())
                char, nsite, nimp = (line.split())
                #print(nsite, nimp, doping)
            iteration += 1
            continue
        R_range.append(float(line.split()[0]))
        e_tot_hf.append(float(line.split()[1]))
        e_tot_fci.append(float(line.split()[2]))
        e_tot_rdm_oneshot.append(float(line.split()[3]))
        e_tot_rdm_pdmet.append(float(line.split()[4]))
        e_tot_rdm_brueck.append(float(line.split()[5]))
        e_tot_rdm_dmet.append(float(line.split()[6]))

        iteration += 1
    
plt.title(str(nsite)+' nsite '+str(nimp)+' nimp Hydrogen ring\nReduced density matrix projection')

plt.plot(R_range, e_tot_hf, '.', color='black', label='HF')
plt.plot(R_range, e_tot_hf, color='black')


plt.plot(R_range, e_tot_fci, '.', color='green', label='FCI')
plt.plot(R_range, e_tot_fci, color='green')


plt.plot(R_range, e_tot_rdm_oneshot, '.', color='blue', label='Oneshot')
plt.plot(R_range, e_tot_rdm_oneshot, color='blue')

#plt.plot(R_range, e_tot_rdm_pdmet, '.', color='red', label='P-DMET')
#plt.plot(R_range, e_tot_rdm_pdmet, color='red')

#plt.plot(R_range, e_tot_rdm_brueck, '.', color='orange', label='Brueckner')
#plt.plot(R_range, e_tot_rdm_brueck, color='orange')

plt.plot(R_range, e_tot_rdm_dmet, '.', color='purple', label='DMET')
plt.plot(R_range, e_tot_rdm_dmet, color='purple')

plt.legend()
plt.grid()

plt.xlabel('Bond length [A]')
plt.ylabel('Energy per electron [Hartree]')

plt.savefig('Hydrogen_e_tot_rdm.jpeg')
plt.close()

plt.title(str(nsite)+' nsite '+str(nimp)+' nimp Hydrogen ring\nReduced density matrix projection')

#plt.plot(R_range, rel_error(e_tot_fci, e_tot_hf), '.', color='black', label='HF')
#plt.plot(R_range, rel_error(e_tot_fci, e_tot_hf), color='black')


plt.plot(R_range, rel_error(e_tot_fci, e_tot_fci), '.', color='green', label='FCI')
plt.plot(R_range, rel_error(e_tot_fci, e_tot_fci), color='green')


plt.plot(R_range, rel_error(e_tot_fci, e_tot_rdm_oneshot), '.', color='blue', label='Oneshot')
plt.plot(R_range, rel_error(e_tot_fci, e_tot_rdm_oneshot), color='blue')

#plt.plot(R_range, rel_error(e_tot_fci, e_tot_rdm_pdmet), '.', color='red', label='P-DMET')
#plt.plot(R_range, rel_error(e_tot_fci, e_tot_rdm_pdmet), color='red')

#plt.plot(R_range, rel_error(e_tot_fci, e_tot_rdm_brueck), '.', color='orange', label='Brueckner')
#plt.plot(R_range, rel_error(e_tot_fci, e_tot_rdm_brueck), color='orange')

plt.plot(R_range, rel_error(e_tot_fci, e_tot_rdm_dmet), '.', color='purple', label='DMET')
plt.plot(R_range, rel_error(e_tot_fci, e_tot_rdm_dmet), color='purple')

plt.semilogy()
plt.legend()
plt.grid()

plt.xlabel('Bond length [A]')
plt.ylabel('Energy per electron FCI residual [Hartrees]')

plt.savefig('Hydrogen_e_tot_rdm_err.jpeg')
plt.close()
