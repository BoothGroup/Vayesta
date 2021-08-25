import numpy as np
from scipy.interpolate import interp1d

import vayesta
import vayesta.ewf
import vayesta.lattmod
from bethe import hubbard1d_bethe_energy
from bethe import hubbard1d_bethe_docc

from scmf import SelfConsistentMF

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


def read_exact(file):
    '''
    Read exact energy for Hubbard model versus U/t for comparison
    '''
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

def rel_error(a, b):
    '''
    Returns relative error (as a fraction) for two arrays
    where b[i] values estimate a[i] reference values
    '''
    assert np.array(a).shape == np.array(b).shape
    
    result = []
    for i in range(len(a)):
        if (a[i] == b[i]):
            result.append(0.0)
        elif (b[i] == 0.0):
            results.append(0.0)
        else:
            result.append(100*abs((a[i]-b[i])/a[i]))
            
    return result
def abs_error(a, b):
    '''
    Returns absolute error (as a difference) for two arrays
    where b[i] values estimate a[i] reference values
    '''
    assert np.array(a).shape == np.array(b).shape
    
    result = []
    for i in range(len(a)):
        if (a[i] == b[i]):
            result.append(0.0)
        else:
            result.append(abs((a[i]-b[i])))
            
    return result
    

dimension='1D'
nsite = 52
nimp = 2
doping = 0

u_min = 0
u_max = 0
u_step = 0
do_fci = (nsite <= 12)

if doping:
    ne_target = (nsite + doping)/nsite * nimp
else:
    ne_target = None

mo_pdmet = None
mo_brueck = None

# All enegies should be energy per site
hubbard_u_range = []

e_tot_fci = []
e_tot_dmrg = []

e_tot_rdm_oneshot = []
e_tot_rdm_pdmet = []
e_tot_rdm_brueck = []
e_tot_rdm_dmet = []

e_tot_ewf_oneshot = []
e_tot_ewf_pdmet = []
e_tot_ewf_brueck = []
e_tot_ewf_dmet= []

docc_fci = []
docc_dmrg = []

docc_oneshot = []
docc_pdmet = []
docc_brueck = []
docc_dmet = []


with open('%s-energies-dm.txt'%dimension, 'r') as f:
    iteration = 0.0
    for line in f.read().splitlines():
        segments = line.split()
        if (line.split()[0] == '#'):
            if iteration == 1:
                # Read initial conditions
                print(line.split())
                char, nsite, nimp, doping = (line.split())
                #print(nsite, nimp, doping)
            continue
        hubbard_u_range.append(float(line.split()[0]))
        e_tot_fci.append(float(line.split()[1]))
        e_tot_rdm_oneshot.append(float(line.split()[2]))
        e_tot_rdm_pdmet.append(float(line.split()[3]))
        e_tot_rdm_brueck.append(float(line.split()[4]))
        e_tot_rdm_dmet.append(float(line.split()[5]))

        iteration += 1
        
with open('%s-energies-ewf.txt' % dimension, 'r') as f:
    iteration = 0
    for line in f.read().splitlines():
        segments = line.split()
        if (line.split()[0] == '#'):
            if iteration == 1:
                # Read initial conditions
                print(line.split())
                char, nsite, nimp, doping = (line.split())
                #print(nsite, nimp, doping)
            continue
        #hubbard_u_range.append(float(line.split()[0]))
        #e_tot_fci.append(float(line.split()[1]))
        e_tot_ewf_oneshot.append(float(line.split()[2]))
        e_tot_ewf_pdmet.append(float(line.split()[3]))
        e_tot_ewf_brueck.append(float(line.split()[4]))
        e_tot_ewf_dmet.append(float(line.split()[5]))

        iteration += 1

with open('%s-docc.txt' % dimension, 'r') as f:
    iteration = 0.0
    for line in f.read().splitlines():
        segments = line.split()
        if (line.split()[0] == '#'):
            if iteration == 1:
                # Read initial conditions
                print(line.split())
                char, nsiter, nimpr, dopingr = (line.split())
                #print(nsite, nimp, doping)
                
                nsite = nsiter
                nimp = nimpr
                doping = dopingr
            continue
        #hubbard_u_range.append(float(line.split()[0]))
        docc_fci.append(float(line.split()[1]))
        docc_oneshot.append(float(line.split()[2]))
        docc_pdmet.append(float(line.split()[3]))
        docc_brueck.append(float(line.split()[4]))
        docc_dmet.append(float(line.split()[5]))

        iteration += 1

exact_name='FCI'
# If FCI is not available, plot Bethe ansatz/other reference result instead
if (not do_fci and dimension=='1D'):
    u_exact, e_exact = read_exact('hubbard1d-bethe.txt')
    docc_exact = np.diff(e_exact)
    f_exact = interp1d(u_exact, e_exact)
    # Differentiate exact energy curve to get double occupancy
    print(len(u_exact[1:]), len(docc_exact))
    f2_exact = interp1d(u_exact[:len(u_exact)-1], docc_exact)
    
    exact_name='Bethe'
    e_tot_fci = []
    docc_fci = []
    # Get Vayesta Bethe solver
    for u in hubbard_u_range:
        e_tot_fci.append(hubbard1d_bethe_energy(t=1,u=u))
        docc_fci.append(hubbard1d_bethe_docc(t=1,u=u))
    # DMRG Reference from George
    e_tot_dmrg = np.array([-45.894825416274,\
                          -37.510899307165,\
                          -30.438353288162,\
                          -24.863384762387,\
                          -20.672172039924,\
                          -17.528748714811,\
                          -15.135977789180,\
                          -13.276842306328,\
                          -11.801762604846,\
                          -10.608485054073,\
                          -9.626325526879,\
                          -8.805530676375,\
                          -8.110373624703])/36
                          
    docc_dmrg = np.array([2.49999659582594e-01,\
                           2.15444045779449e-01,\
                           1.76274794040956e-01,\
                           1.34178104840918e-01,\
                           1.00318502463466e-01,\
                           7.57028480567305e-02,\
                           5.82251478793901e-02,\
                           4.57428140124752e-02,\
                           3.66721894202989e-02,\
                           2.99434383478908e-02,\
                           2.48480214387685e-02,\
                           2.09148048017384e-02,\
                           1.78247139874706e-02])
                           
    e_dmrg = interp1d(np.linspace(0, 12, 13), e_tot_dmrg)
    d_dmrg = interp1d(np.linspace(0, 12, 13), docc_dmrg)
    
    e_tot_dmrg = e_dmrg(hubbard_u_range)
    docc_dmrg = d_dmrg(hubbard_u_range)
    
if (not do_fci and dimension=='2D' and doping==0):
    
    u_exact = [0.0, 2.0, 4.0, 6.0, 8.0, 12.0]
    e_tot_exact = [e_tot_ewf_oneshot[0], -1.176,-0.8605, -0.6565,-0.5241,-0.3689]
    docc_exact = [0.25, 0.188, 0.126, 0.0809,0.0539,0.0278]
    
    f_exact = interp1d(u_exact, e_tot_exact)
    f2_exact = interp1d(u_exact, docc_exact)
    exact_name='DMRG'
   
    e_tot_fci = f_exact(np.array(hubbard_u_range))
    docc_fci = f2_exact(np.array(hubbard_u_range))
    
    e_tot_dmrg = e_tot_fci
    docc_dmrg = docc_fci
    
if (not do_fci and dimension=='2D' and doping<0):
    u_exact = [0.0, 2.0, 4.0, 6.0, 8.0, 12.0]
    e_tot_exact = [e_tot_ewf_oneshot[0], -1.3062,-1.108, -0.977 ,-0.88, 0.0]
    #docc_exact = [0.25, 0.188, 0.126, 0.0809,0.0539,0.0278]
    
    f_exact = interp1d(u_exact, e_tot_exact)
    #f2_exact = interp1d(u_exact, docc_exact)
    exact_name='DMET-Ref'
   
    e_tot_fci = f_exact(np.array(hubbard_u_range))
    docc_fci = None
    
    #e_tot_dmrg = e_tot_fci
   # docc_dmrg = docc_fci


# Set range for unconverged DMET resultsa
N = len(hubbard_u_range)

# EWF Energy comparison
plt.title('%s Hubbard %16d sites %16d frag_size %16d doping \n Amplitude projection' % (dimension, nsite, nimp, doping))
plt.ylabel('Total energy per electron [t]')
plt.xlabel('Hubbard U/t')
plt.grid()


plt.plot(hubbard_u_range, e_tot_fci, color='green', label=exact_name, linewidth=3.0)
'''
if not do_fci:
    plt.plot(hubbard_u_range, e_tot_dmrg, color='black', label='DMRG', linewidth=3.0)
'''
plt.plot(hubbard_u_range, e_tot_ewf_oneshot, '*', color='blue', label='Oneshot')
plt.plot(hubbard_u_range, e_tot_ewf_oneshot, color='blue')

plt.plot(hubbard_u_range, e_tot_ewf_pdmet, 'x', color='red', label='P-DMET')
plt.plot(hubbard_u_range, e_tot_ewf_pdmet, color='red')

plt.plot(hubbard_u_range, e_tot_ewf_brueck, '.', color='orange', label='Brueckner')
plt.plot(hubbard_u_range, e_tot_ewf_brueck, color='orange')

#plt.plot(hubbard_u_range, e_tot_ewf_dmet, 'o', color='purple', label='DMET', markersize=18)
#plt.plot(hubbard_u_range, e_tot_ewf_dmet, color='purple')

plt.legend()
plt.savefig('%s-Hubbard_e_tot_ewf.jpeg' % dimension)
plt.close()

# EWF Energy error comparison

plt.title('%s Hubbard %16d sites %16d frag_size %16d doping \n Amplitude projection' % (dimension, nsite, nimp, doping))
plt.ylabel('Relative energy error [%]')
plt.xlabel('Hubbard U/t')
plt.grid()


plt.plot(hubbard_u_range, rel_error(e_tot_fci, e_tot_fci), color='green', label=exact_name, linewidth=3.0)
'''
if not do_fci:
    plt.plot(hubbard_u_range, rel_error(e_tot_fci, e_tot_dmrg), color='black', label='DMRG', linewidth=3.0)
'''
plt.plot(hubbard_u_range, rel_error(e_tot_fci, e_tot_ewf_oneshot), '*', color='blue', label='Oneshot')
plt.plot(hubbard_u_range, rel_error(e_tot_fci, e_tot_ewf_oneshot), color='blue')

plt.plot(hubbard_u_range, rel_error(e_tot_fci, e_tot_ewf_pdmet), 'x', color='red', label='P-DMET')
plt.plot(hubbard_u_range, rel_error(e_tot_fci, e_tot_ewf_pdmet), color='red')

plt.plot(hubbard_u_range, rel_error(e_tot_fci, e_tot_ewf_brueck), '.', color='orange', label='Brueckner')
plt.plot(hubbard_u_range, rel_error(e_tot_fci, e_tot_ewf_brueck), color='orange')

plt.plot(hubbard_u_range[:N],  rel_error(e_tot_fci[:N], e_tot_ewf_dmet[:N]), 'o', color='purple', label='DMET', markersize=18)
plt.plot(hubbard_u_range[:N],  rel_error(e_tot_fci[:N], e_tot_ewf_dmet[:N]), color='purple')

plt.legend()
plt.savefig('%s-Hubbard_e_err_ewf.jpeg' % dimension)
plt.close()

# RDM energy comparison

plt.title('%s Hubbard %16d sites %16d frag_size %16d doping \n Reduced density matrix projection' % (dimension, nsite, nimp, doping))

plt.ylabel('Total energy per electron [t]')
plt.xlabel('Hubbard U/t')
plt.grid()


plt.plot(hubbard_u_range, e_tot_fci, color='green', label=exact_name, linewidth=3.0)
'''
if not do_fci:
    plt.plot(hubbard_u_range, e_tot_dmrg, color='black', label='DMRG', linewidth=3.0)
'''
plt.plot(hubbard_u_range, e_tot_rdm_oneshot, '*', color='blue', label='Oneshot')
plt.plot(hubbard_u_range, e_tot_rdm_oneshot, color='blue')

plt.plot(hubbard_u_range, e_tot_rdm_pdmet, 'x', color='red', label='P-DMET')
plt.plot(hubbard_u_range, e_tot_rdm_pdmet, color='red')

plt.plot(hubbard_u_range, e_tot_rdm_brueck, '.', color='orange', label='Brueckner')
plt.plot(hubbard_u_range, e_tot_rdm_brueck, color='orange')

plt.plot(hubbard_u_range[:N], e_tot_rdm_dmet[:N], 'o', color='purple', label='DMET', markersize=18)
plt.plot(hubbard_u_range[:N], e_tot_rdm_dmet[:N], color='purple')


plt.legend()
plt.savefig('%s-Hubbard_e_tot_rdm.jpeg' % dimension)
plt.close()
# RDM energy relative error

plt.title('%s Hubbard %16d sites %16d frag_size %16d doping \n Reduced density matrix projection' % (dimension, nsite, nimp, doping))
plt.ylabel('Relative energy error [%]')
plt.xlabel('Hubbard U/t')
plt.grid()


plt.plot(hubbard_u_range, rel_error(e_tot_fci, e_tot_fci), color='green', label=exact_name, linewidth=3.0)
'''
if not do_fci:
    plt.plot(hubbard_u_range, rel_error(e_tot_fci, e_tot_dmrg), color='black', label='DMRG', linewidth=3.0)
'''

plt.plot(hubbard_u_range, rel_error(e_tot_fci, e_tot_rdm_oneshot), '*', color='blue', label='Oneshot')
plt.plot(hubbard_u_range, rel_error(e_tot_fci, e_tot_rdm_oneshot), color='blue')

plt.plot(hubbard_u_range, rel_error(e_tot_fci, e_tot_rdm_pdmet), 'x', color='red', label='P-DMET')
plt.plot(hubbard_u_range, rel_error(e_tot_fci, e_tot_rdm_pdmet), color='red')

plt.plot(hubbard_u_range, rel_error(e_tot_fci, e_tot_rdm_brueck), '.', color='orange', label='Brueckner')
plt.plot(hubbard_u_range, rel_error(e_tot_fci, e_tot_rdm_brueck), color='orange')

plt.plot(hubbard_u_range[:N],  rel_error(e_tot_fci[:N], e_tot_rdm_dmet[:N]), 'o', color='purple', label='DMET', markersize=18)
plt.plot(hubbard_u_range[:N],  rel_error(e_tot_fci[:N], e_tot_rdm_dmet[:N]), color='purple')


plt.legend()
plt.savefig('%s-Hubbard_e_err_rdm.jpeg' % dimension)
plt.close()

plt.title('%s Hubbard %16d sites %16d frag_size %16d doping \n Reduced density matrix projection' % (dimension, nsite, nimp, doping))
plt.ylabel('Absolute energy error [t]')
plt.xlabel('Hubbard U/t')
plt.grid()

plt.plot(hubbard_u_range, rel_error(e_tot_fci, e_tot_fci), color='green', label=exact_name, linewidth=3.0)
'''
if not do_fci:
    plt.plot(hubbard_u_range, rel_error(e_tot_fci, e_tot_dmrg), color='black', label='DMRG', linewidth=3.0)
'''

plt.plot(hubbard_u_range, abs_error(e_tot_fci, e_tot_rdm_oneshot), '*', color='blue', label='Oneshot')
plt.plot(hubbard_u_range, abs_error(e_tot_fci, e_tot_rdm_oneshot), color='blue')

plt.plot(hubbard_u_range, abs_error(e_tot_fci, e_tot_rdm_pdmet), 'x', color='red', label='P-DMET')
plt.plot(hubbard_u_range, abs_error(e_tot_fci, e_tot_rdm_pdmet), color='red')

plt.plot(hubbard_u_range, abs_error(e_tot_fci, e_tot_rdm_brueck), '.', color='orange', label='Brueckner')
plt.plot(hubbard_u_range, abs_error(e_tot_fci, e_tot_rdm_brueck), color='orange')

plt.plot(hubbard_u_range[:N],  abs_error(e_tot_fci[:N], e_tot_rdm_dmet[:N]), 'o', color='purple', label='DMET', markersize=18)
plt.plot(hubbard_u_range[:N],  abs_error(e_tot_fci[:N], e_tot_rdm_dmet[:N]), color='purple')


plt.legend()
plt.savefig('%s-Hubbard_e_abs_err_rdm.jpeg' % dimension)
plt.close()

# Docc
plt.title('%s Hubbard %16d sites %16d frag_size %16d doping' % (dimension, nsite, nimp, doping))
plt.ylabel(r'Double occupancy $\left <n_{i\sigma}n_{i(-\sigma)} \right>$')
plt.xlabel('Hubbard U/t')
plt.grid()


plt.plot(hubbard_u_range, docc_fci, color='green', label=exact_name, linewidth=3.0)
if not do_fci:
    plt.plot(hubbard_u_range, docc_dmrg, color='black', label='DMRG', linewidth=3.0)

plt.plot(hubbard_u_range, docc_oneshot, '*', color='blue', label='Oneshot')
plt.plot(hubbard_u_range, docc_oneshot, color='blue')

plt.plot(hubbard_u_range, docc_pdmet, 'x', color='red', label='P-DMET')
plt.plot(hubbard_u_range, docc_pdmet, color='red')

plt.plot(hubbard_u_range, docc_brueck, '.', color='orange', label='Brueckner')
plt.plot(hubbard_u_range, docc_brueck, color='orange')

plt.plot(hubbard_u_range[:N], docc_dmet[:N], 'o', color='purple', label='DMET', markersize=18)
plt.plot(hubbard_u_range[:N], docc_dmet[:N], color='purple')

plt.legend()
plt.savefig('%s-Hubbard_docc.jpeg' % dimension)
plt.close()

# Docc error

plt.title('%s Hubbard %16d sites %16d frag_size %16d doping' % (dimension, nsite, nimp, doping))
plt.ylabel(r'Double occupancy error [%]')
plt.xlabel('Hubbard U/t')
plt.grid()


plt.plot(hubbard_u_range, rel_error(docc_fci, docc_fci), color='green', label=exact_name, linewidth=3.0)
if not do_fci:
    plt.plot(hubbard_u_range, rel_error(docc_fci, docc_dmrg), color='black', label='DMRG', linewidth=3.0)
plt.plot(hubbard_u_range, rel_error(docc_fci, docc_oneshot), '*', color='blue', label='Oneshot')
plt.plot(hubbard_u_range, rel_error(docc_fci, docc_oneshot), color='blue')

plt.plot(hubbard_u_range, rel_error(docc_fci, docc_pdmet), 'x', color='red', label='P-DMET')
plt.plot(hubbard_u_range, rel_error(docc_fci, docc_pdmet), color='red')

plt.plot(hubbard_u_range, rel_error(docc_fci, docc_brueck), '.', color='orange', label='Brueckner')
plt.plot(hubbard_u_range, rel_error(docc_fci, docc_brueck), color='orange')


plt.plot(hubbard_u_range[:N], rel_error(docc_fci[:N], docc_dmet[:N]),'o',markersize=18, color='purple', label='DMET')
plt.plot(hubbard_u_range[:N], rel_error(docc_fci[:N], docc_dmet[:N]), color='purple')

plt.legend()
plt.savefig('%s-Hubbard_docc_err.jpeg' % dimension)
plt.close()

plt.title('%s Hubbard %16d sites %16d frag_size %16d doping' % (dimension, nsite, nimp, doping))
plt.ylabel(r'Absolute double occupancy error')
plt.xlabel('Hubbard U/t')
plt.grid()


plt.plot(hubbard_u_range, abs_error(docc_fci, docc_fci), color='green', label=exact_name, linewidth=3.0)
if not do_fci:
    plt.plot(hubbard_u_range, abs_error(docc_fci, docc_dmrg), color='black', label='DMRG', linewidth=3.0)
plt.plot(hubbard_u_range, abs_error(docc_fci, docc_oneshot), '*', color='blue', label='Oneshot')
plt.plot(hubbard_u_range, abs_error(docc_fci, docc_oneshot), color='blue')

plt.plot(hubbard_u_range, abs_error(docc_fci, docc_pdmet), 'x', color='red', label='P-DMET')
plt.plot(hubbard_u_range, abs_error(docc_fci, docc_pdmet), color='red')

plt.plot(hubbard_u_range, abs_error(docc_fci, docc_brueck), '.', color='orange', label='Brueckner')
plt.plot(hubbard_u_range, abs_error(docc_fci, docc_brueck), color='orange')


plt.plot(hubbard_u_range[:N], abs_error(docc_fci[:N], docc_dmet[:N]),'o',markersize=18, color='purple', label='DMET')
plt.plot(hubbard_u_range[:N], abs_error(docc_fci[:N], docc_dmet[:N]), color='purple')

plt.legend()
plt.savefig('%s-Hubbard_docc_err_abs.jpeg' % dimension)
plt.close()
