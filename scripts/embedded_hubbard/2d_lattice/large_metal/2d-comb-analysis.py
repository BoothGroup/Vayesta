import numpy as np
from scipy.interpolate import interp1d

import vayesta
import vayesta.ewf
import vayesta.lattmod

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
    

dimension='2D'
nsite = 400
nimp = 4
doping = 0
boundary = ('PBC', 'APBC')

u_min = 0
u_max = 12
u_step = 1
do_fci = (nsite <= 12)

if doping:
    ne_target = (nsite + doping)/nsite * nimp
else:
    ne_target = None

def plot_sc(name, quantity, abs_error=False, rel_error=False):
    '''
    Plotter of self-consistency estimations
    '''
    names = ['oneshot', 'pdmet', 'brueck', 'vdmet', 'dmrg']
    colors = ['red', 'blue', 'orange', 'purple', 'green']
    labels = ['Oneshot', 'P-DMET', 'Brueckner', 'V-DMET', 'DMRG']
    symbols = ['.', '.', '.', '.', '.' ]
    
    assert (name in names)
    assert (quantity == 'energies-dm' or quantity == 'docc')
    
    hubbard_u = []
    estimates = []

    
    if name == 'DMRG':
        #"Manually input reference DMRG results"
        u_exact = [0.0, 2.0, 4.0, 6.0, 8.0, 12.0]
        e_tot_exact = [0.0, -1.176,-0.8605, -0.6565,-0.5241,-0.3689]
        docc_exact = [0.25, 0.188, 0.126, 0.0809,0.0539,0.0278]
        
        f_exact = interp1d(u_exact, e_tot_exact)
        f2_exact = interp1d(u_exact, docc_exact)
        hubbard_u = np.array(u_exact)
        
        if quantity == 'energies-dm':
            estimates = np.array(f_exact(np.array(hubbard_u_range)))
        else:
            estimates = np.array(f2_exact(np.array(hubbard_u_range)))
        
    else:
        filename = dimension+'-'+quantity+'-'+name+'.txt'
        with open(filename, 'r') as f:
            iteration = 0.0
            for line in f.read().splitlines():
                segments = line.split()
                print(segments)
                if (line.split()[0] == '#'):
                    if iteration == 1:
                        # Read initial conditions
                        print(segments)
                        char, nsiter, nimpr, dopingr, boundary_cond = (segments)
                        #print(nsite, nimp, doping)
                        
                        assert nsiter == str(nsite)
                        assert nimpr == str(nimp)
                        assert dopingr == str(doping)
                        assert boundary_cond == str(boundary)
                            
                    continue
                hubbard_u.append(float(segments[0]))
                estimates.append(float(segments[1]))
            
        hubbard_u = np.array(hubbard_u)
        estimates = np.array(estimates)
        
        color = colors[names.index(name)]
        symbol = symbols[names.index(name)]
        label = labels[names.index(name)]
        if abs_error:
            estimates = abs_error(estimates)
            
        if rel_error:
            estimates = rel_error(estimates)
        
        plt.plot(hubbard_u, estimates, symbol, color=color, label=label)
        plt.plot(hubbard_u, estimates, color=color)
        
        
plt.title('%s Hubbard %16d sites %16d frag_size %16d doping %16s boundary \n Reduced density matrix projection' % (dimension, nsite, nimp, doping, boundary))
plot_sc('oneshot', 'energies-dm')
plot_sc('pdmet', 'energies-dm')
plot_sc('brueck', 'energies-dm')
plot_sc('vdmet', 'energies-dm')

plt.ylabel('Energy per electron [t]')
plt.xlabel('Hubbard U/t')
plt.grid()
plt.legend()
plt.savefig('%s-Hubbard_e_tot_rdm.jpeg' % dimension)
plt.close()


plt.title('%s Hubbard %16d sites %16d frag_size %16d doping %16s boundary \n Reduced density matrix projection' % (dimension, nsite, nimp, doping, boundary))

plot_sc('oneshot', 'docc')
plot_sc('pdmet', 'docc')
plot_sc('brueck', 'docc')
plot_sc('vdmet', 'docc')

plt.ylabel(r'Double occupancy $\left <n_{i\sigma}n_{i(-\sigma)} \right>$')
plt.xlabel('Hubbard U/t')
plt.grid()
plt.legend()
plt.savefig('%s-Hubbard_docc_rdm.jpeg' % dimension)
plt.close()
