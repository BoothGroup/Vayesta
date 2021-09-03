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
    print(np.array(a).shape, np.array(b).shape)
    assert np.array(a).shape == np.array(b).shape
    
    result = []
    for i in range(len(a)):
        if (a[i] == b[i]):
            result.append(0.0)
        else:
            result.append(abs((a[i]-b[i])))
            
    return np.array(result)
    

nsite = 12
nimp = 4
doping = 0
boundary = ('PBC')
basis = 'STO-6G'

if doping:
    ne_target = (nsite + doping)/nsite * nimp
else:
    ne_target = None

def plot_sc(name, quantity, compare_quantity=None, compare_name=None, abs_errors=True, clabel=None):
    '''
    Plotter of self-consistency estimations
    '''
    
    def read_file(name, quantity):
        distance = []
        estimates = []
        
        filename = 'H'+'-'+quantity+'-'+name+'.txt'
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
                distance.append(float(segments[0]))
                estimates.append(float(segments[1]))
                
        return (np.array(distance), np.array(estimates))
    
    names = ['oneshot', 'pdmet', 'brueck', 'vdmet', 'fci', 'hf']
    colors = ['red', 'blue', 'orange', 'purple', 'green', 'black']
    labels = ['Oneshot', 'P-DMET', 'Brueckner', 'V-DMET', 'FCI', 'HF']
    symbols = ['.', '.', '.', '.', '.', '.']
    
    assert (name in names)
    assert (quantity == 'energies-dm' or quantity == 'docc' or 'energies-amp')

    if compare_quantity and compare_name:
        assert (compare_name in names)
        assert (compare_quantity == 'energies-dm' or compare_quantity == 'docc' or compare_quantity == 'energies-amp')

    distance, estimates = read_file(name, quantity)
        
    color = colors[names.index(name)]
    symbol = symbols[names.index(name)]
    label = labels[names.index(name)]
    
    if clabel:
        label = clabel
        
    if compare_quantity and compare_name:
        compare_estimates = read_file(compare_name, compare_quantity)[1]
        if (abs_errors):
            estimates = abs_error(compare_estimates, estimates)
        else:
            estimates = rel_error(compare_estimates, estimates)
        color = colors[names.index(compare_name)]
        symbol = symbols[names.index(compare_name)]
        label = labels[names.index(compare_name)]

    plt.plot(distance, estimates, symbol, color=color, label=label)
    plt.plot(distance, estimates, color=color)
        
        
plt.title('Symmetric Hydrogen ring – No. atoms: %1d   Basis: %s   No. frags: %1d\n Amplitude Projection' % (nsite, basis, nimp))
plot_sc('hf', 'energies-dm')
plot_sc('fci', 'energies-dm')
plot_sc('oneshot', 'energies-amp')
plot_sc('pdmet', 'energies-amp')
plot_sc('brueck', 'energies-amp')
plot_sc('vdmet', 'energies-dm', clabel='V-DMET (RDM)')

plt.ylabel('Energy per electron [Ha]')
plt.xlabel('Ring stretching [A]')
plt.grid()
plt.legend()
plt.savefig('Hydrogen_e_tot_amp.jpeg')
plt.close()

plt.title('Symmetric Hydrogen ring – No. atoms: %1d   Basis: %s   No. frags: %1d\n Amplitude Projection' % (nsite, basis, nimp))
plot_sc('fci', 'energies-dm', compare_name='fci', compare_quantity='energies-dm')
plot_sc('fci', 'energies-dm', compare_name='oneshot', compare_quantity='energies-amp')
plot_sc('fci', 'energies-dm', compare_name='pdmet', compare_quantity='energies-amp')
plot_sc('fci', 'energies-dm', compare_name='brueck', compare_quantity='energies-amp')
plot_sc('fci', 'energies-dm', compare_name='vdmet', compare_quantity='energies-dm', clabel='V-DMET (RDM)')

plt.ylabel('Energy difference per electron [Ha]')
plt.xlabel('Ring stretching [A]')
plt.grid()
plt.legend()
plt.savefig('Hydrogen_e_tot_amp_error.jpeg')
plt.close()


plt.title('Symmetric Hydrogen ring – No. atoms: %1d   Basis: %s   No. frags: %1d\n Reduced Density Matrix Projection' % (nsite, basis, nimp))
plot_sc('hf', 'energies-dm')
plot_sc('fci', 'energies-dm')
plot_sc('oneshot', 'energies-dm')
plot_sc('pdmet', 'energies-dm')
plot_sc('brueck', 'energies-dm')
plot_sc('vdmet', 'energies-dm')

plt.ylabel('Energy per electron [Ha]')
plt.xlabel('Ring stretching [A]')
plt.grid()
plt.legend()
plt.savefig('Hydrogen_e_tot_rdm.jpeg')
plt.close()

plt.title('Symmetric Hydrogen ring – No. atoms: %1d   Basis: %s   No. frags: %1d\n Reduced Density Matrix Projection' % (nsite, basis, nimp))
plot_sc('fci', 'energies-dm', compare_name='fci', compare_quantity='energies-dm')
plot_sc('fci', 'energies-dm', compare_name='oneshot', compare_quantity='energies-dm')
plot_sc('fci', 'energies-dm', compare_name='pdmet', compare_quantity='energies-dm')
plot_sc('fci', 'energies-dm', compare_name='brueck', compare_quantity='energies-dm')
plot_sc('fci', 'energies-dm', compare_name='vdmet', compare_quantity='energies-dm', clabel='V-DMET (RDM)')

plt.ylabel('Energy difference per electron [Ha]')
plt.xlabel('Ring stretching [A]')
plt.grid()
plt.legend()
plt.savefig('Hydrogen_e_tot_rdm_error.jpeg')
plt.close()


plt.title('Symmetric Hydrogen ring – No. atoms: %1d   Basis: %s   No. frags: %1d' % (nsite, basis, nimp))
#plot_sc('hf', 'docc')
plot_sc('fci', 'docc')
plot_sc('oneshot', 'docc')
plot_sc('pdmet', 'docc')
plot_sc('brueck', 'docc')
plot_sc('vdmet', 'docc')

plt.ylabel('Electron double occupancy')
plt.xlabel('Ring stretching [A]')
plt.grid()
plt.legend()
plt.savefig('Hydrogen_docc_rdm.jpeg')
plt.close()

plt.title('Symmetric Hydrogen ring – No. atoms: %1d   Basis: %s   No. frags: %1d' % (nsite, basis, nimp))
plot_sc('fci', 'docc', compare_name='fci', compare_quantity='docc')
plot_sc('fci', 'docc', compare_name='oneshot', compare_quantity='docc')
plot_sc('fci', 'docc', compare_name='pdmet', compare_quantity='docc')
plot_sc('fci', 'docc', compare_name='brueck', compare_quantity='docc')
plot_sc('fci', 'docc', compare_name='vdmet', compare_quantity='docc')

plt.ylabel('Electron double occupancy error')
plt.xlabel('Ring stretching [A]')
plt.grid()
plt.legend()
plt.savefig('Hydrogen_docc_rdm_error.jpeg')
plt.close()
