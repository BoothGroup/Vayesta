import numpy as np

import vayesta
import vayesta.ewf
import vayesta.lattmod

import numpy as np

import matplotlib.pyplot as plt

from pyscf import gto, scf, ao2mo
from pyscf import cc
from pyscf import fci
from scipy.interpolate import interp1d

import vayesta

import vayesta.lattmod
import vayesta.ewf

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


def rel_error(a, b):
        '''
        Return relative error of some estimate "a" of quantity "b" scaled to reference value "b"
        '''
        if (a == b):
            return 0.0
        elif (b != 0.0):
            return (a-b)/b
        else:
            return 0.0


def run_cpt(nsite = 10,
            nelectron = 6,
            nimp = 1,
            hubbard_u = 12.0):
            
    # From example 61

    filling = nelectron/nsite
    mol = vayesta.lattmod.Hubbard1D(nsite, nelectron=nelectron, hubbard_u=hubbard_u, output='pyscf.out')
    mf = vayesta.lattmod.LatticeMF(mol)
    mf.kernel()

    def get_e_dmet(dm1, dm2):
        """DMET (projected DM) energy"""
        imp = np.s_[:nimp]
        e1 = np.einsum('ij,ij->', mf.get_hcore()[imp], dm1[imp])
        e2 = hubbard_u * np.einsum('iiii->', dm2[imp,imp,imp,imp])
        return e1 + e2
        
    def trace(dm2):
        return np.einsum('iiii->', dm2)
        

    # Without chemical potential optimization
    ecc = vayesta.ewf.EWF(mf, bno_threshold=np.inf, fragment_type='Site', solver='FCI', make_rdm1=True, make_rdm2=True)
    f = ecc.make_atom_fragment(list(range(nimp)), sym_factor=nsite/nimp)
    ecc.kernel()
    # c transforms to fragment site(s)
    c = f.c_active
    dm1 = np.einsum('ij,ai,bj->ab', f.results.dm1, c, c)
    dm2 = np.einsum('ijkl,ai,bj,ck,dl->abcd', f.results.dm2, c, c, c, c)/2

    # With chemical potential optimization (set nelectron_target)
    ecc_cpt = vayesta.ewf.EWF(mf, bno_threshold=np.inf, fragment_type='Site', solver='FCI', make_rdm1=True, make_rdm2=True)
    f = ecc_cpt.make_atom_fragment(list(range(nimp)), sym_factor=nsite/nimp, nelectron_target=nimp*filling)
    ecc_cpt.kernel()
    # c transforms to fragment site(s)
    c = f.c_active
    dm1_cpt = np.einsum('ij,ai,bj->ab', f.results.dm1, c, c)
    dm2_cpt = np.einsum('ijkl,ai,bj,ck,dl->abcd', f.results.dm2, c, c, c, c)/2

    # Exact energy
    fci = vayesta.ewf.EWF(mf, solver='FCI', bno_threshold=-np.inf, fragment_type='Site')
    fci_f = fci.make_atom_fragment(list(range(nsite)), name='lattice',make_rdm1=True, make_rdm2=True)
    fci.kernel()
    
    dm1_fci = fci_f.results.dm1
    dm2_fci = fci_f.results.dm2
    
    c = fci_f.c_active
    print(dm2_fci.shape)
    print(c.shape)
    dm1_fci = np.einsum('ij,ai,bj->ab', fci_f.results.dm1, c, c)
    dm2_fci = np.einsum('ijkl,ai,bj,ck,dl->abcd', fci_f.results.dm2, c, c, c, c)/2
    
    


    # --- OUTPUT
    print("U/t = ", hubbard_u)
    print("--------")
    print("Filling= %f" % filling)
    print("1DM without/with cpt: %f %f" % (dm1[0,0], dm1_cpt[0,0]))
    print("2DM without/with cpt: %f %f" % (dm2[0,0,0,0], dm2_cpt[0,0,0,0]))

    print("ENERGIES")

    print("--------")
    print("E%-20s %+16.8f Ha" % ('(MF)=',              mf.e_tot/nelectron))
    print("E%-20s %+16.8f Ha" % ('(proj-DM)[cpt=0]=', get_e_dmet(dm1, dm2)*nsite/(nimp*nelectron)))
    print("E%-20s %+16.8f Ha" % ('(proj-DM)[opt. cpt]=',        get_e_dmet(dm1_cpt, dm2_cpt)*nsite/(nimp*nelectron)))
    print("E%-20s %+16.8f Ha" % ('(proj-T)[cpt=0]=',  ecc.e_tot/nelectron))
    print("E%-20s %+16.8f Ha" % ('(proj-T)[opt. cpt]=',         ecc_cpt.e_tot/nelectron))
    print("E%-20s %+16.8f Ha" % ('(EXACT)=', fci.e_tot/nelectron))
    
    print('Diagonal RDM2 -- No CPT')
    for i in range(nsite):
        print(dm2[i,i,i,i])
    print('Diagonal RDM2 -- CPT')
    for i in range(nsite):
        print(dm2_cpt[i,i,i,i])
    print('Diagonal RDM2 -- FCI')
    for i in range(nsite):
        print(dm2_fci[i,i,i,i])
    
    return (mf.e_tot/nelectron,  # Mean field
            fci.e_tot/nelectron, # Exact lattice FCI
            get_e_dmet(dm1, dm2)*nsite/(nimp*nelectron), # RDM projection no CPT
            get_e_dmet(dm1_cpt, dm2_cpt)*nsite/(nimp*nelectron), # RDM projection and CPT
            ecc.e_tot/nelectron, # Amplitude projection no CPT
            ecc_cpt.e_tot/nelectron, # Amplitude projection CPT
            dm2[0,0,0,0], # Double occupation no CPT
            dm2_cpt[0, 0, 0, 0], # Double occupation of 0th basis CPT
            dm2_fci[0, 0, 0, 0]) # Double occupation of 0th basis FCI
            
            
            
            
hubbard_u_range = np.linspace(0, 10, 11)

nsite = 10
nelectron = 14
nimp = 2

E_mf = []
E_fci = []
E_rdm = []
E_rdm_cpt = []
E_amp = []
E_amp_cpt = []
double_occ = []
double_occ_cpt = []
double_occ_fci = []

for hubbard_u in hubbard_u_range:
    a, b, c, d, e, f, g, h, i = run_cpt(nsite, nelectron, nimp, hubbard_u)
    E_mf.append(a)
    E_fci.append(b)
    E_rdm.append(c)
    E_rdm_cpt.append(d)
    E_amp.append(e)
    E_amp_cpt.append(f)
    double_occ.append(g)
    double_occ_cpt.append(h)
    double_occ_fci.append(i)
    
# Plot double occupation
plt.title('nsite= '+str(nsite)+' , filling= '+str(np.round(nelectron/nsite, 2))+' , frag_size= '+str(nimp))
plt.plot(hubbard_u_range, np.array(double_occ), color='red', label='No CPT')
plt.plot(hubbard_u_range, np.array(double_occ_cpt), color='blue', label='CPT')
plt.plot(hubbard_u_range, np.array(double_occ_fci), color='green', label='FCI')

plt.xlabel('U/t')
plt.ylabel('Double occupancy per site')
plt.legend()
plt.grid()
plt.savefig('CPT_Hubbard_Occupation.jpeg')
plt.close()
    
# Plot energy comparison
plt.title('nsite= '+str(nsite)+' , filling= '+str(np.round(nelectron/nsite, 2))+' , frag_size= '+str(nimp))
plt.plot(hubbard_u_range, np.array(E_mf), color='black', label='HF Mean Field')
plt.plot(hubbard_u_range, np.array(E_fci), color='green', label='FCI')
plt.plot(hubbard_u_range, np.array(E_amp), 'x', color='red', label='Amplitude')
plt.plot(hubbard_u_range, np.array(E_rdm), 'x', color='blue', label='RDM')
plt.plot(hubbard_u_range, np.array(E_amp_cpt), 'x', color='orange', label='Amplitude-CPT')
plt.plot(hubbard_u_range, np.array(E_rdm_cpt), 'x', color='purple', label='RDM-CPT')

plt.xlabel('U/t')
plt.ylabel('Total energy per electron [t]')
plt.legend()
plt.grid()
plt.savefig('CPT_Hubbard_Energy.jpeg')
plt.close()
        
E_corr_FCI_error = np.array(E_fci)
E_corr_EWF_error = np.array(E_amp)
E_corr_RDM_error = np.array(E_rdm)
E_corr_EWF_CPT_error = np.array(E_amp_cpt)
E_corr_RDM_CPT_error = np.array(E_rdm_cpt)

# Calculate relative error in correlation energy
for i in range(len(E_fci)):
    E_corr_FCI_error[i] = rel_error(E_fci[i]-E_mf[i], E_fci[i]-E_mf[i])
    E_corr_EWF_error[i] = rel_error(E_amp[i]-E_mf[i], E_fci[i]-E_mf[i])
    E_corr_RDM_error[i] = rel_error(E_rdm[i]-E_mf[i], E_fci[i]-E_mf[i])
    E_corr_EWF_CPT_error[i] = rel_error(E_amp_cpt[i]-E_mf[i], E_fci[i]-E_mf[i])
    E_corr_RDM_CPT_error[i] = rel_error(E_rdm_cpt[i]-E_mf[i], E_fci[i]-E_mf[i])
    
plt.title('nsite= '+str(nsite)+' , filling= '+str(np.round(nelectron/nsite, 2))+' , frag_size= '+str(nimp))
plt.plot(hubbard_u_range, np.array(E_corr_FCI_error)*100, color='green', label='FCI')
plt.plot(hubbard_u_range, np.array(E_corr_EWF_error)*100, 'x', color='red', label='Amplitude')
plt.plot(hubbard_u_range, np.array(E_corr_RDM_error)*100, 'x', color='blue', label='RDM')
plt.plot(hubbard_u_range, np.array(E_corr_EWF_CPT_error)*100, 'x', color='orange', label='Amplitude-CPT')
plt.plot(hubbard_u_range, np.array(E_corr_RDM_CPT_error)*100, 'x', color='purple', label='RDM-CPT')


plt.xlabel('U/t')
plt.ylabel('Correlation energy per e- relative error [%]')
plt.legend()
plt.grid()
plt.savefig('CPT_Hubbard_Energy_Error.jpeg')
plt.close()
