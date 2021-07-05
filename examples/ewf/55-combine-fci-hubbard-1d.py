import numpy as np

import matplotlib.pyplot as plt

from pyscf import gto, scf, ao2mo
from pyscf import cc
from pyscf import fci
from scipy.interpolate import interp1d

import vayesta
import vayesta.lattmod
import vayesta.ewf

def amplitude_conversion(fragment):
    '''     Map single/double FCI amplitudes c1/c2 into CCSD cluster amplitudes t1/t2 for some fragment
    c1, c2 should be the amplitudes in the complete active space (CAS) (ie. after projecting out occupied/virtual subspaces and carrying out
    basis transformation into site basis)
    '''
    t1 = fragment.results.c1/fragment.results.c0  # Employ intermediate normalisation
    t2 = fragment.results.c2/fragment.results.c0  # Employ intermediate normalisation
    n_occ = t1.shape[0]
    n_vir = t1.shape[1]
    
    print(t1.shape)
    print(t2.shape)
    
    # Include uncorrelated double excitations for t2 too:
    # Loop through occupied orbital indices
    for i in range(n_occ):
        for j in range(n_occ):
        # Loop through unoccupied orbital indices if
            for a in range(n_vir):
                for b in range(n_vir):
                    t2[i, j, a, b] -= t1[i, a]*t1[j, b] #- t1[i, b]*t1[j,a] # Use normalised c1 amplitudes from above
    
    return t1, t2


def iterate(nsite=8, nelectron=8, hubbard_u=2.0):

    mol = vayesta.lattmod.Hubbard1D(nsite, nelectron=nelectron, hubbard_u=hubbard_u, output='pyscf.out')
    mf = vayesta.lattmod.LatticeMF(mol)
    mf.kernel()

    ecc = vayesta.ewf.EWF(mf, solver='FCI', fragment_type='Site')

    # Store each site as a separate fragment (assume translationsal symmetry on lattice)
    fragments = []
    # FCI solver results for fragment
    results = []
    # FCISD amplitude tensors for each fragment
    frag_c1 = []
    frag_c2 = []
    # Projection matrix into single-excitation subspace for each fragment
    projectors = []
    
    # Fragmentation
    for frag_index in range(nsite):
        fragments.append(ecc.make_atom_fragment(frag_index))

    # Carry out FCI calculation
    for frag_index in range(nsite):
        results.append(fragments[frag_index].kernel(np.inf))
    
    # Get amplitudes for each fragment and occupied basis projectors

    for frag_index in range(nsite):
        # Intermediate normalisation with MF amplitude
        frag_c1.append(results[frag_index].c1 / results[frag_index].c0)
        frag_c2.append(results[frag_index].c2 / results[frag_index].c0)
        
        # Fragment amplitudes
        #print(frag_c1)
        #print(frag_c2)
        
        projectors.append(fragments[frag_index].get_fragment_projector(fragments[frag_index].c_active_occ))
        
    # Carry out fragment projection:
    for frag_index in range(nsite):
        frag_c1[frag_index] = np.matmul(projectors[frag_index], frag_c1[frag_index])
        # Using einstein summation:
        frag_c2[frag_index] = np.einsum('kjab,ik->ijab', frag_c2[frag_index], projectors[frag_index])
        #frag_c2[frag_index] = np.tensordot(projectors[frag_index], frag_c2[frag_index], axes = 1)
        # Symmetrisation of c2 tensor per fragment -- unused
        # frag_c2[frag_index] = (frag_c2[frag_index] + frag_c2[frag_index].transpose(1, 0, 3, 2))

    # Total system amplitude containers:

    c1 = np.zeros((nsite, nsite))
    c2 = np.zeros(4*[nsite])

    # Carry out basis transformation and summation of fragments
    for frag_index in range(nsite):
        # Get transformation matrices
        c1_occ = fragments[frag_index].c_active_occ
        c1_vir = fragments[frag_index].c_active_vir
        
        # Carry out basis transformation
        frag_c1[frag_index] = np.matmul(c1_occ,np.matmul(frag_c1[frag_index], c1_vir.transpose()))
        frag_c2[frag_index] = np.einsum('ijab,xi,yj,pa,qb->xypq', frag_c2[frag_index], c1_occ, c1_occ, c1_vir, c1_vir)
        
        # Overwrite original fragment CI amplitudes:
        
        fragments[frag_index].results.c1 = frag_c1[frag_index]
        fragments[frag_index].results.c2 = frag_c2[frag_index]
        
        print('Amplitude Conversion.')
        # Apply amplitude conversion for the FCI active space already obtained and prepared:
        fragments[frag_index].results.t1, fragments[frag_index].results.t2 = amplitude_conversion(fragments[frag_index])
       # print(fragments[frag_index].results.t2)
        
        # Combining fragments together for representing the entirety of the system -- UNSURE HOW
        # For the time being, use simple addition (alternatives: direct sum/products?

    # 6) Use full c1, c2 to tailor a CCSD calculation
    # TODO: Tailored CC


    ecc = vayesta.ewf.EWF(mf, solver='CCSD',fragment_type='Site', bno_threshold=-np.inf)
    lattice = ecc.make_atom_fragment(list(range(nsite)))
    for fragment in fragments:
        lattice.couple_to_fragment(fragment)

    ecc.kernel()

    print("E%-11s %+16.8f Ha" % ('(HF)=', mf.e_tot))
    print("E%-11s %+16.8f Ha" % ('(EWF-CCSD)=', ecc.e_tot))
    
    '''
    cisolver = fci.FCI(mf)
    fci_energy, c0 = cisolver.kernel()

    
    mycc = cc.CCSD(mf)
    cc.diis = False
    #cc.iterative_damping = 0.1
    #cc.max_cycle = 300
    mycc.kernel()
    print('CCSD total energy ', mycc.e_tot)
    '''
    return (mf.e_tot, ecc.e_tot)#, fci_energy, mycc.e_tot)


#iterate(nsite=4, nelectron=4, hubbard_u=2.0)



    
U_range = np.linspace(0.0, 5.0, 44)
nsite = 10
nelectron = nsite # Half-filling

E_tot_EWF = []
E_tot_FCI = []
E_tot_CCSD = []
E_tot_MF = []
E_corr_EWF = []
E_corr_MP2 = []
E_corr_CCSD = []

for hubbard_u in U_range:
    e_tot_MF, e_tot_EWF  = iterate(nsite, nelectron, hubbard_u) #, e_tot_FCI, e_tot_CCSD = iterate(nsite, nelectron, hubbard_u)
    E_tot_MF.append(e_tot_MF)
    E_tot_EWF.append(e_tot_EWF)
    #E_tot_FCI.append(e_tot_FCI)
    #E_tot_CCSD.append(e_tot_CCSD)

    print()
    print('U/t = ', hubbard_u)
    print('Iteration finished.')


params = {
   'axes.labelsize': 40,
   'font.size': 40,
   'legend.fontsize': 40,
   'xtick.labelsize': 40,
   'ytick.labelsize': 40,
   'figure.figsize': [40, 15]
   }
plt.rcParams.update(params)
    
plt.title(str(nsite)+' electron, half-filled, 1D Hubbard model')
plt.plot(U_range, np.array(E_tot_EWF)/nelectron, 'x', color='red', label='EWF-CCSD')
plt.plot(U_range, np.array(E_tot_MF)/nelectron, color='orange', label='EWF-MF')
#plt.plot(U_range, np.array(E_tot_CCSD)/nelectron, color='blue', label='CCSD')
#plt.plot(U_range, np.array(E_tot_FCI)/nelectron, color='green', label='FCI')

#plt.plot(U_range, f_bethe(U_range), 'blue', label='Bethe R')
#plt.plot(U_range, (f(U_range)), 'green', label='Reference')

plt.xlabel('U/t')
plt.ylabel('Total energy per electron [t]')
plt.legend()
plt.grid()
plt.savefig('EWF_Hubbard_Energy.jpeg')
plt.close()
        


'''
# Earlier fragmentation test
f1 = ecc.make_atom_fragment(0)
f2 = ecc.make_atom_fragment(1)
#f3 = ecc.make_atom_fragment(3)
#f4 = ecc.make_atom_fragment(4)


results1 = f1.kernel(np.inf)
results2 = f2.kernel(np.inf)
#results3 = f3.kernel(np.inf)
#results4 = f4.kernel(np.inf)

# results have attributes 'c0', 'c1', 'c2'
# 1) Get intermediately normalized c1, c2
c1_1 = results1.c1 / results1.c0        # C_i^a
c2_1 = results1.c2 / results1.c0        # C_ij^ab

#c1_2 = results2.c1 / results2.c0
#c2_2 = results2.c2 / results2.c0
#c1_3 = results3.c1 / results3.c0
#c2_3 = results3.c2 / results3.c0
#c1_4 = results4.c1 / results4.c0
#c2_4 = results4.c2 / results4.c0

# 2) Get fragment projector
p1 = f1.get_fragment_projector(f1.c_active_occ)
#p2 = f2.get_fragment_projector(f2.c_active_occ)
#p3 = f3.get_fragment_projector(f3.c_active_occ)
#p4 = f4.get_fragment_projector(f4.c_active_occ)

# 3) Project c1 and c2 in first occupied index (apply p1 / p2)

print(c1_1.shape)
print(c2_1.shape)
print(p1.shape)


c1_1_bar = np.matmul(p1, c1_1)
c2_1_bar = np.tensordot(p1, c2_1, axes=1)

# Symmetrise c2 for each fragment:

c2_1_bar = (c2_1 + c2_1.transpose(1,0,3,2))/2

print('Projected tensor dim: ')
print(c1_1_bar.shape)
print(c2_1_bar.shape)

# 4) Transform each c1, c2 to a full, common basis (HF MO basis OR site basis?)
c1_occ = f1.c_active_occ    # (site, occupied orbital)
c1_vir = f1.c_active_vir    # (site, virtual orbital)

print('Basis generator dim:')
print(c1_occ.shape)
print(c1_vir.shape)

# Transform to site basis

c1_1_site= np.matmul(c1_occ,np.matmul(c1_1_bar, c1_vir.transpose()))

c2_1_site = np.einsum('ijab,xi,yj,pa,qb->xypq', c2_1_bar, c1_occ, c1_occ, c1_vir, c1_vir)

print('Basis transformed tensor dim: ')
print(c1_1_site.shape)
print(c2_1_site.shape)

# 5) Combine (add) to a single c1, c2
c1 = np.zeros((nsite, nsite))
c2 = np.zeros(4*[nsite])

c1 = c1_1_site
c2 = c2_1_site
#...

'''
