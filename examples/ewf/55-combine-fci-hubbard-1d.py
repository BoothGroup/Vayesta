import numpy as np

import matplotlib.pyplot as plt

from pyscf import gto, scf, ao2mo
from pyscf import cc
from pyscf import fci
from scipy.interpolate import interp1d

import vayesta
import vayesta.lattmod
import vayesta.ewf

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
    
def fragmentation_2D(nsites, frag_size):
    '''
    Partition 2D, (nsites[0], nsites[1]) square lattice into non-overlapping fragments of (frag_size[0], frag_size[1]) (DMET baths will overlap from fragment to fragment)
    Return list of index arrays with lattice indices belonging to fragments
    '''
    assert (nsites[0] % frag_size[0] == 0)
    assert (nsites[1] % frag_size[1] == 0)
    
    nfrag_x = int(nsites[0]/frag_size[0])
    nfrag_y = int(nsites[1]/frag_size[1])
    # No. fragments in directions
    
    fragment_indices = []
    fragment_index = []
    
    for nx in range(nfrag_x):
        for ny in range(nfrag_y):
            for i in range(frag_size[0]):
                for j in range(frag_size[1]):
                    fragment_index.append([frag_size[0]*nx + i, frag_size[1]*ny + j])
            fragment_indices.append(fragment_index)
            fragment_index = []
                
                        
    return fragment_indices
            
    
def fragmentation_1D(nsites, frag_size):
    '''
    Partition a 1D chain into non-overlapping fragments (DMET baths will overlap from fragment to fragment)
    Return list of fragment indices
    '''
    
    assert (nsites % frag_size == 0) # Ensure non-overlapping tiling
    
    nfragments = int(nsites/frag_size) # No. fragments
    
    fragment_indices = []
    frag_index = []
    
    for i in range(nsites):
        print(i)
        frag_index.append(i)
        if ((i+1) % frag_size == 0):
            fragment_indices.append(frag_index)
            frag_index = []
            
    #print(fragment_indices)
    return fragment_indices
    

def amplitude_conversion(fragment):
    '''Map single/double FCI amplitudes c1/c2 into CCSD cluster amplitudes t1/t2 for some fragment
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
        # Loop through unoccupied orbital indices
            for a in range(n_vir):
                for b in range(n_vir):
                    t2[i, j, a, b] -= t1[i, a]*t1[j, b] #- t1[i, b]*t1[j,a]
                    # Use normalised c1 amplitudes from above. Antisymmetric term is not needed as only an occupied/unoccupied basis states are assumed for Hubbard model
    
    return t1, t2
    
def c_projection(fragment):
    '''
    Projection into first occupied subspace of CISD amplitudes
    '''
    projection = fragment.get_fragment_projector(fragment.c_active_occ)
    
    c1_proj = np.einsum('ik, kj->ij', projector, fragment.results.c1)
    c2_proj = np.einsum('kjab, ik->ijab', fragment.results.c2, projector)
    
    return c1_proj, c2_proj
    
def t_projection(fragment):
    '''
    Projection into first occupied subspace of CCSD amplitudes
    '''
    projector = fragment.get_fragment_projector(fragment.c_active_occ)
    
    t1_proj = np.einsum('ik, kj->ij', projector, fragment.results.t1)
    t2_proj = np.einsum('kjab, ik->ijab', fragment.results.t2, projector)
    
    return t1_proj, t2_proj

            


def test_rdm(nsite=8, nelectron=8, hubbard_u=2.0, fragment_size=1):
    
    mol = vayesta.lattmod.Hubbard1D(nsite, nelectron=nelectron, hubbard_u=hubbard_u, output='pyscf.out', verbose=10)
    mol.build()


    ecc = vayesta.ewf.EWF(mf, solver='FCI', fragment_type='Site', make_rdm1=True, make_rdm2=True)

    # Store each site as a separate fragment (assume translationsal symmetry on lattice)
    fragments = []
    # FCI solver results for fragment
    results = []
    
    # Containers for fragment energy from density matrices
    E_dm = 0 # Unprojected density matrix energy
    E_dm_proj = 0 # Projected density matrix energy in the first fragment subspace (see below)

    # Fragmentation -> create fragment out of each lattice site separately
    assert (nsite % fragment_size == 0) # Ensure non-overlapping tiling
    
    nfragments = int(nsite/fragment_size)
    fragment_indices = fragmentation_1D(nsite, fragment_size)
    
    for frag_index in range(nfragments):
        fragments.append(ecc.make_atom_fragment(fragment_indices[frag_index]))
    
    for frag_index in range(nfragments):
        # Get amplitudes for each fragment
        # No normalisation of c amplitudes with MF amplitudes (carried out in
        fragments[frag_index].results.c1 = fragments[frag_index].kernel(np.inf).c1
        fragments[frag_index].results.c2 = fragments[frag_index].kernel(np.inf).c2
        
        # Carry out CISD->CCSD mapping
        fragments[frag_index].results.t1, fragments[frag_index].results.t2  = amplitude_conversion(fragments[frag_index])
        

        # Sanity check for RDM-s from the fragmented-uncoupled FCI
                
        dm1 = fragments[frag_index].results.dm1
        dm2 = fragments[frag_index].results.dm2
        
        print('Density Matrix dimensions')
        print(dm1.shape)
        print(dm2.shape)
        print()
        print('Calculating Density Matrix energies')
        
        # Get MO -> Occupation basis transformation matrix
        mo_coeff = np.hstack(( fragments[frag_index].c_active_occ,  fragments[frag_index].c_active_vir))
        print('Basis transformation matrix')
        print(mo_coeff.shape)
        
        # Transform density matrices into site basis Hamiltonian

        dm1 = np.einsum('pq, ip, jq->ij', dm1, mo_coeff, mo_coeff)
        dm2 = np.einsum('pqrs,ip,jq,kr,ls->ijkl', dm2, mo_coeff, mo_coeff, mo_coeff, mo_coeff)

        print('Site basis DM sdimension')
        print(dm1.shape)
        print(dm2.shape)
        print()

        # Calculate/trace out energy using density matrix
        

        h1 = mf.get_hcore() # Hopping matrix in site basis
        h2 = mf._eri # Repulsion tensor in site basis
       
        print('Hamiltonian shapes')
        print(h1.shape)
        print(h2.shape)

        if frag_index == 0:
            E_dm += np.einsum('ij, ij->', h1, dm1)
            #E_dm += np.einsum('ijkl, ijkl->', h2, dm2)
            E_dm += 0.5*hubbard_u*np.einsum('pppp->', dm2)

            E_dm_proj += np.einsum('ij, ij->', h1[:fragment_size], dm1[:fragment_size])
            E_dm_proj += 0.5*np.einsum('ijkl, ijkl->', h2[:fragment_size], dm2[:fragment_size])


    # Each fragment is translationally invariant so these energies will be the same for each fragment... One can just use the last one to
    
    # Run CCSD energy calculation
        
    # Carry out embedded CCSD calculation with the projected t1, t2 amplitudes
    ecc = vayesta.ewf.EWF(mf, solver='CCSD',fragment_type='Site', bno_threshold=-np.inf)
    lattice = ecc.make_atom_fragment(list(range(nsite)))
    for fragment in fragments:
        lattice.couple_to_fragment(fragment)

    ecc.kernel()

    #Compare energy of fragments
    E_CCSD = ecc.e_tot
    
    E_FCI = mf.e_tot + (nsite/fragment_size)*fragments[0].results.e_corr # Embedded FCI
    cisolver = fci.FCI(mf)
    fci_energy, c0 = cisolver.kernel() # non-embedded FCI
    
    E_FCI_ref = fci_energy
    E_RDM = (nsite/fragment_size)*E_dm # Non-projected density matrix energy
    E_RDM_proj = (nsite/fragment_size)*E_dm_proj # Projected density matrix energy
    
    print('Fragment energies')
    for fragment in fragments:
        print(fragment.results.e_corr*(nsite/fragment_size) + mf.e_tot )
    print('ENERGY COMPARISON')
    print('nelectron = nsite = ', nsite)
    print('fragment_size = ', fragment_size)

    print('U/t = ', hubbard_u)
    print('-----------------')
    print('MF total energy ', mf.e_tot)
    print('CCSD (coupled fragments) total energy ', E_CCSD)
    print('FCI total energy ', E_FCI)
    print('Unprojected RDM energy ', E_RDM)
    print('Projected RDM energy ', E_RDM_proj)
    print('-----------------')
    print('MF total energy per e- ', mf.e_tot/nelectron)
    
    # Consider
    
    return (mf.e_tot, E_FCI, E_RDM_proj, E_FCI_ref)
    
        
def test_energy_comparison(nsite=8, nelectron=8, U_range=np.linspace(0.0, 10.0, 11), fragment_size=1):
    
    E_MF = []
    E_FCI_EWF = []
    E_RDM_proj = []
    E_FCI = []
    for hubbard_u in U_range:
        a, b, c, d = test_rdm(nsite, nelectron, hubbard_u, fragment_size)
        E_MF.append(a)
        E_FCI_EWF.append(b)
        E_RDM_proj.append(c)
        E_FCI.append(d)
        
        
    params = {
    
       'axes.labelsize': 40,
       'font.size': 40,
       'legend.fontsize': 40,
       'lines.linewidth' : 4,
       'lines.markersize' : 10,
       'xtick.labelsize': 40,
       'ytick.labelsize': 40,
       'figure.figsize': [40, 15]
       }
    plt.rcParams.update(params)
        
    plt.title(str(nsite)+' site, '+str(nelectron)+' electron, 1D Hubbard model - Fragment size '+str(fragment_size))
    plt.plot(U_range, np.array(E_MF)/nelectron, color='orange', label='HF Mean Field')
    plt.plot(U_range, np.array(E_FCI)/nelectron, color='green', label='FCI')
    plt.plot(U_range, np.array(E_FCI_EWF)/nelectron, 'x', color='red', label='Projected Amplitude')
    plt.plot(U_range, np.array(E_RDM_proj)/nelectron, 'x', color='blue', label='Projected Density Matrix')
    

    #plt.plot(U_range, f_bethe(U_range), color='purple', label='Bethe Ansatz')

    #plt.plot(U_range, f_bethe(U_range), 'blue', label='Bethe R')
    #plt.plot(U_range, (f(U_range)), 'green', label='Reference')

    plt.xlabel('U/t')
    plt.ylabel('Total energy per electron [t]')
    plt.legend()
    plt.grid()
    plt.savefig('EWF_Hubbard_Energy.jpeg')
    plt.close()
    
    # Plot relative errors, remove zero divisions
    E_corr_exact = np.array(E_FCI)-np.array(E_MF)
    E_corr_fraction_amp = (np.array(E_FCI_EWF)-np.array(E_MF))
    E_corr_fraction_rdm = (np.array(E_RDM_proj)-np.array(E_MF))
    
    for i in range(len(E_corr_fraction_amp)):
        if E_corr_exact[i] == E_corr_fraction_amp[i]:
            E_corr_fraction_amp[i] = 0.0
        if E_corr_exact[i] == E_corr_fraction_rdm[i]:
            E_corr_fraction_rdm[i] = 0.0
        
        if (E_corr_exact[i]!=0.0):
            E_corr_fraction_rdm[i] -= E_corr_exact[i]
            E_corr_fraction_amp[i] -= E_corr_exact[i]
            E_corr_fraction_rdm[i] /= E_corr_exact[i]
            E_corr_fraction_amp[i] /= E_corr_exact[i]
            
        E_corr_exact[i] = 0.0
            
            
    
    plt.title(str(nsite)+' site, '+str(nelectron)+' electron, 1D Hubbard model - Fragment size '+str(fragment_size))
    #plt.plot(U_range, np.array(E_MF)/nelectron, color='orange', label='HF Mean Field')
    plt.plot(U_range, E_corr_exact*100, color='green', label='FCI')
    plt.plot(U_range, E_corr_fraction_amp*100, 'x', color='red', label='Projected Amplitude')
    plt.plot(U_range, E_corr_fraction_rdm*100, 'x', color='blue', label='Projected Density Matrix')
    

    #plt.plot(U_range, f_bethe(U_range), color='purple', label='Bethe Ansatz')

    #plt.plot(U_range, f_bethe(U_range), 'blue', label='Bethe R')
    #plt.plot(U_range, (f(U_range)), 'green', label='Reference')

    plt.xlabel('U/t')
    plt.ylabel('Correlation energy relative error [%]')
    plt.legend()
    plt.grid()
    plt.savefig('EWF_Hubbard_Energy_Fraction.jpeg')
    plt.close()
    

#test_energy_comparison(nsite=10, fragment_size=2)
#test_rdm(fragment_size=2)


def iterate(nsite=8, nelectron=8, hubbard_u=2.0, fragment_size=1):

    mol = vayesta.lattmod.Hubbard1D(nsite, nelectron=nelectron, hubbard_u=hubbard_u, output='pyscf.out', verbose=10)
    #mol.build()
    mf = vayesta.lattmod.LatticeMF(mol)
    mf.kernel()

    ecc = vayesta.ewf.EWF(mf, solver='FCI', fragment_type='Site')

    # Store each site as a separate fragment (assume translationsal symmetry on lattice)
    fragments = []
    # FCI solver results for fragment
    results = []
    
    # Fragmentation -> create fragment out of each lattice site separately
    
    assert (nsite % fragment_size == 0) # Ensure non-overlapping tiling
    
    nfragments = int(nsite/fragment_size)
    fragment_indices = fragmentation_1D(nsite, fragment_size)
    
    for frag_index in range(nfragments):
        fragments.append(ecc.make_atom_fragment(fragment_indices[frag_index]))
    
    for frag_index in range(nfragments):
        # Get amplitudes for each fragment
        # No normalisation of c amplitudes with MF amplitudes (carried out in
        fragments[frag_index].results.c1 = fragments[frag_index].kernel(np.inf).c1
        fragments[frag_index].results.c2 = fragments[frag_index].kernel(np.inf).c2
        
        '''
        # Alternative: carry out occupied basis projection first for CISD amplitudes, then carry out CISD -> CCSD mapping
        
        fragments[frag_index].results.c1, fragments[frag_index].results.c2 = c_projection(fragments[frag_index])
        
        # Carry out CISD->CCSD mapping
        fragments[frag_index].results.t1, fragments[frag_index].results.t2  = amplitude_conversion(fragments[frag_index])
        '''
        
        # Carry out CISD->CCSD mapping
        fragments[frag_index].results.t1, fragments[frag_index].results.t2  = amplitude_conversion(fragments[frag_index])
        '''
        # Unused -- Already performed in tailored CCSD calculation
        # Carry out first occupied basis projection for t1,t2 amplitude tensors:
        fragments[frag_index].results.t1, fragments[frag_index].results.t2 = t_projection(fragments[frag_index])
        '''
        
    # 6) Use full c1, c2 to tailor a CCSD calculation
    # TODO: Tailored CC

    # Carry out embedded CCSD calculation with the projected t1, t2 amplitudes
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

    '''
    
    '''
    mycc = cc.CCSD(mf)
    
    
    cc.diis = False
    #cc.iterative_damping = 0.1
    #cc.max_cycle = 300
    mycc.kernel()
    print('CCSD total energy ', mycc.e_tot)
    '''
    return (mf.e_tot, ecc.e_tot)#, fci_energy)#), mycc.e_tot)


def energy_calculation():
    
    U_range = np.linspace(0.0, 10.0, 11)
    nsite = 10
    nelectron = nsite # Half-filling
    fragment_size = 2

    E_tot_EWF = []
    E_tot_FCI = []
    E_tot_CCSD = []
    E_tot_MF = []
    E_corr_EWF = []
    E_corr_MP2 = []
    E_corr_CCSD = []

    for hubbard_u in U_range:
        e_tot_MF, e_tot_EWF = iterate(nsite, nelectron, hubbard_u, fragment_size)# e_tot_FCI = iterate(nsite, nelectron, hubbard_u, fragment_size)#, e_tot_CCSD = iterate(nsite, nelectron, hubbard_u, fragment_size)
        E_tot_MF.append(e_tot_MF)
        E_tot_EWF.append(e_tot_EWF)
        #E_tot_FCI.append(e_tot_FCI)
        #E_tot_CCSD.append(e_tot_CCSD)

        print()
        print('U/t = ', hubbard_u)
        print('Iteration finished.')
        

    U_bethe, E_pere_bethe = read_exact('hubbard1d-bethe.txt')
    f_bethe = interp1d(U_bethe, E_pere_bethe)

    params = {
       'axes.labelsize': 40,
       'font.size': 40,
       'legend.fontsize': 40,
       'lines.linewidth' : 4,
       'lines.markersize' : 10,
       'xtick.labelsize': 40,
       'ytick.labelsize': 40,
       'figure.figsize': [40, 15]
       }
    plt.rcParams.update(params)
        
    plt.title(str(nsite)+' electron, half-filled, 1D Hubbard model - Fragment size '+str(fragment_size))
    plt.plot(U_range, np.array(E_tot_EWF)/nelectron, 'x', color='red', label='EWF-CCSD')
    plt.plot(U_range, np.array(E_tot_MF)/nelectron, color='orange', label='EWF-MF')
    #plt.plot(U_range, np.array(E_tot_CCSD)/nelectron, color='blue', label='CCSD')
    #plt.plot(U_range, np.array(E_tot_FCI)/nelectron, color='green', label='FCI')
    plt.plot(U_range, f_bethe(U_range), color='purple', label='Bethe Ansatz')

    #plt.plot(U_range, f_bethe(U_range), 'blue', label='Bethe R')
    #plt.plot(U_range, (f(U_range)), 'green', label='Reference')

    plt.xlabel('U/t')
    plt.ylabel('Total energy per electron [t]')
    plt.legend()
    plt.grid()
    plt.savefig('EWF_Hubbard_Energy.jpeg')
    plt.close()

test_energy_comparison(nsite=10,nelectron=10, fragment_size=2)

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
