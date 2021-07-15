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
   'lines.linewidth' : 4,
   'lines.markersize' : 10,
   'xtick.labelsize': 40,
   'ytick.labelsize': 40,
   'figure.figsize': [40, 15]
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
        Return relative error of some estimate "a" of quantity "b" scaled to reference value "b"
        '''
        if (a == b):
            return 0.0
        elif (b != 0.0):
            return (a-b)/b
        else:
            return 0.0

    
class Test_Hubbard_1D:
    def __init__(self, nsite=10, nelectron=10, hubbard_u=2.0, fragment_size=1):
        
        if (nsite != nelectron):
            print('Moving away from half-filling')
            
        assert (nsite % fragment_size == 0) # Ensure non-overlapping tiling
        
        # Lattice parameters
        
        self.__nfragments = int(nsite/fragment_size) # No. fragments in divided lattice
        self.__nimp = 1 # No. fragments to run the calculation on (exploit translational symmetry
    
        
        self.__nsite = nsite
        self.__nelectron = nelectron
        self.__hubbard_u = hubbard_u
        self.__fragment_size = fragment_size
        
    
        self.__filling = self.__nelectron/(self.__nsite)
        assert (self.__filling <= 2.0 and self.__filling > 0.0)
        
        # Initialise lattice Hamiltonian
        self.__mol = vayesta.lattmod.Hubbard1D(nsite, nelectron=nelectron, hubbard_u=hubbard_u, output='pyscf.out', verbose=10)
        self.__mol.build()

        # Initialise mean-field solver and obtain MF parameters:
        
        self.__mf = vayesta.lattmod.LatticeMF(self.__mol)
        self.__mf.kernel()
    
        # Initialise embedding solvers
        self.__ecc = vayesta.ewf.EWF(self.__mf, bno_threshold=np.inf, solver='FCI', fragment_type='Site', make_rdm1=True, make_rdm2=True)
        
        # Initialise solver for fragment couplings
        self.__ecc_lattice = vayesta.ewf.EWF(self.__mf, solver='CCSD',fragment_type='Site', bno_threshold=-np.inf)
        self.__lattice = self.__ecc_lattice.make_atom_fragment(list(range(nsite)))

        # Initalise fragments
        
        # "Brute-force" calculation for all fragments
        self.__fragments = self.create_fragments()
        
        # Chemical potential correction for fragments
        self.__cpt_fragments = self.create_fragments(cpt=True)
        

    def create_fragments(self, cpt=False):
        '''
        Carry out fragmentation with fragmented indices
        '''
        fragments = []
        indices = self.get_fragment_indices()
        for i in range(self.__nimp):
            if (cpt):
                fragments.append(self.__ecc.make_atom_fragment(indices[i]), sym_factor=self.__nsite/self.__nimp, nelectron_target=self.__nimp*self.__filling)
            else:
                fragments.append(self.__ecc.make_atom_fragment(indices[i]))
        
        return fragments
        
    def get_fragment_indices(self):
        '''
        Partition a 1D chain into non-overlapping fragments (DMET baths will overlap from fragment to fragment)
        Return list of fragment indices
        '''
        
        fragment_indices = []
        frag_index = []
        
        for i in range(self.__nsite):
            print(i)
            frag_index.append(i)
            if ((i+1) % self.__fragment_size == 0):
                fragment_indices.append(frag_index)
                frag_index = []
                
        #print(fragment_indices)
        return fragment_indices
        
    def amplitude_conversion(self, fragment):
        '''
        Map single/double FCI amplitudes c1/c2 into CCSD cluster amplitudes t1/t2 for some fragment
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
            
        
        
    def get_rdm_energy(self, fragment):
        '''
        Calculate energy estimation (projected) of a single fragment + bath system
        from reduced density matrices
        '''
        
        E_dm = 0.0
        E_dm_proj = 0.0
        
        # Obtain density matrices
        
        dm1 = fragment.results.dm1
        dm2 = fragment.results.dm2
        
        # Get MO -> Occupation basis transformation matrix
        mo_coeff = np.hstack(( fragment.c_active_occ,  fragment.c_active_vir))
        '''
        print('Basis transformation matrix')
        print(mo_coeff.shape)
        '''
        # Transform density matrices into site basis Hamiltonian

        dm1 = np.einsum('pq, ip, jq->ij', dm1, mo_coeff, mo_coeff)
        dm2 = np.einsum('pqrs,ip,jq,kr,ls->ijkl', dm2, mo_coeff, mo_coeff, mo_coeff, mo_coeff)
        '''
        print('Site basis DM sdimension')
        print(dm1.shape)
        print(dm2.shape)
        print()
        '''
        # Calculate/trace out energy using density matrices
        
        h1 = self.__mf.get_hcore() # Hopping matrix in site basis
        h2 = self.__mf._eri # Repulsion tensor in site basis
        '''
        print('Hamiltonian shapes')
        print(h1.shape)
        print(h2.shape)
        '''
        E_dm += np.einsum('ij, ij->', h1, dm1)
        #E_dm += np.einsum('ijkl, ijkl->', h2, dm2)
        E_dm += 0.5*self.__hubbard_u*np.einsum('pppp->', dm2)

        # PROJECTION ONLY WORKS FOR FIRST FRAGMENT
        # FOR THE REST OF THE FRAGMENTS SUBSPACE DIFFERS
        # FOR MF-FCI EMBEDDING, FRAGMENT RDM-s ARE THE SAME
        E_dm_proj += np.einsum('ij, ij->', h1[:self.__fragment_size], dm1[:self.__fragment_size])
        E_dm_proj += 0.5*np.einsum('ijkl, ijkl->', h2[:self.__fragment_size], dm2[:self.__fragment_size])
    
        # Return projected energy
        return E_dm_proj
        
        
    def cluster_FCI(self):
        '''
        Carry out FCI cluster calculation and modify CISD amplitudes
        '''
        # For each fragment:
        
        for fragment in self.__fragments:
            # Calcuate CISD amplitudes
            fragment.results.c1 = fragment.kernel(np.inf).c1
            fragment.results.c2 = fragment.kernel(np.inf).c2
            # Convert CISD amplitudes to CCSD amplitudes
            fragment.results.t1, fragment.results.t2  = self.amplitude_conversion(fragment)
    
    def lattice_coupling(self):
        '''
        Solve fragment problem on lattice
        '''
        for fragment in self.__fragments:
            self.__lattice.couple_to_fragment(fragment)
    
        self.__ecc_lattice.kernel()
        
    def run(self):
        '''
        Carry out embedding calculation
        '''
        self.cluster_FCI()
        self.lattice_coupling()
        
    def get_MF_tot_energy(self):
        '''
        Return full lattice mean-field energy
        '''
        return self.__mf.e_tot
        
        
    def get_FCI_tot_energy(self):
        '''
        Run FCI calculation on entire lattice (no fragmentation)
        '''
        cisolver = fci.FCI(self.__mf)
        fci_energy, c0 = cisolver.kernel() # non-embedded FCI
        
        return fci_energy
        
    def get_amp_tot_energy(self):
        '''
        Return total energy calculated from amplitudes
        '''
        return self.__ecc_lattice.e_tot
        
    def get_rdm_tot_energy(self):
        '''
        Return scaled RDM energy estimate from first fragment's rdms
        '''
        return (self.__nsite/self.__fragment_size)*self.get_rdm_energy(self.__fragments[0])
        
    def fragment_rdm_spectrum(self):
        '''
        Return spectrum of total RDM energy calculated for each fragment
        '''
        fragment_energies = []
        for fragment in self.__fragments:
            fragment_energies.append((self.__nsite/self.__fragment_size)*self.get_rdm_energy(fragment))
        
        return fragment_energies
        
    def fragment_amp_spectrum(self):
        '''
        Return spectrum of total amplitude calculated for each fragment
        '''
        fragment_energies = []
        for fragment in self.__fragments:
            fragment_energies.append(fragment.results.e_corr*(self.__nsite/self.__fragment_size) + self.get_MF_tot_energy())
        
        return fragment_energies
                
    
    def energy_comparison(self):
        '''
        Run energy checks
        '''
        print('ENERGY COMPARISON')
        print('nsite = ', self.__nsite)
        print('nelectron = ', self.__nelectron)
        print('fragment_size = ', self.__fragment_size)
        print('U/t = ', self.__hubbard_u)
        print('-----------------')
        print('MF total energy ', self.get_MF_tot_energy())
        print('CCSD (coupled fragments) total energy ', self.get_amp_tot_energy())
        print('FCI total energy ', self.get_FCI_tot_energy())
        print('Projected RDM energy ', self.get_rdm_tot_energy())
        
        print('-----------------')

def energy_test(nsite=10, nelectron=10, hubbard_u_range=np.linspace(0.0, 10.0, 11), fragment_size=1):
    '''
    Compare amplitude energy expectation and rdm energy expectation versus FCI/MF energies and their errors
    '''
    E_MF = []
    E_FCI_EWF = []
    E_RDM_proj = []
    E_FCI = []

    for hubbard_u in hubbard_u_range:
        # Run embedding simulation with specifications
        Simulation = Test_Hubbard_1D(nsite, nelectron, hubbard_u, fragment_size)
        Simulation.run()
        # Collect energy diagnostics
        E_MF.append(Simulation.get_MF_tot_energy()) # Mean field
        E_FCI_EWF.append(Simulation.get_amp_tot_energy()) # From cluster amplitudes
        E_RDM_proj.append(Simulation.get_rdm_tot_energy()) # Reduced density matrix
        E_FCI.append(Simulation.get_FCI_tot_energy()) # From full CI lattice calculation

    # Plot energy comparison
    plt.title(str(nsite)+' site, '+str(nelectron)+' electron, 1D Hubbard model - Fragment size '+str(fragment_size))
    plt.plot(hubbard_u_range, np.array(E_MF)/nelectron, color='orange', label='HF Mean Field')
    plt.plot(hubbard_u_range, np.array(E_FCI)/nelectron, color='green', label='FCI')
    plt.plot(hubbard_u_range, np.array(E_FCI_EWF)/nelectron, 'x', color='red', label='Projected Amplitude')
    plt.plot(hubbard_u_range, np.array(E_RDM_proj)/nelectron, 'x', color='blue', label='Projected Density Matrix')
    

    #plt.plot(U_range, f_bethe(U_range), color='purple', label='Bethe Ansatz')

    #plt.plot(U_range, f_bethe(U_range), 'blue', label='Bethe R')
    #plt.plot(U_range, (f(U_range)), 'green', label='Reference')

    plt.xlabel('U/t')
    plt.ylabel('Total energy per electron [t]')
    plt.legend()
    plt.grid()
    plt.savefig('Test_Hubbard_Energy.jpeg')
    plt.close()
    # Plot relative errors, remove zero divisions
    
    E_corr_FCI_error = np.array(E_FCI)
    E_corr_EWF_error = np.array(E_FCI_EWF)
    E_corr_RDM_error = np.array(E_RDM_proj)
    assert len(E_MF) == len(E_FCI) == len(E_RDM_proj) == len(E_FCI_EWF)
    
    # Calculate relative error in correlation energy
    for i in range(len(E_MF)):
        E_corr_FCI_error[i] = rel_error(E_FCI[i]-E_MF[i], E_FCI[i]-E_MF[i])
        E_corr_EWF_error[i] = rel_error(E_FCI_EWF[i]-E_MF[i], E_FCI[i]-E_MF[i])
        E_corr_RDM_error[i] = rel_error(E_RDM_proj[i]-E_MF[i], E_FCI[i]-E_MF[i])
    
    
    plt.title(str(nsite)+' site, '+str(nelectron)+' electron, 1D Hubbard model - Fragment size '+str(fragment_size))
    plt.plot(hubbard_u_range, np.array(E_corr_FCI_error)*100, color='green', label='FCI')
    plt.plot(hubbard_u_range, np.array(E_corr_EWF_error)*100, 'x', color='red', label='Projected Amplitude')
    plt.plot(hubbard_u_range, np.array(E_corr_RDM_error)*100, 'x', color='blue', label='Projected Density Matrix')
    
    plt.xlabel('U/t')
    plt.ylabel('Correlation energy relative error [%]')
    plt.legend()
    plt.grid()
    plt.savefig('Test_Hubbard_Energy_Error.jpeg')
    plt.close()
    
