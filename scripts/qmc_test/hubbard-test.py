import numpy as np

import pyscf
import pyscf.gto
import pyscf.scf
import pyscf.fci
import pyscf.tools.ring

import vayesta
import vayesta.ewf
import vayesta.lattmod

nsite = 10
nelectron = nsite
nfrag = 2
hubbard_u = 2.0

mol = vayesta.lattmod.Hubbard1D(nsite, nelectron=nelectron, hubbard_u=hubbard_u, output='pyscf.out')
mf = vayesta.lattmod.LatticeMF(mol)
mf.kernel()
# FCIQMC input depends on specifying electron repulson integrals
mf._eri = mol.get_eri()

e_hf = mf.e_tot / nsite

fci = vayesta.ewf.EWF(mf, solver='FCI', fragment_type='Site', bath_type=None)
fci.make_atom_fragment(list(range(nsite)))
fci.kernel()
e_fci = fci.e_tot / nsite

ewf = vayesta.ewf.EWF(mf, solver='FCI', fragment_type='Site', bath_type=None, make_rdm1=True, make_rdm2=True)
for n in range(0, nsite, nfrag):
    ewf.make_atom_fragment(list(range(n, n+nfrag)), nelectron_target=nfrag)

# --- One-shot calculation:
ewf.kernel()
e_pwf_1 = ewf.get_total_energy() / nsite
e_pdm_1 = ewf.get_dmet_energy() / nsite

    # --- Self-consistent MF calculations:
    # These set the attribute ewf.with_scmf
    # Check vayesta/core/qemb/scmf.py for arguments

qmc_solver = vayesta.ewf.EWF(mf, solver="FCIQMC", bno_threshold=np.inf, fragment_type='Site', make_rdm1=True, make_rdm2=True)

frag_index = 0.0
for site in range(0, nsite, nfrag):
    f_qmc = qmc_solver.make_atom_fragment(list(range(site, site+nfrag)))
    f_qmc.kernel() # Automatically creates FCIDUMP/PKL for FCIQMC
    #fragment_H.to_pickle('Hubbard_Hamiltonian%d.pkl'%frag_index)
    #fragment_H.write_fcidump( fname='FCIDUMP_frag%d'%frag_index)
    

    frag_index += 1

e_qmc_amp = qmc_solver.get_total_energy() /nelectron
e_qmc_rdm = qmc_solver.get_dmet_energy()/nelectron
# Here the code assumes M7 has solved the above Hamiltonians outside of
# script and returned FCI amplitudes in "init_fname" .pkl files
# Perform energy calculation for each fragment

print('FCIQMC Embedding Test')
print('---------------------')
print('Energy [t] %1.8f Full system FCI', e_fci)
print('Energy [t] %1.8f EWF/FCIQMC -- Amplitudes', e_qmc_amp)
print('Energy [t] %1.8f EWF/FCI -- Amplitudes', e_pwf_1)

print('Energy [t] %1.8f EWF/FCIQMC -- RDM', e_qmc_rdm)
print('Energy [t] %1.8f EWF/FCI -- RDM ', e_pdm_1)

