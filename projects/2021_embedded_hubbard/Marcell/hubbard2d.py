import numpy as np

import vayesta
import vayesta.ewf
import vayesta.lattmod

from scmf import SelfConsistentMF

nsites = (4,2) # Lattice shape
nsite = nsites[0]*nsites[1] # Total no. sites

nimps = (2,1) # Fragment shape
nimp = nimps[0]*nimps[1] # Total no. sites in a fragment

doping = 0
ne_target = None # For doping, electron target for chem-pot

if doping != 0:
    ne_target = (nsite + doping)/nsite + nimp

u_min, u_max, u_step = (0, 12, 1) # hubbard u range

boundary_cond=('APBC', 'APBC')
do_fci = (nsite <= 12)

#tvecs = None
tvecs = [nsites[0]//nimps[0], nsites[1]//nimps[1], 1]

mo_pdmet = None
mo_brueck = None

for uidx, hubbard_u in enumerate(range(u_min, u_max+1, u_step)):
    
    print("Hubbard-U= %4.1f" % hubbard_u)
    print("===============")
    
    # Setup lattice Hamiltonian
    mol = vayesta.lattmod.Hubbard2D(nsites=nsites, nelectron=nsite+doping, hubbard_u=hubbard_u, output='pyscf.out', verbose=1, boundary=boundary_cond, tiles=nimps)
    
    # Run mean-field calculation
    mf = vayesta.lattmod.LatticeMF(mol)
    mf.kernel()
    
    # Full system FCI
    if do_fci:
        fci = vayesta.ewf.EWF(mf, solver='FCI', make_rdm1=True, make_rdm2=True, bath_type=None, fragment_type='Site')
        f = fci.make_atom_fragment(list(range(nsite)))
        fci.kernel()
        if f.results.converged:
            e_exact = fci.e_tot / mol.nelectron
            docc_exact = np.einsum("ijkl,i,j,k,l->", f.results.dm2, *(4*[f.c_active[0]]))/2
            
        else:
            print("Full-system FCI not converged.")
            e_exact = docc_exact = np.nan
        print("E(exact)= %.8f" % e_exact)
        print("Docc(exact)= %.8f" % docc_exact)
        
    else:
        e_exact = docc_exact = np.nan
        
    oneshot = SelfConsistentMF(mf, sc_type=None, nelectron_target=ne_target)
    oneshot.kernel(nimp=nimp)
    
    sc_pdmet = SelfConsistentMF(mf, sc_type='pdmet', tvecs= None, nelectron_target=ne_target)
    sc_pdmet.kernel(nimp=nimp, mo_coeff=mo_pdmet)
    mo_pdmet = sc_pdmet.mf.mo_coeff

    sc_brueck = SelfConsistentMF(mf, sc_type='brueckner', tvecs= None, nelectron_target=ne_target)
    sc_brueck.kernel(nimp=nimp, mo_coeff=mo_brueck)
    mo_brueck = sc_brueck.mf.mo_coeff

    with open('2D-energies-dm.txt', 'a') as f:
        if uidx == 0:
            f.write("%12s  %16s  %16s  \n" % ("# nsite", "nimp", "doping"))
            f.write("%12s  %16d  %16d %16d \n" % ("# ", (nsite), (nimp), (doping)))
            f.write("%12s  %16s  %16s  %16s  %16s\n" % ("# Hubbard-U", "Exact", "Oneshot", "PDMET", "Brueckner"))
        f.write('%12.1f  %16.8f  %16.8f  %16.8f  %16.8f\n' % (hubbard_u, e_exact, oneshot.e_dmet, sc_pdmet.e_dmet, sc_brueck.e_dmet))

    with open('2D-energies-ewf.txt', 'a') as f:
        if uidx == 0:
            f.write("%12s  %16s  %16s  \n" % ("# nsite", "nimp", "doping"))
            f.write("%12s  %16d  %16d %16d \n" % ("# ", (nsite), (nimp), (doping)))
            f.write("%12s  %16s  %16s  %16s  %16s\n" % ("# Hubbard-U", "Exact", "Oneshot", "PDMET", "Brueckner"))
        f.write('%12.1f  %16.8f  %16.8f  %16.8f  %16.8f\n' % (hubbard_u, e_exact, oneshot.e_ewf, sc_pdmet.e_ewf, sc_brueck.e_ewf))

    with open('2D-docc.txt', 'a') as f:
        if uidx == 0:
            f.write("%12s  %16s  %16s  \n" % ("# nsite", "nimp", "doping"))
            f.write("%12s  %16d  %16d %16d \n" % ("# ", (nsite), (nimp), (doping)))
            f.write("%12s  %16s  %16s  %16s  %16s\n" % ("# Hubbard-U", "Exact", "Oneshot", "PDMET", "Brueckner"))
        s = nimp // 2 # Take site at the center of fragment 0
        f.write('%12.1f  %16.8f  %16.8f  %16.8f  %16.8f\n' % (hubbard_u, docc_exact, oneshot.docc[s], sc_pdmet.docc[s], sc_brueck.docc[s]))

