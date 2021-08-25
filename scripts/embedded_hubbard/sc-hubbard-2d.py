import numpy as np
from numpy import einsum

import vayesta
import vayesta.ewf
import vayesta.lattmod

from scmf import SelfConsistentMF

nsites = (10,10) # Lattice shape
nsite = nsites[0]*nsites[1] # Total no. sites

nimps = (2,2) # Fragment shape
nimp = nimps[0]*nimps[1] # Total no. sites in a fragment

doping = 0
ne_target = None # For doping, electron target for chem-pot

if doping != 0:
    ne_target = (nsite + doping)/nsite * nimp

nelectron = nsite + doping

u_min, u_max, u_step = (0, 12, 1) # hubbard u range

boundary_cond=('PBC', 'APBC')
do_fci = (nsite <= 12)

#tvecs = None
tvecs = [nsites[0]//nimps[0],1,nsites[1]//nimps[1]]

mo_pdmet = None
mo_brueck = None
mo_dmet = None
hubbard_t = 1
for uidx, hubbard_u in enumerate(range(u_min, u_max+1, u_step)):
    
    print("Hubbard-U= %4.1f" % (hubbard_u/hubbard_t))
    print("===============")
    
    # Setup lattice Hamiltonian
    mol = vayesta.lattmod.Hubbard2D(nsites=nsites, nelectron=nsite+doping, hubbard_u=hubbard_u, output='pyscf.out', verbose=1, boundary=boundary_cond, tiles=nimps)
    
    # Run mean-field calculation
    mf = vayesta.lattmod.LatticeMF(mol)
    mf.kernel()
    
    # Full system FCI
    fci_converged = False
    if do_fci:
        fci = vayesta.ewf.EWF(mf, solver='FCI', make_rdm1=True, make_rdm2=True, bath_type=None, fragment_type='Site')
        f = fci.make_atom_fragment(list(range(nsite)))
        fci.kernel()
        if f.results.converged:
            e_exact = fci.e_tot / mol.nelectron
            docc_exact = np.einsum("ijkl,i,j,k,l->", f.results.dm2, *(4*[f.c_active[0]]))/2
            fci_converged = True
        else:
            print("Full-system FCI not converged.")
            e_exact = docc_exact = np.nan
        print("E(exact)= %.8f" % e_exact)
        print("Docc(exact)= %.8f" % docc_exact)
        
    else:
        e_exact = docc_exact = np.nan
    
    # Run DMET calculation
    fci_dmet = vayesta.dmet.DMET(mf, fragment_type='Site', solver='FCI', make_rdm1=True, make_rdm2=True, charge_consistent=False, bno_threshold=None)
    
    #f_dmet = fci_dmet.make_atom_fragment(list(range(nimp)))
    #frags_dmet = []
    #frags_dmet = f_dmet.make_tsymmetric_fragments(tvecs=tvecs)
    #for fragment in frags_dmet:
    #    fragment.kernel()
    for site in range(0, nsite, nimp):
       f_dmet = fci_dmet.make_atom_fragment(list(range(site, site + nimp)))
    fci_dmet.kernel()
    #assert (len(frags_dmet) == nsite//nimp)
    for f in fci_dmet.fragments:
        assert f.results.converged
        
    dmet_e_tot = fci_dmet.e_tot/mf.mol.nelectron
    dmet_docc = np.zeros((nsite,))
    for f, s in enumerate(range(0, nsite, nimp)):
        imp = fci_dmet.fragments[f].c_active[s:s+nimp]
        d = einsum("ijkl,xi,xj,xk,xl->x", fci_dmet.fragments[f].results.dm2, *(4*[imp]))/2
        dmet_docc[s:s+nimp] = d
    dmet_converged = fci_dmet.converged
    
    
    oneshot = SelfConsistentMF(mf, sc_type=None, nelectron_target=ne_target, tvecs=None)
    oneshot.kernel(nimp=nimp)
    
    sc_pdmet = SelfConsistentMF(mf, sc_type='pdmet', tvecs=None, nelectron_target=ne_target)
    sc_pdmet.kernel(nimp=nimp, mo_coeff=mo_pdmet)
    mo_pdmet = sc_pdmet.mf.mo_coeff

    sc_brueck = SelfConsistentMF(mf, sc_type='brueckner', tvecs=None, nelectron_target=ne_target)
    sc_brueck.kernel(nimp=nimp, mo_coeff=mo_brueck)
    mo_brueck = sc_brueck.mf.mo_coeff

    with open('2D-energies-dm.txt', 'a') as f:
        if uidx == 0:
            f.write("%12s  %16s  %16s %16s  \n" % ("#", "nsites", "nimps", "doping"))
            f.write("%12s  %16d  %16d %16d  \n" % ("# ", (nsite), (nimp), (doping)))
            f.write("%12s  %16s  %16s  %16s  %16s %16s %16s\n" % ("# Hubbard-U", "Exact", "Oneshot", "PDMET", "Brueckner", "DMET", "DMET_ncopt"))
        f.write('%12.1f  %16.8f  %16.8f  %16.8f  %16.8f %16.8f %16.8f \n' % (hubbard_u/hubbard_t, e_exact, oneshot.e_dmet, sc_pdmet.e_dmet, sc_brueck.e_dmet, dmet_e_tot, dmet_e_tot))

    with open('2D-energies-ewf.txt', 'a') as f:
        if uidx == 0:
            f.write("%12s  %16s  %16s %16s  \n" % ("#", "nsites", "nimps", "doping"))
            f.write("%12s  %16d  %16d %16d  \n" % ("# ", (nsite), (nimp), (doping)))
            f.write("%12s  %16s  %16s  %16s  %16s %16s %16s\n" % ("# Hubbard-U", "Exact", "Oneshot", "PDMET", "Brueckner", "DMET", "DMET_ncopt"))
        f.write('%12.1f  %16.8f  %16.8f  %16.8f  %16.8f %16.8f %16.8f\n' % (hubbard_u/hubbard_t, e_exact, oneshot.e_ewf, sc_pdmet.e_ewf, sc_brueck.e_ewf, dmet_e_tot, dmet_e_tot))

    with open('2D-docc.txt', 'a') as f:
        if uidx == 0:
            f.write("%12s  %16s  %16s %16s  \n" % ("#", "nsites", "nimps", "doping"))
            f.write("%12s  %16d  %16d %16d  \n" % ("# ", (nsite), (nimp), (doping)))
            f.write("%12s  %16s  %16s  %16s  %16s %16s %16s\n" % ("# Hubbard-U", "Exact", "Oneshot", "PDMET", "Brueckner", "DMET", "DMET_ncopt"))
        s = nimp // 2 # Take site at the center of fragment 0
        f.write('%12.1f  %16.8f  %16.8f  %16.8f  %16.8f %16.8f %16.8f\n' % (hubbard_u/hubbard_t, docc_exact, oneshot.docc[s], sc_pdmet.docc[s], sc_brueck.docc[s], dmet_docc[s], dmet_docc[s]))
        
    with open('2D-convergence.txt', 'a') as f:
        if uidx == 0:
            f.write("%12s  %16s  %16s %16s  \n" % ("#", "nsites", "nimps", "doping"))
            f.write("%12s  %16d  %16d %16d  \n" % ("# ", (nsite), (nimp), (doping)))
            f.write("%12s  %16s  %16s  %16s  %16s %16s %16s\n" % ("# Hubbard-U", "Exact", "Oneshot", "PDMET", "Brueckner", "DMET", "DMET_ncopt"))
        f.write('%12.1f %16.8s %d  %16.8s %d  %16.8s %d  %16.8s %d %16.8s %d %16.8s %d\n' % (hubbard_u/hubbard_t, str(fci_converged), 1,  str(oneshot.converged), oneshot.iteration, str(sc_pdmet.converged), sc_pdmet.iteration, str(sc_brueck.converged), sc_brueck.iteration,str(dmet_converged), 1,str(dmet_converged), 1))

