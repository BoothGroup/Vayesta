import numpy as np
from numpy import einsum

import vayesta
import vayesta.ewf
import vayesta.lattmod

from scmf import SelfConsistentMF

nsites = (20,20) # Lattice shape
nsite = nsites[0]*nsites[1] # Total no. sites

nimps = (2,2) # Fragment shape
nimp = nimps[0]*nimps[1] # Total no. sites in a fragment

doping = 0
ne_target = None # For doping, electron target for chem-pot

if doping != 0:
    ne_target = (nsite + doping)/nsite * nimp

nelectron = nsite + doping

u_min, u_max, u_step = (0, 12, 0.1) # hubbard u range

boundary_cond=('PBC', 'APBC')
do_fci = (nsite <= 12)

#tvecs = None
tvecs = [nsites[0]//nimps[0],1,nsites[1]//nimps[1]]


dmet_vcorr = None

hubbard_t = 1
hubbard_u_range = np.linspace(u_min, u_max, (u_max-u_min)/u_step+1)
uidx = 0
for hubbard_u in hubbard_u_range:
    print("Hubbard-U= %4.1f" % (hubbard_u/hubbard_t))
    print("===============")
    
    # Setup lattice Hamiltonian
    mol = vayesta.lattmod.Hubbard2D(nsites=nsites, nelectron=nsite+doping, hubbard_u=hubbard_u, output='pyscf.out', verbose=1, boundary=boundary_cond, tiles=nimps)
    
    # Run mean-field calculation
    mf = vayesta.lattmod.LatticeMF(mol, allocate_eri = False)
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
    
    if dmet_vcorr is None:
        fci_dmet = vayesta.dmet.LatticeDMET(mf, fragment_type='Site', solver='FCI', make_rdm1=True, make_rdm2=True, charge_consistent=False)
    else:
        fci_dmet = vayesta.dmet.LatticeDMET(mf, fragment_type='Site', solver='FCI', make_rdm1=True, make_rdm2=True, charge_consistent=False)
        fci_dmet.vcorr = dmet_vcorr
    
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
    dmet_e_dmet = fci_dmet.e_tot/mf.mol.nelectron
    dmet_docc = np.zeros((nsite,))
   
    for f, s in enumerate(range(0, nsite, nimp)):
        imp = fci_dmet.fragments[f].c_active[s:s+nimp]
        d = einsum("ijkl,xi,xj,xk,xl->x", fci_dmet.fragments[f].results.dm2, *(4*[imp]))/2
        dmet_docc[s:s+nimp] = d
    dmet_converged = fci_dmet.converged
        
    dmet_e_tot = fci_dmet.e_tot/mf.mol.nelectron
    dmet_docc = np.zeros((nsite,))
    
    for f, s in enumerate(range(0, nsite, nimp)):
        imp = fci_dmet.fragments[f].c_active[s:s+nimp]
        d = einsum("ijkl,xi,xj,xk,xl->x", fci_dmet.fragments[f].results.dm2, *(4*[imp]))/2
        dmet_docc[s:s+nimp] = d
    dmet_converged = fci_dmet.converged
    
    # Restart next calculation from the current correlation potential
    dmet_vcorr = fci_dmet.vcorr
    
    with open('2D-energies-dm-vdmet.txt', 'a') as f:
        if uidx == 0:
            f.write("%12s %16s %16s %16s %16s \n" % ("#", "nsites", "nimps", "doping", "boundary"))
            f.write("%12s  %16d  %16d %16d %16s \n" % ("# ", (nsite), (nimp), (doping), str(boundary_cond)))
            f.write("%12s  %16s \n" % ("# Hubbard-U", "dmet"))
        f.write('%12.1f  %16.10f  \n' % (hubbard_u/hubbard_t, dmet_e_dmet))
    s = nimp //2
    with open('2D-docc-vdmet.txt', 'a') as f:
        if uidx == 0:
            f.write("%12s %16s %16s %16s %16s \n" % ("#", "nsites", "nimps", "doping", "boundary"))
            f.write("%12s  %16d  %16d %16d %16s \n" % ("# ", (nsite), (nimp), (doping), str(boundary_cond)))
            f.write("%12s  %16s \n" % ("# Hubbard-U", "dmet"))
        f.write('%12.1f  %16.10f \n' % (hubbard_u/hubbard_t, dmet_docc[s]))
        
    with open('2D-convergence-vdmet.txt', 'a') as f:
        if uidx == 0:
            f.write("%12s %16s %16s %16s %16s \n" % ("#", "nsites", "nimps", "doping", "boundary"))
            f.write("%12s  %16d  %16d %16d %16s \n" % ("# ", (nsite), (nimp), (doping), str(boundary_cond)))
            f.write("%12s  %16s \n" % ("# Hubbard-U", "dmet"))
        f.write('%12.1f %16.10s %d \n' % (hubbard_u/hubbard_t,  str(dmet_converged), 1)) # convergence iteration inaccessible currently
    
    uidx += 1
