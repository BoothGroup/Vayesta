import numpy as np
from numpy import einsum
import vayesta
import vayesta.ewf
import vayesta.lattmod

from scmf import SelfConsistentMF
nsite = 52
nimp = 2
boundary='PBC'
doping = 0
u_min = 0
u_max = 12
u_step = 1
do_fci = (nsite <= 12)

if doping:
    ne_target = (nsite + doping)/nsite * nimp
else:
    ne_target = None

nelectron = nsite + doping

mo_pdmet = None
mo_brueck = None


tvecs = [nsite//nimp, 1, 1]
# Infty hubbard test
'''
u_range = []
us_range = np.linspace(0.0, 1.0, 10)
for us in us_range:
    u_range.append(4*us/(1-us))
u_range = np.array(u_range)
'''

'''
t_max = 1.0
t_min = 1.0
t_step = abs(t_max-t_min)/10
'''
hubbard_t = 1
for uidx, hubbard_u in enumerate(range(u_min, u_max+1, u_step)):
    print("Hubbard-U= ", hubbard_u/hubbard_t)
    print("===============")
    mol = vayesta.lattmod.Hubbard1D(nsite, nelectron=nsite+doping, hubbard_t=hubbard_t, hubbard_u=hubbard_u)
    mf = vayesta.lattmod.LatticeMF(mol)
    mf.kernel()

    # Full system FCI
    fci_converged = False
    if do_fci:
        fci = vayesta.ewf.EWF(mf, solver='FCI', make_rdm1=True, make_rdm2=True, bath_type=None, fragment_type='Site')
        f = fci.make_atom_fragment(list(range(nsite)))
        fci.kernel()
        fci_converged = f.results.converged
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
    
    sc_pdmet = SelfConsistentMF(mf, sc_type='pdmet', tvecs=tvecs, nelectron_target=ne_target)
    sc_pdmet.kernel(nimp=nimp, mo_coeff=mo_pdmet)
    mo_pdmet = sc_pdmet.mf.mo_coeff

    sc_brueck = SelfConsistentMF(mf, sc_type='brueckner', tvecs=tvecs, nelectron_target=ne_target)
    sc_brueck.kernel(nimp=nimp, mo_coeff=mo_brueck)
    mo_brueck = sc_brueck.mf.mo_coeff
    
    # Run DMET calculation
    '''
    vcorr = None
    if (uidx != 0):
        vcorr_prev = fci_dmet.vcorr
    '''
    fci_dmet = vayesta.dmet.DMET(mf, fragment_type='Site', solver='FCI', make_rdm1=True, make_rdm2=True, charge_consistent=True)
    
    f_dmet = fci_dmet.make_atom_fragment(list(range(nimp)))
    frags_dmet = f_dmet.make_tsymmetric_fragments(tvecs=tvecs)
    '''
    if (uidx!=0):
        fci_dmet.vcorr = vcorr_prev
    '''
    fci_dmet.kernel()
    assert (len(frags_dmet)+1 == nsite//nimp)
    for f in fci_dmet.fragments:
        assert f.results.converged
        
    dmet_e_tot = fci_dmet.e_tot/mf.mol.nelectron
    dmet_docc = np.zeros((nsite,))
    for f, s in enumerate(range(0, nsite, nimp)):
        imp = fci_dmet.fragments[f].c_active[s:s+nimp]
        d = einsum("ijkl,xi,xj,xk,xl->x", fci_dmet.fragments[f].results.dm2, *(4*[imp]))/2
        dmet_docc[s:s+nimp] = d
    dmet_converged = fci_dmet.converged


    with open('1D-energies-dm.txt', 'a') as f:
        if uidx == 0:
            f.write("%12s  %16s  %16s %16s  \n" % ("#", "nsites", "nimps", "doping"))
            f.write("%12s  %16d  %16d %16d  \n" % ("# ", (nsite), (nimp), (doping)))
            f.write("%12s  %16s  %16s  %16s  %16s %16s %16s\n" % ("# Hubbard-U", "Exact", "Oneshot", "PDMET", "Brueckner", "DMET", "DMET_ncopt"))
        f.write('%12.1f  %16.8f  %16.8f  %16.8f  %16.8f %16.8f %16.8f \n' % (hubbard_u/hubbard_t, e_exact, oneshot.e_dmet, sc_pdmet.e_dmet, sc_brueck.e_dmet, dmet_e_tot, dmet_e_tot))

    with open('1D-energies-ewf.txt', 'a') as f:
        if uidx == 0:
            f.write("%12s  %16s  %16s %16s  \n" % ("#", "nsites", "nimps", "doping"))
            f.write("%12s  %16d  %16d %16d  \n" % ("# ", (nsite), (nimp), (doping)))
            f.write("%12s  %16s  %16s  %16s  %16s %16s %16s\n" % ("# Hubbard-U", "Exact", "Oneshot", "PDMET", "Brueckner", "DMET", "DMET_ncopt"))
        f.write('%12.1f  %16.8f  %16.8f  %16.8f  %16.8f %16.8f %16.8f\n' % (hubbard_u/hubbard_t, e_exact, oneshot.e_ewf, sc_pdmet.e_ewf, sc_brueck.e_ewf, dmet_e_tot, dmet_e_tot))

    with open('1D-docc.txt', 'a') as f:
        if uidx == 0:
            f.write("%12s  %16s  %16s %16s  \n" % ("#", "nsites", "nimps", "doping"))
            f.write("%12s  %16d  %16d %16d  \n" % ("# ", (nsite), (nimp), (doping)))
            f.write("%12s  %16s  %16s  %16s  %16s %16s %16s\n" % ("# Hubbard-U", "Exact", "Oneshot", "PDMET", "Brueckner", "DMET", "DMET_ncopt"))
        s = nimp // 2 # Take site at the center of fragment 0
        f.write('%12.1f  %16.8f  %16.8f  %16.8f  %16.8f %16.8f %16.8f\n' % (hubbard_u/hubbard_t, docc_exact, oneshot.docc[s], sc_pdmet.docc[s], sc_brueck.docc[s], dmet_docc[s], dmet_docc[s]))
        
    with open('1D-convergence.txt', 'a') as f:
        if uidx == 0:
            f.write("%12s  %16s  %16s %16s  \n" % ("#", "nsites", "nimps", "doping"))
            f.write("%12s  %16d  %16d %16d  \n" % ("# ", (nsite), (nimp), (doping)))
            f.write("%12s  %16s  %16s  %16s  %16s %16s %16s\n" % ("# Hubbard-U", "Exact", "Oneshot", "PDMET", "Brueckner", "DMET", "DMET_ncopt"))
        f.write('%12.1f %16.8s %d  %16.8s %d  %16.8s %d  %16.8s %d %16.8s %d %16.8s %d\n' % (hubbard_u/hubbard_t, str(fci_converged), 1,  str(oneshot.converged), oneshot.iteration, str(sc_pdmet.converged), sc_pdmet.iteration, str(sc_brueck.converged), sc_brueck.iteration,str(dmet_converged), 1,str(dmet_converged), 1))

