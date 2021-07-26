import numpy as np

import vayesta
import vayesta.ewf
import vayesta.lattmod

from vayesta.misc import brueckner

nsite = 10
nimp = 2
#nelectron = (nsite - 4)
#nelectron = (nsite + 0)
#nelectron = (nsite + 4)
nelectron = (nsite - 4)
hubbard_u = 12.0
filling = nelectron/nsite
#nelectron_target = None
nelectron_target = nimp*filling

mol = vayesta.lattmod.Hubbard1D(nsite, nelectron=nelectron, hubbard_u=hubbard_u, output='pyscf.out', verbose=10)
#mol = vayesta.lattmod.Hubbard1D(nsite, nelectron=nelectron, hubbard_u=hubbard_u, v_nn=v_nn, output='pyscf.out', verbose=10)
#mol = vayesta.lattmod.Hubbard2D((4,4), nelectron=nelectron, hubbard_u=hubbard_u, output='pyscf.out', verbose=10)
mol.build()
mf = vayesta.lattmod.LatticeMF(mol)
mf.kernel()

ewf_opts = { 'fragment_type' : 'Site', 'make_rdm1' : True, 'make_rdm2' : True }

#fci = vayesta.ewf.EWF(mf, solver='FCI', bno_threshold=np.inf, **ewf_opts)
#fci.make_atom_fragment(list(range(nsite)))
#fci.kernel()
#print(fci.e_tot/nelectron)
#print(fci.fragments[0].results.dm2[0,0,0,0]/2)
#1/0

def get_e_dmet(f):
    """DMET (projected DM) energy"""
    c = f.c_active
    dm1 = np.einsum('ij,ai,bj->ab', f.results.dm1, c, c)
    dm2 = np.einsum('ijkl,ai,bj,ck,dl->abcd', f.results.dm2, c, c, c, c)
    imp = np.s_[:nimp]
    e1 = np.einsum('ij,ij->', mf.get_hcore()[imp], dm1[imp])
    e2 = hubbard_u * np.einsum('iiii->', dm2[imp,imp,imp,imp])/2
    return e1 + e2


e0 = None
etol = 1e-6
maxiter = 100
for it in range(1, maxiter+1):

    #print('MF diagonal:')
    #print(np.diag(mf.make_rdm1(), k=1))
    #print(np.diag(mf.make_rdm1()))
    #if it == 1:
    #    1/0

    # Calculate FCI fragments
    fci = vayesta.ewf.EWF(mf, solver='FCI', bno_threshold=np.inf, **ewf_opts)
    for site in range(0, nsite, nimp):
        f = fci.make_atom_fragment(list(range(site, site+nimp)), nelectron_target=nelectron_target)
        f.kernel()
    e_mf = mf.e_tot / nelectron
    e_corr = fci.fragments[0].results.e_corr/nimp / filling
    e_tot = e_mf + e_corr
    e_dmet = get_e_dmet(fci.fragments[0])/nimp / filling

    if e0 is None:
        e0 = e_tot
        delta = np.inf
    else:
        delta = (e_tot - e0)
        print("d(E)= %f" % delta)
    e0 = e_tot
    with open('results.txt', 'a') as f:
        f.write('%3d  %f  %f  %f  %f  %f  %f\n' % (it, e_mf, e_corr, e_tot, e_dmet, delta, fci.fragments[0].results.dm2[0,0,0,0]/2))

    if abs(delta) < etol:
        print("Brueckner MOs converged in %d iterations; exiting loop." % it)
        break

    # Get combined T1 amplitudes
    nocc = nelectron//2
    nvir = nsite - nocc
    occ = np.s_[:nocc]
    vir = np.s_[nocc:]
    t1 = np.zeros((nocc, nvir))
    for x in fci.fragments:
        px = x.get_fragment_projector(x.c_active_occ)
        pt1 = np.dot(px, x.results.c1/x.results.c0)
        # Rotate from cluster basis to MO basis
        ro = np.dot(x.c_active_occ.T, mf.mo_coeff[:,occ])
        rv = np.dot(x.c_active_vir.T, mf.mo_coeff[:,vir])
        t1 += np.einsum('ia,ip,aq->pq', pt1, ro, rv)
    print("Iteration %d: Norm of T1 amplitudes= %.3e" % (it, np.linalg.norm(t1)))

    # Update MF orbitals
    mf = brueckner.update_mf(mf, t1=t1, inplace=True, damping=0)

    #ecc = vayesta.ewf.EWF(mf, bno_threshold=np.inf, fragment_type='Site')
    #lattice = ecc.make_atom_fragment(list(range(nsite)))
    #lattice.couple_to_fragments(fci.fragments)
    #ecc.kernel()
    #print(ecc.e_tot / nelectron)
    #    
    #1/0
