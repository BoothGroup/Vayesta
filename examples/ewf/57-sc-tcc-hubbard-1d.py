import numpy as np

import vayesta
import vayesta.ewf
import vayesta.lattmod

from vayesta.misc import brueckner

nsite = 10
#nsites = [4, 4]
#nsite = nsites[0]*nsites[1]
nimp = 2
nelectron = nsite
#nelectron = 16
hubbard_u = 10.0
#v_nn = 2.0

mol = vayesta.lattmod.Hubbard1D(nsite, nelectron=nelectron, hubbard_u=hubbard_u, output='pyscf.out', verbose=10)
#mol = vayesta.lattmod.Hubbard1D(nsite, nelectron=nelectron, hubbard_u=hubbard_u, v_nn=v_nn, output='pyscf.out', verbose=10)
#mol = vayesta.lattmod.Hubbard2D((4,4), nelectron=nelectron, hubbard_u=hubbard_u, output='pyscf.out', verbose=10)
mol.build()
mf = vayesta.lattmod.LatticeMF(mol)
mf.kernel()

#fci = vayesta.ewf.EWF(mf, solver='FCI', bno_threshold=np.inf, fragment_type='Site')
#fci.make_atom_fragment(list(range(nsite)))
#fci.kernel()
#print(fci.e_tot/nelectron)
#1/0

e0 = None
etol = 1e-6
maxiter = 100
for it in range(1, maxiter+1):
    # Calculate FCI fragments
    #fci = vayesta.ewf.EWF(mf, solver='FCI', bno_threshold=np.inf, fragment_type='Site')
    fci = vayesta.ewf.EWF(mf, solver='CCSD', bno_threshold=np.inf, fragment_type='Site')
    for site in range(0, nsite, nimp):
        f = fci.make_atom_fragment(list(range(site, site+nimp)))
        f.kernel()
    e_mf = mf.e_tot / nsite
    e_corr = fci.fragments[0].results.e_corr/nimp
    e_tot = e_mf + e_corr

    if e0 is None:
        e0 = e_tot
        delta = np.inf
    else:
        delta = (e_tot - e0)
        print("d(E)= %f" % delta)
    e0 = e_tot
    with open('energies.txt', 'a') as f:
        f.write('%3d  %f  %f  %f  %f\n' % (it, e_mf, e_corr, e_tot, delta))

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
    #mf = brueckner.update_mf(mf, t1=t1, inplace=True)

    ecc = vayesta.ewf.EWF(mf, bno_threshold=np.inf, fragment_type='Site')
    lattice = ecc.make_atom_fragment(list(range(nsite)))
    lattice.couple_to_fragments(fci.fragments)
    ecc.kernel()
    print(ecc.e_tot / nelectron)
        
    1/0
