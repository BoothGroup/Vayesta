import numpy as np

import vayesta
import vayesta.ewf
import vayesta.lattmod

#nsites = (8, 8)
#nsite = nsites[0]*nsites[1]
nsite = 10
nimp = 1
fci_bno_threshold = np.inf
exact = True
#exact = False
nelectron = nsite+4
#nelectron = ns
hubbard_u = 10.0

bound = 'auto'
#bound = 'pbc'
mol = vayesta.lattmod.Hubbard1D(nsite, nelectron=nelectron, hubbard_u=hubbard_u, boundary=bound, output='pyscf.out', verbose=10)
#mol = vayesta.lattmod.Hubbard2D(nsites, nelectron=nelectron, hubbard_u=hubbard_u, output='pyscf.out', verbose=10)
mol.build()
mf = vayesta.lattmod.LatticeMF(mol)
mf.kernel()

# Calculate FCI fragments
fci = vayesta.ewf.EWF(mf, solver='FCI', bno_threshold=fci_bno_threshold, fragment_type='Site', make_rdm1=True, make_rdm2=True)
e_corr = 0.0
e_corrs = []
e_dmet = 0.0
for idx, site in enumerate(range(0, nsite, nimp)):
#for site in [0]:
    f = fci.add_atomic_fragment(list(range(site, site + nimp)))
    f.kernel()
    e_corr_f = fci.fragments[0].results.e_corr
    print("Correlation energy fragment %2d= % 16.8f" % (idx, e_corr_f))
    e_corr += e_corr_f
    #e_corrs.append(e_corr_f)

    mo = np.hstack((f.c_active_occ, f.c_active_vir))
    dm1 = np.einsum('ij,pi,qj->pq', f.results.dm1, mo, mo)
    dm2 = np.einsum('ijkl,pi,qj,rk,sl->pqrs', f.results.dm2, mo, mo, mo, mo)

    frag = np.s_[site:site+nimp]
    #print(dm1[frag,frag])
    #1/0
    e_dmet_f = np.einsum('ij,ij->', mf.get_hcore()[frag], dm1[frag])
    e_dmet_f += 0.5*np.einsum('pqrs,pqrs->', dm2[frag], mf._eri[frag])
    #e_corrs.append(e_dmet_f)
    e_dmet += e_dmet_f

#print(e_corrs)
#1/0
e_fci = mf.e_tot + e_corr

ccsd = vayesta.ewf.EWF(mf, solver='CCSD', bno_threshold=-np.inf, fragment_type='Site')
f = ccsd.add_atomic_fragment(list(range(nsite)), name='lattice')
# "Tailor" CCSD with FCI calculations
f.couple_to_fragments(fci.fragments)
ccsd.kernel()
print("E%-11s %+16.8f Ha" % ('(EWF-CCSD)=', ccsd.e_tot/nsite))
#1/0

if exact:
    fci = vayesta.ewf.EWF(mf, solver='FCI', bno_threshold=-np.inf, fragment_type='Site')
    fci.add_atomic_fragment(list(range(nsite)), name='lattice')
    fci.kernel()

print("E%-11s %+16.8f Ha" % ('(MF)=', mf.e_tot/nsite))
print("E%-11s %+16.8f Ha" % ('(DMET)=', e_dmet/nsite))
print("E%-11s %+16.8f Ha" % ('(FCI)=', e_fci/nsite))
print("E%-11s %+16.8f Ha" % ('(EWF-CCSD)=', ccsd.e_tot/nsite))
if exact:
    print("E%-11s %+16.8f Ha" % ('(exact)=', fci.e_tot/nsite))
