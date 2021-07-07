import numpy as np

import vayesta
import vayesta.ewf
import vayesta.lattmod

#nsites = (8, 8)
#nsite = nsites[0]*nsites[1]
nsite = 10
nimp = 2
fci_bno_threshold = np.inf
exact = True
nelectron = nsite
hubbard_u = 5.0

mol = vayesta.lattmod.Hubbard1D(nsite, nelectron=nelectron, hubbard_u=hubbard_u, output='pyscf.out', verbose=10)
#mol = vayesta.lattmod.Hubbard2D(nsites, nelectron=nelectron, hubbard_u=hubbard_u, output='pyscf.out', verbose=10)
mol.build()
mf = vayesta.lattmod.LatticeMF(mol)
mf.kernel()

# Calculate FCI fragments
fci = vayesta.ewf.EWF(mf, solver='FCI', bno_threshold=fci_bno_threshold, fragment_type='Site', make_rdm2=True)
for site in range(0, nsite, nimp):
    f = fci.make_atom_fragment(list(range(site, site+nimp)))
    f.kernel()
e_fci = mf.e_tot + nsite/nimp*fci.fragments[0].results.e_corr

ccsd = vayesta.ewf.EWF(mf, solver='CCSD', bno_threshold=-np.inf, fragment_type='Site', make_rdm1=True, make_rdm2=True)
f = ccsd.make_atom_fragment(list(range(nsite)), name='lattice')
# "Tailor" CCSD with FCI calculations
f.couple_to_fragments(fci.fragments)
ccsd.kernel()
print("E%-11s %+16.8f Ha" % ('(EWF-CCSD)=', ccsd.e_tot/nsite))

if exact:
    fci = vayesta.ewf.EWF(mf, solver='FCI', bno_threshold=-np.inf, fragment_type='Site')
    fci.make_atom_fragment(list(range(nsite)), name='lattice')
    fci.kernel()

print("E%-11s %+16.8f Ha" % ('(MF)=', mf.e_tot/nsite))
print("E%-11s %+16.8f Ha" % ('(FCI)=', e_fci/nsite))
print("E%-11s %+16.8f Ha" % ('(EWF-CCSD)=', ccsd.e_tot/nsite))
if exact:
    print("E%-11s %+16.8f Ha" % ('(exact)=', fci.e_tot/nsite))
