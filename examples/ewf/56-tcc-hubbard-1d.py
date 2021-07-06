import numpy as np

import vayesta
import vayesta.ewf
import vayesta.lattmod

nsite = 12
nimp = 1
fci_bno_threshold = np.inf
nelectron = nsite
hubbard_u = 5.0
mol = vayesta.lattmod.Hubbard1D(nsite, nelectron=nelectron, hubbard_u=hubbard_u, output='pyscf.out')
mf = vayesta.lattmod.LatticeMF(mol)
mf.kernel()

# Calculate FCI fragments
fci = vayesta.ewf.EWF(mf, solver='FCI', bno_threshold=fci_bno_threshold, fragment_type='Site')
for site in range(0, nsite, nimp):
    f = fci.make_atom_fragment(list(range(site, site+nimp)), energy_factor=0)
    f.kernel()

ccsd = vayesta.ewf.EWF(mf, solver='CCSD', bno_threshold=-np.inf, fragment_type='Site')
f = ccsd.make_atom_fragment(list(range(nsite)), name='lattice')
# "Tailor" CCSD with FCI calculations
f.couple_to_fragments(fci.fragments)
ccsd.kernel()

print("E%-11s %+16.8f Ha" % ('(MF)=', mf.e_tot/nsite))
print("E%-11s %+16.8f Ha" % ('(EWF-CCSD)=', ccsd.e_tot/nsite))
