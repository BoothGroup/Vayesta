# Compare to PySCF:
# pyscf/examples/scf/40-customizing_hamiltonian.py

import vayesta
import vayesta.ewf
import vayesta.lattmod

nsite = 10
nelectron = nsite
hubbard_u = 2.0
mol = vayesta.lattmod.Hubbard1D(nsite, nelectron=nelectron, hubbard_u=hubbard_u, output='pyscf.out')
mf = vayesta.lattmod.LatticeMF(mol)
mf.kernel()

# Single site embedding:
ewf = vayesta.ewf.EWF(mf, bno_threshold=1e-8, fragment_type='Site')
ewf.make_atom_fragment(0, sym_factor=nsite)
ewf.kernel()
print("E%-11s %+16.8f Ha" % ('(MF)=', mf.e_tot/nelectron))
print("E%-11s %+16.8f Ha" % ('(EWF-CCSD)=', ewf.e_tot/nelectron))

# Double site embedding:
ewf = vayesta.ewf.EWF(mf, bno_threshold=1e-8, fragment_type='Site')
ewf.make_atom_fragment([0,1], sym_factor=nsite/2)
ewf.kernel()
print("E%-11s %+16.8f Ha" % ('(MF)=', mf.e_tot/nelectron))
print("E%-11s %+16.8f Ha" % ('(EWF-CCSD)=', ewf.e_tot/nelectron))
