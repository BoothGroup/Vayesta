# Compare to PySCF:
# pyscf/examples/scf/40-customizing_hamiltonian.py

import vayesta
import vayesta.ewf
import vayesta.lattmod

nsite = 10
nelectron = nsite
hubbard_u = 2.0
mol = vayesta.lattmod.Hubbard1D(nsite, nelectron=nelectron, hubbard_u=hubbard_u)
mf = vayesta.lattmod.LatticeMF(mol)
mf.kernel()

# Single site embedding:
emb = vayesta.ewf.EWF(mf, bath_options=dict(threshold=1e-4))
with emb.site_fragmentation() as frag:
    frag.add_atomic_fragment(0, sym_factor=nsite)
emb.kernel()

# Double site embedding:
emb2 = vayesta.ewf.EWF(mf, bath_options=dict(threshold=1e-4))
with emb2.site_fragmentation() as frag:
    frag.add_atomic_fragment([0,1], sym_factor=nsite/2)
emb2.kernel()

print("E(MF)=                %+16.8f Ha" % (mf.e_tot/nelectron))
print("E(Emb. CCSD, 1-site)= %+16.8f Ha" % (emb.e_tot/nelectron))
print("E(Emb. CCSD, 2-site)= %+16.8f Ha" % (emb2.e_tot/nelectron))
