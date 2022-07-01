import vayesta
import vayesta.ewf
import vayesta.lattmod

nsite = 16
nelectron = nsite
hubbard_u = 2.0
mol = vayesta.lattmod.Hubbard1D(nsite, nelectron=nelectron, hubbard_u=hubbard_u, output='pyscf.out')
mf = vayesta.lattmod.LatticeMF(mol)
mf.kernel()

# Single site embedding:
ewf = vayesta.ewf.EWF(mf, bno_threshold=1e-8, fragment_type='Site')
ewf.site_fragmentation()
ewf.add_atomic_fragment(0, sym_factor=nsite)
ewf.kernel()
print("E(MF)=       %+16.8f Ha" % (mf.e_tot/nelectron))
print("E(EWF-CCSD)= %+16.8f Ha" % (ewf.e_tot/nelectron))

# Double site embedding:
ewf = vayesta.ewf.EWF(mf, bno_threshold=1e-8, fragment_type='Site')
ewf.site_fragmentation()
ewf.add_atomic_fragment([0,1], sym_factor=nsite/2)
ewf.kernel()
print("E(MF)=       %+16.8f Ha" % (mf.e_tot/nelectron))
print("E(EWF-CCSD)= %+16.8f Ha" % (ewf.e_tot/nelectron))
