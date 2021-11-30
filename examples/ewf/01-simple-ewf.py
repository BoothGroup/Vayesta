import pyscf
import pyscf.gto
import pyscf.scf
import pyscf.cc

import vayesta
import vayesta.ewf

mol = pyscf.gto.Mole()
mol.atom = """
O  0.0000   0.0000   0.1173
H  0.0000   0.7572  -0.4692
H  0.0000  -0.7572  -0.4692
"""
mol.basis = 'cc-pVDZ'
mol.output = 'pyscf.out'
mol.build()

# Hartree-Fock
mf = pyscf.scf.RHF(mol)
mf.kernel()

emb = vayesta.ewf.EWF(mf, bno_threshold=1e-6)
emb.iao_fragmentation()
emb.add_atomic_fragment(0)
emb.add_atomic_fragment(1)
emb.add_atomic_fragment(2)
# Alternative: emb.make_all_atom_fragments()
emb.kernel()

print("E(HF)=        %+16.8f Ha" % mf.e_tot)
print("E(Emb. CCSD)= %+16.8f Ha" % emb.e_tot)

# Reference full system CCSD:
cc = pyscf.cc.CCSD(mf)
cc.kernel()
print("E(CCSD)=   %+16.8f Ha" % cc.e_tot)
