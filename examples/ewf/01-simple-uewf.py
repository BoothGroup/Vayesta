import pyscf
import pyscf.gto
import pyscf.scf
import pyscf.cc

import vayesta
import vayesta.ewf

mol = pyscf.gto.Mole()
mol.atom = """
N   0.0000   0.0000	0.0000
O   0.0000   1.0989	0.4653
O   0.0000  -1.0989     0.4653
"""
mol.basis = 'cc-pVDZ'
mol.output = 'pyscf.out'
mol.spin = 1
mol.build()

# Hartree-Fock
mf = pyscf.scf.UHF(mol)
mf.kernel()

emb = vayesta.ewf.EWF(mf, bno_threshold=1e-6)
emb.iao_fragmentation()
emb.make_all_atom_fragments()
emb.kernel()

print("E(HF)=        %+16.8f Ha" % mf.e_tot)
print("E(Emb. CCSD)= %+16.8f Ha" % emb.e_tot)

# Reference full system CCSD:
cc = pyscf.cc.UCCSD(mf)
cc.kernel()
print("E(CCSD)=   %+16.8f Ha" % cc.e_tot)
