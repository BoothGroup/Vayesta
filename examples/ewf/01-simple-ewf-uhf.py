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
mol.output = 'pyscf-01.out'
mol.build()

# Hartree-Fock
mf = pyscf.scf.UHF(mol)
mf.kernel()

ecc = vayesta.ewf.EWF(mf, bath_type='full')
ecc.iaopao_fragmentation()
ecc.add_atomic_fragment(0)
ecc.add_atomic_fragment(1)
ecc.add_atomic_fragment(2)
# Alternative: ecc.make_all_atom_fragments()
ecc.kernel()

print("E(HF)=     %+16.8f Ha" % mf.e_tot)
print("E(E-CCSD)= %+16.8f Ha" % ecc.e_tot)

# Reference full system CCSD:
cc = pyscf.cc.UCCSD(mf)
cc.kernel()
print("E(CCSD)=   %+16.8f Ha" % cc.e_tot)
