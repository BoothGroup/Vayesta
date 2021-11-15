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
mol.basis = 'aug-cc-pvdz'
mol.output = 'pyscf.out'
mol.build()

# Hartree-Fock
mf = pyscf.scf.RHF(mol)
mf.kernel()

ecc = vayesta.ewf.EWF(mf)
ecc.iao_fragmentation()
ecc.add_all_atomic_fragments()

bno_threshold=[1e-5, 1e-6, 1e-7, 1e-8]
energies = ecc.kernel(bno_threshold=bno_threshold)

print("E(HF)=     %+16.8f Ha" % mf.e_tot)
print("E(E-CCSD)= %+16.8f Ha" % ecc.e_tot)  # Returns the result for the last threshold
for i, threshold in enumerate(bno_threshold):
    print("E(eta=%.0e)= %+16.8f Ha" % (threshold, energies[i]))

# Reference full system CCSD:
cc = pyscf.cc.CCSD(mf)
cc.kernel()
print("E(CCSD)=   %+16.8f Ha" % cc.e_tot)
