import pyscf
import pyscf.gto
import pyscf.scf
import pyscf.mp
import pyscf.cc

import vayesta
import vayesta.ewf

eta = 1e-5

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

# Embedded MP2
emp = vayesta.ewf.EWF(mf, solver='MP2', bno_threshold=eta)
emp.kernel()

# Embedded CCSD
ecc = vayesta.ewf.EWF(mf, bno_threshold=eta)
ecc.kernel()

# Reference full system MP2:
mp = pyscf.mp.UMP2(mf)
mp.kernel()

# Reference full system CCSD:
cc = pyscf.cc.UCCSD(mf)
cc.kernel()

print("E(HF)=        %+16.8f Ha" % mf.e_tot)
print("E(MP2)=       %+16.8f Ha" % mp.e_tot)
print("E(Emb. MP2)=  %+16.8f Ha" % emp.e_tot)
print("E(CCSD)=      %+16.8f Ha" % cc.e_tot)
print("E(Emb. CCSD)= %+16.8f Ha" % ecc.e_tot)
