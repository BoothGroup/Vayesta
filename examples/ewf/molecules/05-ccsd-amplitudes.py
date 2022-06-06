import numpy as np
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

# Full system CCSD:
cc = pyscf.cc.CCSD(mf)
cc.kernel()
t1_ref = cc.t1
t2_ref = cc.t2

# Embedded CCSD
emb = vayesta.ewf.EWF(mf, bath_options=dict(threshold=1e-4))
emb.kernel()
t1 = emb.get_global_t1()
t2 = emb.get_global_t2()

print("Error T1= %.3e" % np.linalg.norm(t1 - t1_ref))
print("Error T2= %.3e" % np.linalg.norm(t2 - t2_ref))
