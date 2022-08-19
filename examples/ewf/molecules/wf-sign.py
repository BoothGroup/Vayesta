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

# Embedded CCSD
emb = vayesta.ewf.EWF(mf, solver='MP2', bath_options=dict(threshold=1e-6))
with emb.fragmentation() as frag:
    frag.add_atomic_fragment(0)
    frag.add_atomic_fragment(1)
    frag.add_atomic_fragment(2)
    frag.add_atomic_fragment(0, wf_sign=-1)
    frag.add_atomic_fragment(1, wf_sign=-1)
    frag.add_atomic_fragment(2, wf_sign=-1)
emb.kernel()

dm1 = emb.make_rdm1(with_mf=False)
print(np.linalg.norm(dm1))

# Reference full system CCSD:
cc = pyscf.cc.CCSD(mf)
cc.kernel()

print("E(HF)=            %+16.8f Ha" % mf.e_tot)
print("E(CCSD)=          %+16.8f Ha" % cc.e_tot)
print("E(Emb. CCSD)[WF]= %+16.8f Ha" % emb.e_tot)
print("E(Emb. CCSD)[DM]= %+16.8f Ha" % emb.get_dm_energy())

#print("E(Emb. CCSD)[DM]= %+16.8f Ha" % emb._get_dm_energy_old(global_dm2=True))
