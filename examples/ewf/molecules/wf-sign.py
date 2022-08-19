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
emb = vayesta.ewf.EWF(mf, bath_options=dict(threshold=1e-6))
with emb.fragmentation() as frag:
    frag.add_atomic_fragment(0, solver='CCSD')
    frag.add_atomic_fragment(1, solver='CCSD')
    frag.add_atomic_fragment(2, solver='CCSD')
    frag.add_atomic_fragment(0, solver='MP2')
    frag.add_atomic_fragment(1, solver='MP2')
    frag.add_atomic_fragment(2, solver='MP2')
    frag.add_atomic_fragment(0, solver='MP2', wf_sign=-1)
    frag.add_atomic_fragment(1, solver='MP2', wf_sign=-1)
    frag.add_atomic_fragment(2, solver='MP2', wf_sign=-1)
emb.kernel()

emb2 = vayesta.ewf.EWF(mf, bath_options=dict(threshold=1e-6))
with emb2.fragmentation() as frag:
    frag.add_atomic_fragment(0, solver='CCSD')
    frag.add_atomic_fragment(1, solver='CCSD')
    frag.add_atomic_fragment(2, solver='CCSD')
emb2.kernel()

dm1 = emb.make_rdm1()
dm12 = emb2.make_rdm1()
print(np.linalg.norm(dm1 - dm12))

# Reference full system CCSD:
cc = pyscf.cc.CCSD(mf)
cc.kernel()

print("E(HF)=            %+16.8f Ha" % mf.e_tot)
print("E(CCSD)=          %+16.8f Ha" % cc.e_tot)
print("E(Emb. CCSD)[WF]= %+16.8f Ha" % emb.e_tot)
print("E(Emb. CCSD)[DM]= %+16.8f Ha" % emb.get_dm_energy())
print("E(Emb. CCSD)[WF]= %+16.8f Ha" % emb2.e_tot)
print("E(Emb. CCSD)[DM]= %+16.8f Ha" % emb2.get_dm_energy())


#print("E(Emb. CCSD)[DM]= %+16.8f Ha" % emb._get_dm_energy_old(global_dm2=True))
