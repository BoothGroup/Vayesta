import numpy as np

import pyscf
import pyscf.pbc
import pyscf.pbc.scf
import pyscf.pbc.cc
import pyscf.pbc.tools

import vayesta
import vayesta.ewf

cell = pyscf.pbc.gto.Cell()
a = 3.57
cell.atom = 'He 0.0 0.0 0.0'
cell.a = 3.0*np.eye(3)
cell.basis = 'def2-svp'
cell.verbose = 10
cell.output = 'pyscf.out'
cell.build()

kmesh = [1,1,2]
kpts = cell.make_kpts(kmesh)

# Hartree-Fock with k-points
kmf = pyscf.pbc.scf.KRHF(cell, kpts)
kmf = kmf.density_fit(auxbasis='def2-svp-ri')
kmf.kernel()

# Embedded calculation will automatically fold the k-point sampled mean-field to the supercell
emb = vayesta.ewf.EWF(kmf, bno_threshold=-1, store_dm1=True)
emb.iaopao_fragmentation()
#emb.iao_fragmentation()
emb.add_atomic_fragment(0, add_symmetric=False)
emb.add_atomic_fragment(1, add_symmetric=False)
emb.kernel()

dm1_demo = emb.make_rdm1_demo(ao_basis=True)
dm1_emb = emb.make_rdm1_ccsd(ao_basis=True)

# Full system reference CCSD
mf = emb.mf.density_fit(auxbasis='def2-svp-ri')
cc = pyscf.pbc.cc.CCSD(mf)
cc.kernel()
dm1_cc = cc.make_rdm1(ao_repr=True)

print("E(HF)=        %+16.8f Ha" % kmf.e_tot)
print("E(Emb. CCSD)= %+16.8f Ha" % emb.e_tot)
print("E(CCSD)=      %+16.8f Ha" % cc.e_tot)

print("Error DM1= %.3e" % np.linalg.norm(dm1_demo - dm1_cc))
print("Error DM1= %.3e" % np.linalg.norm(dm1_emb - dm1_cc))
