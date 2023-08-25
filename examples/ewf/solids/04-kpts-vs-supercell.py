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
cell.atom = ["C 0.0 0.0 0.0", "C %f %f %f" % (a / 4, a / 4, a / 4)]
cell.a = np.asarray([[a / 2, a / 2, 0], [0, a / 2, a / 2], [a / 2, 0, a / 2]])
cell.basis = "def2-svp"
cell.output = "pyscf.out"
cell.build()

kmesh = [1, 1, 2]
kpts = cell.make_kpts(kmesh)

# Hartree-Fock with k-points
kmf = pyscf.pbc.scf.KRHF(cell, kpts)
kmf = kmf.density_fit(auxbasis="def2-svp-ri")
kmf.kernel()

# Full system CCSD
kcc = pyscf.pbc.cc.KCCSD(kmf)
kcc.kernel()

# Embedded calculation will automatically fold the k-point sampled mean-field to the supercell
emb = vayesta.ewf.EWF(kmf, bath_options=dict(threshold=1e-6))
with emb.iao_fragmentation() as f:
    f.add_atomic_fragment(0, sym_factor=2)  # 2 C-atoms per unit cell
emb.kernel()

# Hartree-Fock in supercell
scell = pyscf.pbc.tools.super_cell(cell, kmesh)
mf_sc = pyscf.pbc.scf.RHF(scell)
mf_sc = mf_sc.density_fit(auxbasis="def2-svp-ri")
mf_sc.kernel()

emb_sc = vayesta.ewf.EWF(mf_sc, bath_options=dict(threshold=1e-6))
with emb_sc.iao_fragmentation() as f:
    ncells = np.product(kmesh)
    f.add_atomic_fragment(0, sym_factor=2 * ncells)
emb_sc.kernel()

print("E(k-HF)=             %+16.8f Ha" % kmf.e_tot)
print("E(sc-HF)=            %+16.8f Ha" % (mf_sc.e_tot / ncells))
print("E(Emb. CCSD @k-HF)=  %+16.8f Ha" % emb.e_tot)
print("E(Emb. CCSD @sc-HF)= %+16.8f Ha" % (emb_sc.e_tot / ncells))
print("E(k-CCSD)=           %+16.8f Ha" % kcc.e_tot)
