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
cell.atom = ['C 0.0 0.0 0.0', 'C %f %f %f' % (a/4, a/4, a/4)]
cell.a = np.asarray([
    [a/2, a/2, 0],
    [0, a/2, a/2],
    [a/2, 0, a/2]])
cell.basis = 'def2-svp'
cell.output = 'pyscf.out'
cell.build()

kmesh = [1,1,2]
kpts = cell.make_kpts(kmesh)

# Hartree-Fock with k-points
kmf = pyscf.pbc.scf.KRHF(cell, kpts)
kmf = kmf.density_fit(auxbasis='def2-svp-ri')
kmf.kernel()

# Full system CCSD
kcc = pyscf.pbc.cc.KCCSD(kmf)
kcc.kernel()

# Embedded calculation will automatically fold the k-point sampled mean-field to the supercell
ecc = vayesta.ewf.EWF(kmf, bno_threshold=1e-6)
ecc.iao_fragmentation()
ecc.add_atomic_fragment(0, sym_factor=2) # 2 C-atoms per unit cell
ecc.kernel()

# Hartree-Fock in supercell
scell = pyscf.pbc.tools.super_cell(cell, kmesh)
mf_sc = pyscf.pbc.scf.RHF(scell)
mf_sc = mf_sc.density_fit(auxbasis='def2-svp-ri')
mf_sc.kernel()

ecc_sc = vayesta.ewf.EWF(mf_sc, bno_threshold=1e-6)
ecc_sc.iao_fragmentation()
ncells = np.product(kmesh)
ecc_sc.add_atomic_fragment(0, sym_factor=2*ncells)
ecc_sc.kernel()

print("E(k-HF)=            %+16.8f Ha" % kmf.e_tot)
print("E(sc-HF)=           %+16.8f Ha" % (mf_sc.e_tot/ncells))
print("E(EWF-CCSD@k-HF)=   %+16.8f Ha" % ecc.e_tot)
print("E(EWF-CCSD@sc-HF)=  %+16.8f Ha" % (ecc_sc.e_tot/ncells))
print("E(k-CCSD)=          %+16.8f Ha" % kcc.e_tot)
