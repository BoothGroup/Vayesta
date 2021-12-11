import numpy as np

import pyscf
import pyscf.pbc
import pyscf.pbc.scf
import pyscf.pbc.tools

import vayesta
import vayesta.ewf
from vayesta.misc import solids

# Experimental Lattice constant
a0 = 4.4448

cell = pyscf.pbc.gto.Cell()
cell.a, cell.atom = solids.rocksalt(atoms=['Ni', 'O'], a=a0)
cell.basis = 'def2-svp'
cell.output = 'pyscf.out'
cell.build()

# Make magnetic supercell [8x Ni, 8x O]
supercell = [2,2,2]
cell = pyscf.pbc.tools.super_cell(cell, supercell)

kpts = None
#kpts = cell.make_kpts([2,2,2])

# Hartree-Fock
if kpts is not None:
    mf = pyscf.pbc.scf.KUHF(cell, kpts)
else:
    mf = pyscf.pbc.scf.UHF(cell)
mf = mf.density_fit(auxbasis='def2-svp-ri')
mf.kernel()

# Embedded calculation will automatically unfold the k-point sampled mean-field
eta = 1e-6
emb = vayesta.ewf.EWF(mf, bno_threshold=eta)
emb.iao_fragmentation()
ni1 = emb.add_atomic_fragment(0, sym_factor=4)
ni2 = emb.add_atomic_fragment(2, sym_factor=4)
o1 = emb.add_atomic_fragment(1, sym_factor=4)
o2 = emb.add_atomic_fragment(3, sym_factor=4)
emb.kernel()

emb.t1_diagnostics()

dm1_mf = emb.mf.make_rdm1()
emb.pop_analysis(dm1_mf, local_orbitals='mulliken', filename='pop-mf-mulliken.txt')
emb.pop_analysis(dm1_mf, local_orbitals='lowdin', filename='pop-mf-lowdin.txt')
emb.pop_analysis(dm1_mf, local_orbitals='iao+pao', filename='pop-mf-iao.txt')

dm1_cc = emb.make_ccsd_rdm1()
emb.pop_analysis(dm1_cc, local_orbitals='mulliken', filename='pop-cc-mulliken.txt')
emb.pop_analysis(dm1_cc, local_orbitals='lowdin', filename='pop-cc-lowdin.txt')
emb.pop_analysis(dm1_cc, local_orbitals='iao+pao', filename='pop-cc-iao.txt')

print("E(HF)=         %+16.8f Ha" % (mf.e_tot))
print("E(Emb. CCSD)=  %+16.8f Ha" % (emb.e_tot))
