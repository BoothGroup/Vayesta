import numpy as np

import pyscf
import pyscf.pbc
import pyscf.pbc.scf
import pyscf.pbc.cc

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
cell.output = 'pyscf-22.out'
cell.build()

kpts = cell.make_kpts([2,2,2])

# Hartree-Fock with k-points
kmf = pyscf.pbc.scf.KRHF(cell, kpts)
kmf = kmf.density_fit(auxbasis='def2-svp-ri')
kmf.kernel()

# Full system CCSD
kcc = pyscf.pbc.cc.KCCSD(kmf)
kcc.kernel()

# Embedded calculation will automatically unfold the k-point sampled mean-field
ecc = vayesta.ewf.EWF(kmf, bno_threshold=1e-6)
ecc.make_atom_fragment(0, sym_factor=2)
ecc.kernel()

print("E(HF)=       %+16.8f Ha" % kmf.e_tot)
print("E(CCSD)=     %+16.8f Ha" % kcc.e_tot)
print("E(EWF-CCSD)= %+16.8f Ha" % ecc.e_tot)
