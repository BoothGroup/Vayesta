import numpy as np

import pyscf
import pyscf.pbc
import pyscf.pbc.scf

import vayesta
import vayesta.ewf

cell = pyscf.pbc.gto.Cell()
#cell.atom = ['C 0.0 0.0 0.0', 'C %f %f %f' % (a/4, a/4, a/4)]
#cell.a = np.asarray([
#    [a/2, a/2, 0],
#    [0, a/2, a/2],
#    [a/2, 0, a/2]])
cell.atom = ['He 0.0 0.0 0.0']
cell.a = 1.4 * np.eye(3)

cell.basis = 'def2-svp'
cell.verbose = 10
cell.output = 'pyscf_out.txt'
cell.build()

kpts = cell.make_kpts([2,2,2])

# Hartree-Fock with k-points
kmf = pyscf.pbc.scf.KRHF(cell, kpts)
kmf = kmf.density_fit(auxbasis='def2-svp-ri')
kmf.kernel()

# Embedded calculation will automatically unfold the k-point sampled mean-field
ecc = vayesta.ewf.EWF(kmf, bno_threshold=1e-6)
ecc.make_atom_fragment(0)
ecc.kernel()

print("E%-11s %+16.8f Ha" % ('(HF)=', kmf.e_tot))
print("E%-11s %+16.8f Ha" % ('(EWF-CCSD)=', ecc.e_tot))
