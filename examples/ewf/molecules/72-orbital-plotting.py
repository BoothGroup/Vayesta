import numpy as np

import pyscf
import pyscf.pbc
import pyscf.pbc.scf

import vayesta
import vayesta.ewf
# For plotting:
from vayesta.misc.cubefile import CubeFile

cell = pyscf.pbc.gto.Cell()
a = 3.57
cell.atom = ['C 0.0 0.0 0.0', 'C %f %f %f' % (a/4, a/4, a/4)]
cell.a = np.asarray([
    [a/2, a/2, 0],
    [0, a/2, a/2],
    [a/2, 0, a/2]])
cell.basis = 'cc-pVDZ'
cell.output = 'pyscf.out'
cell.build()

kmesh = [2,2,2]
kpts = cell.make_kpts(kmesh)

# Hartree-Fock with k-points
kmf = pyscf.pbc.scf.KRHF(cell, kpts)
kmf = kmf.density_fit(auxbasis='cc-pVDZ-ri')
kmf.kernel()

# Embedded calculation will automatically fold the k-point sampled mean-field to the supercell
emb = vayesta.ewf.EWF(kmf, bno_threshold=1e-6)
emb.iao_fragmentation()
# Use emb.mol instead of mol, since this is the supercell with 2*(2x2x2)=16 atoms:
plot = CubeFile(emb.mol, gridsize=(100, 100, 100))
for atom in range(3):   # Only add the first 3 atomic fragments
    f = emb.add_atomic_fragment(atom)
    plot.add_orbital(f.c_frag)
# The filename can also be set in the initialization of the CubeFile
plot.write('iao/fragments.cube')

# The same with IAO+PAO fragmentation and with CubeFile as context manager:
emb = vayesta.ewf.EWF(kmf, bno_threshold=1e-6)
emb.iaopao_fragmentation()
# Note that in the context manager form,
# the filename always needs to be set in the initialization of CubeFile!
with CubeFile(emb.mol, filename='iao+pao/fragments.cube', gridsize=(100, 100, 100)) as plot:
    for atom in range(3):
        f = emb.add_atomic_fragment(atom)
        plot.add_orbital(f.c_frag)
