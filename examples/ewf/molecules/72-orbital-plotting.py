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
cell.basis = 'sto-6g'
cell.output = 'pyscf.out'
cell.build()

kmesh = [2,2,2]
kpts = cell.make_kpts(kmesh)

# Hartree-Fock with k-points
kmf = pyscf.pbc.scf.KRHF(cell, kpts)
kmf = kmf.density_fit(auxbasis='sto-6g')
kmf.kernel()

# Embedded calculation will automatically fold the k-point sampled mean-field to the supercell
emb = vayesta.ewf.EWF(kmf, bath_options=dict(threshold=1e-6))
plot = CubeFile(emb.mol, gridsize=(100, 100, 100))
with emb.iao_fragmentation() as f:
    # Use emb.mol instead of mol, since this is the supercell with 2*(2x2x2)=16 atoms:
    for atom in range(3):   # Only add the first 3 atomic fragments
        fx = f.add_atomic_fragment(atom)
        plot.add_orbital(fx.c_frag)
# The filename can also be set in the initialization of the CubeFile
plot.write('iao/fragments.cube')

# The same with IAO+PAO fragmentation and with CubeFile as context manager:
emb = vayesta.ewf.EWF(kmf, bath_options=dict(threshold=1e-6))
with emb.iaopao_fragmentation() as f:
    # Note that in the context manager form,
    # the filename always needs to be set in the initialization of CubeFile!
    with CubeFile(emb.mol, filename='iao+pao/fragments.cube', gridsize=(100, 100, 100)) as plot:
        for atom in range(3):
            fx = f.add_atomic_fragment(atom)
            plot.add_orbital(fx.c_frag)
