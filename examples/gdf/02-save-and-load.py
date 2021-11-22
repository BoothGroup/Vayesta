from pyscf.pbc import gto, scf
from vayesta.misc import gdf
import numpy as np
import os

cell = gto.Cell()
cell.atom = 'He 0 0 0'
cell.basis = '6-31g'
cell.a = np.eye(3) * 2
cell.verbose = 0
cell.build()

kpts = cell.make_kpts([2, 2, 1])

# Integrals can be saved using numpy's native ndarray.save method:
with_df = gdf.GDF(cell, kpts)
with_df.build()
with_df.save('cderi.tmp')

# GDF.build must be called, use argument with_j3c=True to prevent
# reconstruction of the integrals:
with_df = gdf.GDF(cell, kpts)
with_df.build(with_j3c=True)
with_df.load('cderi.tmp')

mf = scf.KRHF(cell, kpts)
mf.with_df = with_df
mf.kernel()
print('E(rhf) = %16.12f' % mf.e_tot)

os.remove('cderi.tmp')
