from pyscf.pbc import gto
from pyscf.pbc import df as df_pyscf
from vayesta.misc import gdf
import numpy as np
import time

cell = gto.Cell()
cell.atom = 'He 0 0 0'
cell.basis = '6-31g'
cell.a = np.eye(3) * 2
cell.verbose = 0
cell.build()

kpts = cell.make_kpts([2, 2, 1])

with_df = gdf.GDF(cell, kpts)
with_df.build()

with_df_pyscf = df_pyscf.GDF(cell, kpts)
with_df_pyscf.build()

dm = (
    +        np.random.random((len(kpts), cell.nao, cell.nao))
    + 1.0j * np.random.random((len(kpts), cell.nao, cell.nao))
)
dm = dm + dm.transpose(0, 2, 1).conj()

# Supports accelerated evaluation of J and K, but inputted density
# matrices must enumerate the same k-points as with_df.kpts:
t0 = time.time()
j0, k0 = with_df.get_jk(dm)
t1 = time.time()

t2 = time.time()
j1, k1 = with_df_pyscf.get_jk(dm)
t3 = time.time()

print('J equal:', np.allclose(j0, j1))
print('K equal:', np.allclose(k0, k1))
print('pyscf time:   %.6f s' % (t3-t2))
print('vayesta time: %.6f s' % (t1-t0))
