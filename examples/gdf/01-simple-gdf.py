from pyscf.pbc import gto, scf
from vayesta.misc import gdf
import numpy as np

cell = gto.Cell()
cell.atom = 'He 0 0 0'
cell.basis = '6-31g'
cell.a = np.eye(3) * 2
cell.verbose = 0
cell.build()

kpts = cell.make_kpts([2, 2, 1])

# Initialisation is functionally identical to pyscf.pbc.df.df.GDF:
with_df = gdf.GDF(cell, kpts)
with_df.build()

# Works seamlessly with pyscf MF methods:
mf = scf.KRHF(cell, kpts)
mf.with_df = with_df
mf.kernel()
print('E(rhf) = %16.12f' % mf.e_tot)

mf = scf.KUHF(cell, kpts)
mf.with_df = with_df
mf.kernel()
print('E(uhf) = %16.12f' % mf.e_tot)
