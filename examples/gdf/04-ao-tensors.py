from pyscf.pbc import gto, scf
from vayesta.misc import gdf
import numpy as np

cell = gto.Cell()
cell.atom = 'He 0 0 0'
cell.basis = '6-31g'
cell.a = np.eye(3) * 2
cell.verbose = 0
cell.build()

kpts = cell.make_kpts([2, 2, 2])

with_df = gdf.GDF(cell, kpts)
with_df.build()

ki, kj = 0, 1
kpti = kpts[0]
kptj = kpts[1]

# Similarly to pyscf, obtain four-center AO ERIs via get_eri or get_ao_eri:
pqrs = with_df.get_eri([kpti, kptj, kpti, kptj], compact=False)
pqrs = pqrs.reshape([cell.nao,]*4)

# Also access three-center AO integrals via get_3c_eri or get_ao_3c_eri:
Lpq = with_df.get_3c_eri([kpti, kptj], compact=False)
Lpq = Lpq.reshape(-1, cell.nao, cell.nao)
assert np.allclose(pqrs, np.einsum('Lpq,Lrs->pqrs', Lpq, Lpq))

# ... or alternatively directly via the _cderi attribute:
Lpq = with_df._cderi[ki, kj]
assert np.allclose(pqrs, np.einsum('Lpq,Lrs->pqrs', Lpq, Lpq))
