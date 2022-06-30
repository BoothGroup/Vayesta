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

ki, ka = 0, 1
kpti = kpts[0]
kpta = kpts[1]

ci = (
    +        np.random.random((cell.nao, 4))
    + 1.0j * np.random.random((cell.nao, 4))
)
ca = (
    +        np.random.random((cell.nao, 8))
    + 1.0j * np.random.random((cell.nao, 8))
)

# Similarly to pyscf, obtain four-center MO ERIs via ao2mo or get_mo_eri:
iajb = with_df.ao2mo((ci, ca, ci, ca), [kpti, kpta, kpti, kpta], compact=False)
iajb = iajb.reshape([ci.shape[-1], ca.shape[-1],]*2)

# Also access three-center MO integrals via ao2mo_3c or get_mo_3c_eri:
Lia = with_df.ao2mo_3c((ci, ca), [kpti, kpta], compact=False)
Lia = Lia.reshape(-1, ci.shape[-1], ca.shape[-1])
assert np.allclose(iajb, np.einsum('Lia,Ljb->iajb', Lia, Lia))

# ... or alternatively directly via the _cderi attribute:
Lia = np.einsum('Lpq,pi,qa->Lia', with_df._cderi[ki, ka], ci.conj(), ca)
assert np.allclose(iajb, np.einsum('Lia,Ljb->iajb', Lia, Lia))

# To perform a permanent basis change to the with_df object, simply rotate
# the entire _cderi into a basis:
c = (
    +        np.random.random((len(kpts), cell.nao, cell.nao)) 
    + 1.0j * np.random.random((len(kpts), cell.nao, cell.nao)) 
)
cderi = np.zeros_like(with_df._cderi)
for i in range(len(kpts)):
    for j in range(len(kpts)):
        cderi[i, j] = np.einsum('Lpq,pi,qj->Lij', with_df._cderi[i, j], c[i].conj(), c[j])
with_df._cderi = cderi
