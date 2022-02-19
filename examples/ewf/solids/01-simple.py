import numpy as np

import pyscf
import pyscf.pbc
import pyscf.pbc.scf

import vayesta
import vayesta.ewf

cell = pyscf.pbc.gto.Cell()
cell.atom = ['He 0.0 0.0 0.0']
cell.a = 1.4 * np.eye(3)
cell.basis = 'def2-svp'
cell.output = 'pyscf.out'
cell.build()

kpts = cell.make_kpts([2,2,2])

# Hartree-Fock with k-points
kmf = pyscf.pbc.scf.KRHF(cell, kpts)
kmf = kmf.density_fit(auxbasis='def2-svp-ri')
kmf.kernel()

# Embedded calculation will automatically unfold the k-point sampled mean-field
emb = vayesta.ewf.EWF(kmf, bno_threshold=1e-3)
# If calling the kernel without initializiation fragmentation and fragments,
# IAO fragmentation and atomic fragments are used automatically
emb.kernel()

# Hartree-Fock with supercell
scell = pyscf.pbc.tools.super_cell(cell, [2,2,2])
mf = pyscf.pbc.scf.RHF(scell)
mf = mf.density_fit(auxbasis='def2-svp-ri')
mf.kernel()

emb_sc = vayesta.ewf.EWF(mf, bno_threshold=1e-3)
emb_sc.kernel()

print("k-point (k) and supercell (sc) energies:")
print("E(k-HF)=             %+16.8f Ha" % (kmf.e_tot))
print("E(sc-HF)=            %+16.8f Ha" % (mf.e_tot/8))
print("E(Emb-CCSD@k-HF)=    %+16.8f Ha" % (emb.e_tot))
print("E(Emb-CCSD@sc-HF)=   %+16.8f Ha" % (emb_sc.e_tot/8))
