import numpy as np

import pyscf
import pyscf.pbc
import pyscf.pbc.scf
import pyscf.pbc.df

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
ecc = vayesta.ewf.EWF(kmf, bno_threshold=1e-3)
ecc.iao_fragmentation()
ecc.add_atomic_fragment(0)
ecc.kernel()

# Hartree-Fock with supercell
scell = pyscf.pbc.tools.super_cell(cell, [2,2,2])
mf = pyscf.pbc.scf.RHF(scell)
mf = mf.density_fit(auxbasis='def2-svp-ri')
mf.kernel()

ecc_sc = vayesta.ewf.EWF(mf, bno_threshold=1e-3)
ecc_sc.iao_fragmentation()
ecc_sc.add_atomic_fragment(0, sym_factor=8)
ecc_sc.kernel()

print("k-point (k) and supercell (sc) energies:")
print("E(k-HF)=             %+16.8f Ha" % (kmf.e_tot))
print("E(sc-HF)=            %+16.8f Ha" % (mf.e_tot/8))
print("E(EWF-CCSD@k-HF)=    %+16.8f Ha" % (ecc.e_tot))
print("E(EWF-CCSD@sc-HF)=   %+16.8f Ha" % (ecc_sc.e_tot/8))
