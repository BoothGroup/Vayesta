import numpy as np
import pyscf
import pyscf.pbc
import pyscf.pbc.scf
import vayesta
import vayesta.ewf


cell = pyscf.pbc.gto.Cell()
cell.atom = ["He 0.0 0.0 0.0"]
cell.a = 1.4 * np.eye(3)
cell.basis = "def2-svp"
cell.output = "pyscf.out"
cell.space_group_symmetry = True
cell.symmorphic = True
cell.build()
# Enforce all symmetries in kpoint generation.
kpts = cell.make_kpts([2, 2, 2], space_group_symmetry=True, time_reversal_symmetry=True, symmorphic=True)

# Hartree-Fock with k-points
mf = pyscf.pbc.scf.KRHF(cell, kpts)
mf = mf.density_fit(auxbasis="def2-svp-ri")
mf.kernel()

# Embedded calculation will automatically unfold the k-point sampled mean-field
emb = vayesta.ewf.EWF(mf, bath_options=dict(threshold=1e-4))
# If calling the kernel without initializiation fragmentation and fragments,
# IAO fragmentation and atomic fragments are used automatically.
emb.kernel()

# Some versions of pyscf may encounter an error when generating IAOs with space_group_symmetry=True.
# To avoid this, either update to a version more recent that 2.3, or turn off space group symmetry before generating
# IAOs, as in the lines below.
# emb.mf.mol.space_group_symmetry = False
# emb.kernel()

print("E(HF)=        %+16.8f Ha" % (mf.e_tot))
print("E(Emb. CCSD)= %+16.8f Ha" % (emb.e_tot))
