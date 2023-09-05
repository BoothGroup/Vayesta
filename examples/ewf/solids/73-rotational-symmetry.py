import numpy as np
import pyscf
import pyscf.pbc
import pyscf.pbc.scf
import vayesta
import vayesta.ewf

cell = pyscf.pbc.gto.Cell()
cell.atom = "Li 0 0 0; Li 1 0 0; Li 0 1 0; Li 1 1 0"
cell.a = 8 * np.eye(3)
cell.basis = "sto-6g"
cell.exp_to_discard = 0.1
cell.output = "pyscf.out"
cell.build()

# Hartree-Fock with k-points
kpts = cell.make_kpts([2, 2, 2])
mf = pyscf.pbc.scf.KRHF(cell, kpts)
mf = mf.density_fit(auxbasis="sto-6g")
mf.kernel()

# Embedded CCSD with rotational symmetry:
emb_sym = vayesta.ewf.EWF(mf, bath_options=dict(threshold=1e-4), solver_options=dict(solve_lambda=True))

# Add rotational symmetry:
# Set order of rotation (2: 180 degrees, 3: 120 degrees, 4: 90: degrees,...),
# axis along which to rotate and center of rotation (default units for axis and center are Angstroms):
# emb_sym.symmetry.add_rotation(order=4, axis=[0,0,1], center=[0.5,0.5,0])
# It's also possible to use units of Bohr or lattice vectors:
# emb_sym.symmetry.add_rotation(order=4, axis=[0,0,1], center=[0.5/4,0.5/4,0], unit='latvec')

# Do not call add_all_atomic_fragments, as it add the symmetry related atoms as fragments!
with emb_sym.iao_fragmentation() as frag:
    with frag.rotational_symmetry(order=4, axis=[0, 0, 1], center=[0.5, 0.5, 0]):
        f = frag.add_atomic_fragment(0)
emb_sym.kernel()

# Reference calculation without rotational symmetry:
emb = vayesta.ewf.EWF(mf, bath_options=dict(threshold=1e-4), solver_options=dict(solve_lambda=True))
emb.kernel()

# Compare energies and density matrices:
print("E(Without symmetry)= %+16.8f Ha" % (emb.get_dm_energy()))
print("E(With symmetry)=    %+16.8f Ha" % (emb_sym.get_dm_energy()))
dm1 = emb.make_rdm1(ao_basis=True)
dm1_sym = emb_sym.make_rdm1(ao_basis=True)
print("Difference in 1-RDM= %.3e" % np.linalg.norm(dm1 - dm1_sym))
