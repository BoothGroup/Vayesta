import numpy as np
import pyscf
import pyscf.gto
import pyscf.scf
import vayesta
import vayesta.ewf
from vayesta.misc import molecules

mol = pyscf.gto.Mole()
mol.atom = molecules.ring('C', 6, 1.0)
mol.basis = 'cc-pVDZ'
mol.output = 'pyscf.out'
mol.build()

# Hartree-Fock
mf = pyscf.scf.RHF(mol)
mf.kernel()

# Embedded CCSD with rotational symmetry:
emb_sym = vayesta.ewf.EWF(mf, bath_options=dict(threshold=1e-4), solver_options=dict(solve_lambda=True))

# Add rotational symmetry:
# Set order of rotation (2: 180 degrees, 3: 120 degrees, 4: 90: degrees,...),
# axis along which to rotate and center of rotation (default units for axis and center are Angstroms):
emb_sym.symmetry.add_rotation(order=6, axis=[0,0,1], center=[0,0,0])

# Do not call add_all_atomic_fragments, as it add the symmetry related atoms as fragments!
with emb_sym.iao_fragmentation() as frag:
    frag.add_atomic_fragment(0)
emb_sym.kernel()

# Reference calculation without rotational symmetry:
emb = vayesta.ewf.EWF(mf, bath_options=dict(threshold=1e-4), solver_options=dict(solve_lambda=True))
emb.kernel()

# Compare energies and density matrices:
print("E(without symmetry)= %+16.8f Ha" % emb.get_dm_energy())
print("E(with symmetry)=    %+16.8f Ha" % emb_sym.get_dm_energy())
dm1 = emb.make_rdm1()
dm1_sym = emb_sym.make_rdm1()
print("Difference in 1-RDM= %.3e" % np.linalg.norm(dm1 - dm1_sym))
