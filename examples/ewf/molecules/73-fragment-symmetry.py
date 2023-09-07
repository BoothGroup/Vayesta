import numpy as np
import pyscf
import pyscf.gto
import pyscf.scf
import vayesta
import vayesta.ewf
from vayesta.misc import molecules


mol = pyscf.gto.Mole()
mol.atom = molecules.ferrocene(numbering=True)
mol.basis = "STO-6G"
mol.output = "pyscf.txt"
mol.build()

# Hartree-Fock
mf = pyscf.scf.RHF(mol)
mf.kernel()

# Embedded CCSD with symmetry:
emb_sym = vayesta.ewf.EWF(mf, solver="MP2", bath_options=dict(threshold=1e-4))
# Do not call add_all_atomic_fragments, as it add the symmetry related atoms as fragments!
with emb_sym.iao_fragmentation() as frag:
    frag.add_atomic_fragment("Fe1")
    # Add mirror (reflection) symmetry
    # Set axis of reflection [tuple(3) or 'x', 'y', 'z']
    with frag.mirror_symmetry(axis="z"):
        # Add rotational symmetry
        # Set order of rotation (2: 180 degrees, 3: 120 degrees, 4: 90: degrees,...),
        # axis along which to rotate and center of rotation (default units for axis and center are Angstroms):
        with frag.rotational_symmetry(order=5, axis=[0, 0, 1]):
            frag.add_atomic_fragment("C2")
            frag.add_atomic_fragment("H7")

emb_sym.kernel()
dm1_sym = emb_sym.make_rdm1()

# Reference calculation without symmetry:
emb = vayesta.ewf.EWF(mf, solver="MP2", bath_options=dict(threshold=1e-4))
emb.kernel()
dm1 = emb.make_rdm1()

# Compare energies and density matrices:
print("E(without symmetry)= %+16.8f Ha" % emb.get_dm_energy())
print("E(with symmetry)=    %+16.8f Ha" % emb_sym.get_dm_energy())
print("Difference in 1-RDM= %.3e" % np.linalg.norm(dm1 - dm1_sym))
