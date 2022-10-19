import numpy as np
import pyscf
import pyscf.gto
import pyscf.scf
import vayesta
import vayesta.ewf
from vayesta.misc import molecules


mol = pyscf.gto.Mole()
mol.atom = """
Fe1         0.0000000   0.0000000       0.0000000
C2          0.0000000   1.3557290       1.8934930
C3          1.2893750   0.4189430       1.8934930
C4          0.7968780   -1.0968080      1.8934930
C5          -0.7968780  -1.0968080      1.8934930
C6          -1.2893750  0.4189430       1.8934930
C7          0.0000000   1.3557290       -1.8934930
C8          1.2893750   0.4189430       -1.8934930
C9          0.7968780   -1.0968080      -1.8934930
C10         -0.7968780  -1.0968080      -1.8934930
C11         -1.2893750  0.4189430       -1.8934930
H12         0.0000000   2.4095720       1.7452730
H13         2.2916400   0.7445990       1.7452730
H14         1.4163110   -1.9493850      1.7452730
H15         -1.4163110  -1.9493850      1.7452730
H16         -2.2916400  0.7445990       1.7452730
H17         0.0000000   2.4095720       -1.7452730
H18         2.2916400   0.7445990       -1.7452730
H19         1.4163110   -1.9493850      -1.7452730
H20         -1.4163110  -1.9493850      -1.7452730
H21         -2.2916400  0.7445990       -1.7452730
"""
mol.basis = 'STO-6G'
mol.output = 'pyscf.txt'
mol.build()

# Hartree-Fock
mf = pyscf.scf.RHF(mol)
mf.kernel()

# Embedded CCSD with rotational symmetry:
emb_sym = vayesta.ewf.EWF(mf, solver='MP2', bath_options=dict(threshold=1e-4), symmetry_tol=1e-5)
# Do not call add_all_atomic_fragments, as it add the symmetry related atoms as fragments!
with emb_sym.iao_fragmentation() as frag:
    # Add rotational symmetry:
    # Set order of rotation (2: 180 degrees, 3: 120 degrees, 4: 90: degrees,...),
    # axis along which to rotate and center of rotation (default units for axis and center are Angstroms):
    #with frag.rotational_symmetry(order=6, axis=(0,0,1), center=(0,0,0)):
    #    frag.add_atomic_fragment(0)
    frag.add_atomic_fragment('Fe1')
    with frag.rotational_symmetry(order=5, axis=(0,0,1), center=(0,0,0)):
        frag.add_atomic_fragment('C2')
        frag.add_atomic_fragment('C7')
        frag.add_atomic_fragment('H12')
        frag.add_atomic_fragment('H17')

emb_sym.kernel()
dm1_sym = emb_sym.make_rdm1()

# Reference calculation without rotational symmetry:
emb = vayesta.ewf.EWF(mf, solver='MP2', bath_options=dict(threshold=1e-4))
emb.kernel()
dm1 = emb.make_rdm1()

# Compare energies and density matrices:
print("E(without symmetry)= %+16.8f Ha" % emb.get_dm_energy())
print("E(with symmetry)=    %+16.8f Ha" % emb_sym.get_dm_energy())
print("Difference in 1-RDM= %.3e" % np.linalg.norm(dm1 - dm1_sym))
