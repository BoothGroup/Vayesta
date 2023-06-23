import numpy as np
import pyscf
import pyscf.gto
import pyscf.scf
import vayesta
import vayesta.ewf


mol = pyscf.gto.Mole()
mol.atom = """
C1   0.7854  0.7854  0.7854
C2  -0.7854  0.7854  0.7854
C3   0.7854  0.7854 -0.7854
C4  -0.7854  0.7854 -0.7854
C5   0.7854 -0.7854  0.7854
C6  -0.7854 -0.7854  0.7854
C7   0.7854 -0.7854 -0.7854
C8  -0.7854 -0.7854 -0.7854
H9   1.4188  1.4188  1.4188
H10 -1.4188  1.4188  1.4188
H11  1.4188  1.4188 -1.4188
H12 -1.4188  1.4188 -1.4188
H13  1.4188 -1.4188  1.4188
H14 -1.4188 -1.4188  1.4188
H15  1.4188 -1.4188 -1.4188
H16 -1.4188 -1.4188 -1.4188
"""
mol.basis = 'cc-pVDZ'
mol.output = 'pyscf.out'
mol.build()

# Hartree-Fock
mf = pyscf.scf.RHF(mol)
mf.kernel()

# Embedded CCSD with rotational symmetry:
emb_sym = vayesta.ewf.EWF(mf, solver='MP2', bath_options=dict(threshold=1e-4))

# Do not call add_all_atomic_fragments, as it add the symmetry related atoms as fragments!
with emb_sym.iao_fragmentation() as frag:
    # Add rotational symmetry:
    # Set order of rotation [order = 360/degrees, i.e. order 2: 180 degrees, order 3: 120 degrees,...],
    # axis along which to rotate [tuple(3) or 'x', 'y', 'z'],
    # center of rotation [tuple(3), default= (0,0,0)],
    # and unit for the axis and center [default= 'angstrom'].
    # Important when using multiple rotations:
    # Only use the minimal set of rotations which generate all atomic fragments,
    # i.e. do not add multiple 4th-order rotations here!
    with frag.rotational_symmetry(order=4, axis=[0,0,1]):
        with frag.rotational_symmetry(order=2, axis=[0,1,0]):
            frag.add_atomic_fragment('C1')
            frag.add_atomic_fragment('H9')
emb_sym.kernel()

# Reference calculation without rotational symmetry:
emb = vayesta.ewf.EWF(mf, solver='MP2', bath_options=dict(threshold=1e-4))
emb.kernel()

# Compare energies and density matrices:
print("E(without symmetry)= %+16.8f Ha" % emb.get_dm_energy())
print("E(with symmetry)=    %+16.8f Ha" % emb_sym.get_dm_energy())
dm1 = emb.make_rdm1()
dm1_sym = emb_sym.make_rdm1()
print("Difference in 1-RDM= %.3e" % np.linalg.norm(dm1 - dm1_sym))

dndn_sym = emb_sym.get_corrfunc('dN,dN')
dndn = emb.get_corrfunc('dN,dN')
print("Difference in <dN,dN>= %.3e" % np.linalg.norm(dndn - dndn_sym))

szsz_sym = emb_sym.get_corrfunc('Sz,Sz')
szsz = emb.get_corrfunc('Sz,Sz')
print("Difference in <Sz,Sz>= %.3e" % np.linalg.norm(szsz - szsz_sym))
