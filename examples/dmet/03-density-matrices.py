import numpy as np
import pyscf.gto
import pyscf.scf
import vayesta
import vayesta.dmet
from vayesta.misc.molecules import ring

# H6 ring
mol = pyscf.gto.Mole()
mol.atom = ring(atom='H', natom=6, bond_length=2.0)
mol.basis = 'sto-6g'
mol.output = 'pyscf.out'
mol.build()

# Hartree-Fock
mf = pyscf.scf.RHF(mol)
mf.kernel()

# One-shot DMET
dmet = vayesta.dmet.DMET(mf, solver='FCI', maxiter=1)
with dmet.sao_fragmentation() as f:
    f.add_atomic_fragment([0,1])
    f.add_atomic_fragment([2,3])
    f.add_atomic_fragment([4,5])
dmet.kernel()

# Calculate energy from democratically-partitioned density matrices:
dm1 = dmet.make_rdm1(ao_basis=True)
dm2 = dmet.make_rdm2(ao_basis=True)
# One and two-electron integrals in AO basis:
h1e = dmet.get_hcore()
eris = dmet.get_eris_array(np.eye(mol.nao))
e_dmet = dmet.e_nuc + np.einsum('ij,ij->', h1e, dm1) + np.einsum('ijkl,ijkl->', eris, dm2)/2

print("Energies")
print("========")
print("  HF:                %+16.8f Ha" % mf.e_tot)
print("  DMET:              %+16.8f Ha" % dmet.e_tot)
print("  DMET (from DMs):   %+16.8f Ha" % e_dmet)
