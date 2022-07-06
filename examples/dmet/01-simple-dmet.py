import pyscf.gto
import pyscf.scf
import pyscf.fci
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

# Reference FCI
fci = pyscf.fci.FCI(mf)
fci.kernel()

# Oneshot DMET
dmet = vayesta.dmet.DMET(mf, solver='FCI', maxiter=1)
with dmet.sao_fragmentation() as f:
    f.add_atomic_fragment([0,1])
    f.add_atomic_fragment([2,3])
    f.add_atomic_fragment([4,5])
dmet.kernel()

# Self-consistent DMET
dmet_sc = vayesta.dmet.DMET(mf, solver='FCI')
with dmet_sc.sao_fragmentation() as f:
    f.add_atomic_fragment([0,1])
    f.add_atomic_fragment([2,3])
    f.add_atomic_fragment([4,5])
dmet_sc.kernel()

print("Energies")
print("========")
print("  HF:                   %+16.8f Ha" % mf.e_tot)
print("  FCI:                  %+16.8f Ha" % fci.e_tot)
print("  oneshot DMET:         %+16.8f Ha  (error= %.1f mHa)" % (dmet.e_tot, 1000*(dmet.e_tot-fci.e_tot)))
print("  self-consistent DMET: %+16.8f Ha  (error= %.1f mHa)" % (dmet_sc.e_tot, 1000*(dmet_sc.e_tot-fci.e_tot)))
