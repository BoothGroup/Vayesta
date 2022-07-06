import pyscf.gto
import pyscf.scf
import vayesta
import vayesta.dmet


mol = pyscf.gto.Mole()
mol.atom = 'Li 0 0 0 ; H 0 0 1.4'
mol.basis = 'cc-pVDZ'
mol.output = 'pyscf.out'
mol.build()

# Hartree-Fock
mf = pyscf.scf.RHF(mol)
mf.kernel()

dmet = vayesta.dmet.DMET(mf, solver='FCI')
with dmet.iao_fragmentation() as f:
    f.add_atomic_fragment(0)
    f.add_atomic_fragment(1)
dmet.kernel()

print("E(HF)=   %+16.8f Ha" % mf.e_tot)
print("E(DMET)= %+16.8f Ha" % dmet.e_tot)
