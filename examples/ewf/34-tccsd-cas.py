import pyscf
import pyscf.gto
import pyscf.scf

import vayesta
import vayesta.ewf

mol = pyscf.gto.Mole()
mol.atom = ['N 0.0 0.0 0.0', 'N 0.0, 0.0, 1.1']
mol.basis = 'cc-pvdz'
mol.verbose = 10
mol.output = 'pyscf_out.txt'
mol.build()

# Hartree-Fock
mf = pyscf.scf.RHF(mol)
mf.kernel()

# Automatic CAS (fragment + DMET orbitals)
ecc = vayesta.ewf.EWF(mf, solver='CCSD', bno_threshold=1e-4)
f = ecc.make_atom_fragment(0, sym_factor=2)
ecc.kernel()

# Automatic CAS (fragment + DMET orbitals)
etcc = vayesta.ewf.EWF(mf, solver='TCCSD', bno_threshold=1e-4)
f = etcc.make_atom_fragment(0, sym_factor=2)
etcc.kernel()

# Custom CAS
etcc2 = vayesta.ewf.EWF(mf, solver='TCCSD', bno_threshold=1e-4)
f = etcc2.make_atom_fragment(0, sym_factor=2)
f.set_cas(iaos=['0 N 2p'])
etcc2.kernel()

print("E%-24s %+16.8f Ha" % ('(HF)=', mf.e_tot))
print("E%-24s %+16.8f Ha" % ('(EWF-CCSD)=', ecc.e_tot))
print("E%-24s %+16.8f Ha" % ('(EWF-TCCSD)(DMET CAS)=', etcc.e_tot))
print("E%-24s %+16.8f Ha" % ('(EWF-TCCSD)(2p CAS)  =', etcc2.e_tot))
