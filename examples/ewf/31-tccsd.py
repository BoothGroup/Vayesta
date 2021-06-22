import pyscf
import pyscf.gto
import pyscf.scf
import pyscf.cc

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

# Reference full system CCSD:
cc = pyscf.cc.CCSD(mf)
cc.kernel()

# Reference one-shot EWF-CCSD
ecc = vayesta.ewf.EWF(mf, bno_threshold=1e-4)
ecc.make_atom_fragment(0, sym_factor=2)
ecc.kernel()

etcc = vayesta.ewf.EWF(mf, solver='TCCSD', bno_threshold=1e-4)
etcc.make_atom_fragment(0, sym_factor=2)
etcc.kernel()

print("E%-14s %+16.8f Ha" % ('(HF)=', mf.e_tot))
print("E%-14s %+16.8f Ha" % ('(CCSD)=', cc.e_tot))
print("E%-14s %+16.8f Ha" % ('(EWF-CCSD)=', ecc.e_tot))
print("E%-14s %+16.8f Ha" % ('(EWF-TCCSD)=', etcc.e_tot))
