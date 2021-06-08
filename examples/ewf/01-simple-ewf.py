import pyscf
import pyscf.gto
import pyscf.scf
import pyscf.cc

import vayesta
import vayesta.ewf

mol = pyscf.gto.Mole()
mol.atom = ['Li 0.0 0.0 0.0', 'H 0.0, 0.0, 1.4']
mol.basis = 'aug-cc-pvdz'
mol.verbose = 10
mol.output = 'pyscf_out.txt'
mol.build()

# Hartree-Fock
mf = pyscf.scf.RHF(mol)
mf.kernel()

ecc = vayesta.ewf.EWF(mf, bno_threshold=1e-6)
ecc.make_atom_fragment(0)
ecc.make_atom_fragment(1)
# Alternative: ecc.make_all_atom_fragments()
ecc.kernel()

print("E%-11s %+16.8f Ha" % ('(HF)=', mf.e_tot))
print("E%-11s %+16.8f Ha" % ('(EWF-CCSD)=', ecc.e_tot))
