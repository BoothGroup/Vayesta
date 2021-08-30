
import pyscf.scf

import vayesta.dmet

mol = pyscf.gto.Mole()
mol.atom = ['Li 0.0 0.0 0.0', 'H 0.0, 0.0, 1.4']
mol.basis = 'cc-pvdz'
mol.verbose = 10
mol.output = 'pyscf_out.txt'
mol.build()

# Hartree-Fock
mf = pyscf.scf.RHF(mol)
mf.kernel()

dmet = vayesta.dmet.DMET(mf)
dmet.make_atom_fragment(0)
dmet.make_atom_fragment(1)
# Alternative: dmet.make_all_atom_fragments()
dmet.kernel()

print("E%-11s %+16.8f Ha" % ('(HF)=', mf.e_tot))
print("E%-11s %+16.8f Ha" % ('(DMET)=', dmet.e_tot))
