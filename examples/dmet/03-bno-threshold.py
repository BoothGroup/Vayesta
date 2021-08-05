import pyscf.scf

import vayesta.dmet

mol = pyscf.gto.Mole()
mol.atom = ['Li 0.0 0.0 0.0', 'H 0.0, 0.0, 1.4']
mol.basis = 'aug-cc-pvdz'
mol.verbose = 10
mol.output = 'pyscf_out.txt'
mol.build()

# Hartree-Fock
mf = pyscf.scf.RHF(mol)
mf.kernel()
# Can set threshold at initialisation
dmet1 = vayesta.dmet.DMET(mf, bno_threshold = 1e-2)
dmet1.make_all_atom_fragments()
dmet1.kernel()
# Can also set at runtime.
dmet2 = vayesta.dmet.DMET(mf)
dmet2.make_all_atom_fragments()
dmet2.kernel(bno_threshold = 1e-4)

print("DMET energies at different bno thresholds: 1e-2={:6.4e}, 1e-4={:6.4e}".format(dmet1.e_dmet, dmet2.e_dmet))
