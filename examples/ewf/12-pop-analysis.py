import pyscf
import pyscf.gto
import pyscf.scf

import vayesta
import vayesta.ewf

mol = pyscf.gto.Mole()
mol.atom = ['H 0.0 0.0 0.0', 'F 0.0, 0.0, 2.0']
mol.basis = 'aug-cc-pvdz'
mol.verbose = 10
mol.output = 'pyscf_out.txt'
mol.build()

# Hartree-Fock
mf = pyscf.scf.RHF(mol)
mf.kernel()

# Activate population analysis for all fragments and write to file:
ecc = vayesta.ewf.EWF(mf, pop_analysis='population.txt')
#ecc.make_all_atom_fragments()
ecc.make_atom_fragment([0,1])
ecc.kernel(bno_threshold=-1)

# Activate population analysis for all fragments and write to log:
ecc = vayesta.ewf.EWF(mf, pop_analysis=True)
ecc.make_all_atom_fragments()
# Run as:
#ecc.kernel(bno_threshold=1e-6)

# Activate population analysis for Li cluster only and write to log:
ecc = vayesta.ewf.EWF(mf)
ecc.make_atom_fragment(0, pop_analysis=True)
ecc.make_atom_fragment(1)
# Run as:
#ecc.kernel(bno_threshold=1e-6)
