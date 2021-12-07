import pyscf
import pyscf.gto
import pyscf.scf

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

# At the moment, only the energies are printed
ecc = vayesta.ewf.EWF(mf, bno_threshold=1e-6,
        eom_ccsd=['IP', 'EA', 'EE-S', 'EE-T', 'EE-SF'],     # Default is []
        eom_ccsd_nroots=6)                                  # Default is 5
ecc.add_all_atomic_fragments()
ecc.kernel()
