import numpy as np

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

# Reference full system CCSD:
cc = pyscf.cc.CCSD(mf)
cc.kernel()

# Test exact limit using bno_threshold = -1
ecc = vayesta.ewf.EWF(mf, bno_threshold=-1)
ecc.make_all_atom_fragments()
ecc.kernel()

print("E%-11s %+16.8f Ha" % ('(HF)=', mf.e_tot))
print("E%-11s %+16.8f Ha" % ('(CCSD)=', cc.e_tot))
print("E%-11s %+16.8f Ha" % ('(EWF-CCSD)=', ecc.e_tot))

assert np.allclose(cc.e_tot, ecc.e_tot)
