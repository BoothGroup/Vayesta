import numpy as np

import pyscf
import pyscf.gto
import pyscf.scf
import pyscf.mcscf

import vayesta
import vayesta.ewf

mol = pyscf.gto.Mole()
mol.atom = ['N 0.0 0.0 0.0', 'N 0.0, 0.0, 2']
mol.basis = 'aug-cc-pvdz'
mol.verbose = 10
mol.output = 'pyscf_out.txt'
mol.build()

# Hartree-Fock
mf = pyscf.scf.RHF(mol)
mf.kernel()

ecc = vayesta.ewf.EWF(mf, solver='FCI')
f1 = ecc.make_atom_fragment(0)
f2 = ecc.make_atom_fragment(1)
results1 = f1.kernel(np.inf)
results2 = f2.kernel(np.inf)

# results have attributes 'c0', 'c1', 'c2'
# 1) Get intermediately normalized c1, c2
c1_1 = results1.c1 / results1.c0
c2_1 = results1.c2 / results1.c0
c1_2 = results2.c1 / results2.c0
c2_2 = results2.c2 / results2.c0

# 2) Get fragment projector
p1 = f1.get_fragment_projector(f1.c_active_occ)
p2 = f1.get_fragment_projector(f2.c_active_occ)

# 3) Project c1 and c2 in first occupied index (apply p1 / p2)


# 4) Transform each c1, c2 to a full, common basis (HF MO basis)


# 5) Combine (add) to a single c1, c2


# 6) Use full c1, c2 to tailor a CCSD calculation
ecc = vayesta.ewf.EWF(mf, solver='CCSD', bno_threshold=-1)
ecc.make_atom_fragment([0, 1])
ecc.kernel()

print("E%-11s %+16.8f Ha" % ('(HF)=', mf.e_tot))
print("E%-11s %+16.8f Ha" % ('(EWF-CCSD)=', ecc.e_tot))
