import numpy as np
import pyscf
import pyscf.gto
import pyscf.scf
import pyscf.cc
import vayesta
import vayesta.ewf
from vayesta.misc import molecules

mol = pyscf.gto.Mole()
mol.atom = molecules.ring('H', 6, 1.0)
shift = np.random.rand(3)
mol.atom = [(x[0], x[1]+shift) for x in mol.atom]
mol.basis = 'cc-pVTZ'
mol.output = 'pyscf.out'
mol.build()

# Hartree-Fock
mf = pyscf.scf.RHF(mol)
mf.kernel()

# Embedded CCSD
emb_sym = vayesta.ewf.EWF(mf, bath_options=dict(threshold=1e-6), solver_options=dict(solve_lambda=True))
emb_sym.symmetry.add_rotation(order=6, axis=[0,0,1], center=shift)
with emb_sym.iao_fragmentation() as frag:
    frag.add_atomic_fragment(0)
emb_sym.kernel()

# Embedded CCSD
emb = vayesta.ewf.EWF(mf, bath_options=dict(threshold=1e-6), solver_options=dict(solve_lambda=True))
emb.kernel()

print("E(witout symmetry)= %+16.8f Ha" % emb.get_dm_energy())
print("E(with   symmetry)= %+16.8f Ha" % emb_sym.get_dm_energy())
