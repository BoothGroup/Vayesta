import pyscf
import pyscf.gto
import pyscf.scf
import pyscf.cc

import vayesta
import vayesta.ewf

mol = pyscf.gto.Mole()
mol.atom = """
Se	0.0000	0.0000	0.2807
O 	0.0000	1.3464	-0.5965
O 	0.0000	-1.3464	-0.5965
"""
mol.basis = 'cc-pVDZ'
mol.output = 'pyscf.out'
mol.build()

# Hartree-Fock
mf = pyscf.scf.RHF(mol)
mf.kernel()

# Embedded CCSD
emb = vayesta.ewf.EWF(mf, bno_threshold=1e-6)
# In order to define custom fragments, the fragmentation always needs to be
# initialized manually:
emb.iao_fragmentation()
# Fragment containing the 1s state of O and 1s and 2s states of Se
emb.add_atomic_fragment(['Se', 'O'], orbital_filter=['1s', 'Se 2s'])
# Atoms can be specified by labels or indices
# Fragment containing the 2s state at O and 3s and 4s states of Se
emb.add_atomic_fragment([0, 1, 2], orbital_filter=['O 2s', '3s', '4s'])
# Fragment containing the 2p x- and y-states on the oxygen and the 2p, 3p y- and z- states on selenium
# Note that the oxygen does not have 3p IAO states, such that it is not necessary to specify the element for these states
emb.add_atomic_fragment([0, 1, 2], orbital_filter=['O 2py', 'O 2pz', 'Se 2p', '3py', '3pz'])
# All 4p states on Se and all px states (2px on O, 2-4px on Se)
emb.add_atomic_fragment(['Se', 'O'], orbital_filter=['4p', 'px'])
# 3d states on Se
emb.add_atomic_fragment(0, orbital_filter=['3d'])
emb.kernel()

# Reference full system CCSD:
cc = pyscf.cc.CCSD(mf)
cc.kernel()

print("E(HF)=        %+16.8f Ha" % mf.e_tot)
print("E(Emb. CCSD)= %+16.8f Ha" % emb.e_tot)
print("E(CCSD)=      %+16.8f Ha" % cc.e_tot)
