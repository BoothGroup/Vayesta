import numpy as np

import pyscf
import pyscf.gto
import pyscf.scf
import pyscf.mp
import pyscf.cc

import vayesta
import vayesta.ewf

mol = pyscf.gto.Mole()
mol.atom = """
O  0.0000   0.0000   0.1173
H  0.0000   0.7572  -0.4692
H  0.0000  -0.7572  -0.4692
"""
mol.basis = 'cc-pVDZ'
mol.output = 'pyscf.out'
mol.build()

# Hartree-Fock
mf = pyscf.scf.RHF(mol)
# Intercluster MP2 energy requries density fitting!
mf = mf.density_fit(auxbasis='cc-pVDZ-ri')
mf.kernel()

# Reference full system calculations:
mp2 = pyscf.mp.MP2(mf)
mp2.kernel()

cc = pyscf.cc.CCSD(mf)
cc.kernel()

# Embedded CCSD
emb = vayesta.ewf.EWF(mf, bath_options=dict(threshold=1e-4))
emb.kernel()

# Inter-cluster MP2 correction:
e_icmp2 = emb.get_intercluster_mp2_energy()

print('E(HF)=                 %+16.8f Ha' % mf.e_tot)
print('E(CCSD)=               %+16.8f Ha' % cc.e_tot)
print('E(MP2)=                %+16.8f Ha  (error=  %+12.8f Ha)' % (mp2.e_tot, mp2.e_tot - cc.e_tot))
print('E(Emb. CCSD)=          %+16.8f Ha  (error=  %+12.8f Ha)' % (emb.e_tot, emb.e_tot - cc.e_tot))
print('E(Emb. CCSD) + ICMP2=  %+16.8f Ha  (error=  %+12.8f Ha)' % (emb.e_tot+e_icmp2, emb.e_tot+e_icmp2-cc.e_tot))
