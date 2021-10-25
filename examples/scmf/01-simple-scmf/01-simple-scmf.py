import numpy as np

import pyscf
import pyscf.gto
import pyscf.scf
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
mol.build()

# Hartree-Fock
mf = pyscf.scf.RHF(mol)
mf.kernel()

# CCSD
cc = pyscf.cc.CCSD(mf)
cc.kernel()

ecc = vayesta.ewf.EWF(mf, make_rdm1=True, bno_threshold=1e-4)
ecc.sao_fragmentation()
ecc.add_all_atomic_fragments()
ecc.pdmet_scmf()
# or:
#ecc.brueckner_scmf()
ecc.kernel()
assert ecc.with_scmf.converged

print("E(HF)=           %+16.8f Ha" % mf.e_tot)
print("E(CCSD)=         %+16.8f Ha" % cc.e_tot)
print("E(E-CCSD)=       %+16.8f Ha" % ecc.with_scmf.e_tot_oneshot)
print("E(sc-E-CCSD)=    %+16.8f Ha" % ecc.e_tot)
