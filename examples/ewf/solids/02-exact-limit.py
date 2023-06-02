import numpy as np

import pyscf
import pyscf.pbc
import pyscf.pbc.gto
import pyscf.pbc.scf
import pyscf.pbc.cc
import pyscf.pbc.tools

import vayesta
import vayesta.ewf

mol = pyscf.pbc.gto.Cell()
mol.atom = "He 0 0 0"
mol.a = 1.4 * np.eye(3)
mol.basis = '6-31g'
mol.verbose = 10
mol.output = 'pyscf_out.txt'
mol.build()

# Hartree-Fock
mf = pyscf.pbc.scf.RHF(mol)
mf.kernel()

# Reference full system CCSD:
cc = pyscf.pbc.cc.CCSD(mf)
cc.kernel()

# Test exact limit using bno_threshold = -1
ecc = vayesta.ewf.EWF(mf, bno_threshold=-1)
with ecc.iao_fragmentation() as f:
    f.add_all_atomic_fragments()
ecc.kernel()

nocc = np.count_nonzero(mf.mo_occ>0)
e_exxdiv = -nocc*pyscf.pbc.tools.madelung(mol, kpts=np.zeros((3,)))
print("E(exx-div)=  %+16.8f Ha" % e_exxdiv)

print("E(HF)=       %+16.8f Ha" % mf.e_tot)
print("E(CCSD)=     %+16.8f Ha" % (cc.e_tot + e_exxdiv))
print("E(EWF-CCSD)= %+16.8f Ha" % ecc.e_tot)

assert np.allclose(cc.e_tot + e_exxdiv, ecc.e_tot)
