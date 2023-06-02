import numpy as np

import pyscf
import pyscf.pbc
import pyscf.pbc.gto
import pyscf.pbc.scf
import pyscf.pbc.cc
import pyscf.pbc.tools

import vayesta
import vayesta.ewf

cell = pyscf.pbc.gto.Cell()
cell.atom = "He 0 0 0"
cell.a = 1.4 * np.eye(3)
cell.basis = '6-31g'
cell.verbose = 10
cell.output = 'pyscf.out'
cell.build()

cell = pyscf.pbc.tools.super_cell(cell, [2,2,2])

# Hartree-Fock
mf = pyscf.pbc.scf.UHF(cell)
mf.kernel()

# Reference full system CCSD:
cc = pyscf.pbc.cc.UCCSD(mf)
cc.kernel()

# Test exact limit using bath_type='full'
ecc = vayesta.ewf.EWF(mf, bath_type='full')
with ecc.iao_fragmentation() as f:
    f.add_all_atomic_fragments()
ecc.kernel()

nocca = np.count_nonzero(mf.mo_occ[0]>0)
noccb = np.count_nonzero(mf.mo_occ[1]>0)
nocc = nocca + noccb
e_exxdiv = -nocc*pyscf.pbc.tools.madelung(cell, kpts=np.zeros((3,))) / 2
print("E(exx-div)=  %+16.8f Ha" % e_exxdiv)

print("E(HF)=       %+16.8f Ha" % mf.e_tot)
print("E(CCSD)=     %+16.8f Ha" % cc.e_tot )
print("E(EWF-CCSD)= %+16.8f Ha" % ecc.e_tot)

assert np.allclose(cc.e_tot, ecc.e_tot)
