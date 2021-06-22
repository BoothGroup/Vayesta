import numpy as np

import pyscf
import pyscf.pbc
import pyscf.pbc.gto
import pyscf.pbc.scf

import vayesta
import vayesta.ewf
from vayesta.misc import molstructs

cell = pyscf.pbc.gto.Cell()
cell.a, cell.atom = molstructs.diamond()

cell.basis = 'cc-pvdz'
cell.verbose = 10
cell.output = 'pyscf_out.txt'
cell.build()

kpts = cell.make_kpts([2,2,2])

# Hartree-Fock
mf = pyscf.pbc.scf.KRHF(cell, kpts)
mf = mf.density_fit(auxbasis='cc-pvdz-ri')
mf.kernel()

# EWF-CCSD
ecc = vayesta.ewf.EWF(mf, bno_threshold=1e-5)
ecc.make_atom_fragment(0)
ecc.kernel()

# Tailored EWF-CCSD
etcc = vayesta.ewf.EWF(mf, solver='TCCSD', bno_threshold=1e-5)
etcc.make_atom_fragment(0)
etcc.kernel()

print("E%-14s %+16.8f Ha" % ('(HF)=', mf.e_tot))
print("E%-14s %+16.8f Ha" % ('(EWF-CCSD)=', ecc.e_tot))
print("E%-14s %+16.8f Ha" % ('(EWF-TCCSD)=', etcc.e_tot))
