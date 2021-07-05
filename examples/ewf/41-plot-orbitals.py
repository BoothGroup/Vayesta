import numpy as np

import pyscf
import pyscf.pbc
import pyscf.pbc.scf

import vayesta
import vayesta.ewf
from vayesta.misc.molstructs import graphene

cell = pyscf.pbc.gto.Cell()
cell.a, cell.atom = graphene(atoms=['B', 'N'])
cell.dimension = 2
cell.basis = 'def2-svp'
cell.verbose = 10
cell.output = 'pyscf_out.txt'
cell.build()

kpts = cell.make_kpts([2,2,1])

# Hartree-Fock with k-points
kmf = pyscf.pbc.scf.KRHF(cell, kpts)
kmf = kmf.density_fit(auxbasis='def2-svp-ri')
kmf.kernel()

# Embedded calculation will automatically unfold the k-point sampled mean-field
ecc = vayesta.ewf.EWF(kmf, plot_orbitals='dmet-exit')
ecc.make_atom_fragment(0)
ecc.make_atom_fragment(1)
ecc.kernel()

print("E%-11s %+16.8f Ha" % ('(HF)=', kmf.e_tot))
print("E%-11s %+16.8f Ha" % ('(EWF-CCSD)=', ecc.e_tot))
