import pyscf
import pyscf.pbc
import pyscf.pbc.scf
import pyscf.pbc.dft
import pyscf.pbc.cc

import vayesta
import vayesta.ewf
from vayesta.misc import solids

cell = pyscf.pbc.gto.Cell()
cell.a, cell.atom = solids.graphene()
cell.basis = "sto-3g"
cell.output = "pyscf.out"
cell.dimension = 2
cell.space_group_symmetry = True
cell.symmorphic = True
cell.build()

# HF
kmesh = [2, 2, 1]
kpts = cell.make_kpts(kmesh, space_group_symmetry=True, time_reversal_symmetry=True, symmorphic=True)
hf = pyscf.pbc.scf.KRHF(cell, cell.make_kpts([2, 2, 1]))
hf = hf.density_fit()
hf.kernel()
# This may be required to avoid issues discussed in 01-simple-sym.py.
hf.mol.space_group_symmetry = False
# Run embedded
emb_bare = vayesta.ewf.EWF(hf, bath_options=dict(bathtype="rpa", threshold=1e-2), screening=None)
emb_bare.kernel()
# Run calculation using screened interactions and cumulant correction for nonlocal energy.
emb_mrpa = vayesta.ewf.EWF(
    hf, bath_options=dict(bathtype="rpa", threshold=1e-2), screening="mrpa", ext_rpa_correction="cumulant"
)
emb_mrpa.kernel()

# Reference full system CCSD:
cc = pyscf.pbc.cc.KCCSD(hf)
cc.kernel()

print("Error(HF)=                     %+16.8f Ha" % (hf.e_tot - cc.e_tot))
print("Error(Emb. bare CCSD)=         %+16.8f Ha" % (emb_bare.e_tot - cc.e_tot))
print("Error(Emb. mRPA CCSD)=         %+16.8f Ha" % (emb_mrpa.e_tot - cc.e_tot))
print("Error(Emb. mRPA CCSD + ΔE_k)=  %+16.8f Ha" % (emb_mrpa.e_tot + emb_mrpa.e_nonlocal - cc.e_tot))
print("E(CCSD)=                       %+16.8f Ha" % cc.e_tot)
