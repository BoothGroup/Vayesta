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
cell.basis = 'cc-pVDZ'
cell.output = 'pyscf.out'
cell.dimension = 2
cell.build()

# HF
hf = pyscf.pbc.scf.RHF(cell)
hf = hf.density_fit()#auxbasis='cc-pVDZ-ri')
hf.kernel()

# Run embedded
emb_bare = vayesta.ewf.EWF(hf, bath_options=dict(bathtype="rpa", threshold=1e-1), screening=None)
emb_bare.kernel()

emb_mrpa = vayesta.ewf.EWF(hf, bath_options=dict(bathtype="rpa", threshold=1e-1), screening="mrpa")
emb_mrpa.kernel()

# Reference full system CCSD:
cc = pyscf.pbc.cc.CCSD(hf)
cc.kernel()

print("E(HF)=              %+16.8f Ha" % hf.e_tot)
print("E(Emb. bare CCSD)=  %+16.8f Ha" % emb_bare.e_tot)
print("E(Emb. mRPA CCSD)=  %+16.8f Ha" % emb_mrpa.e_tot)
print("E(CCSD)=            %+16.8f Ha" % cc.e_tot)
