import pyscf
import pyscf.pbc
import pyscf.pbc.scf
import pyscf.pbc.dft
import pyscf.pbc.cc

import vayesta
import vayesta.ewf
from vayesta.misc import molstructs

cell = pyscf.pbc.gto.Cell()
cell.a, cell.atom = molstructs.graphene()
cell.basis = 'cc-pVDZ'
cell.output = 'pyscf.out'
cell.dimension = 2
cell.build()

# LDA
lda = pyscf.pbc.dft.RKS(cell)
lda = lda.density_fit(auxbasis='cc-pVDZ-ri')
lda.xc = 'svwn'
lda.kernel()
# The KS opbject needs to be converted to a HF object
# Do NOT use lda.to_rhf(), as this will remove the periodic boundary conditions
lda_as_hf = pyscf.pbc.scf.RHF(cell)
lda_as_hf.__dict__.update(lda.__dict__)
lda = lda_as_hf

emb_lda = vayesta.ewf.EWF(lda, bno_threshold=1e-6)
emb_lda.kernel()

# HF
hf = pyscf.pbc.scf.RHF(cell)
hf = hf.density_fit(auxbasis='cc-pVDZ-ri')
hf.kernel()

emb_hf = vayesta.ewf.EWF(hf, bno_threshold=1e-6)
emb_hf.kernel()

# Reference full system CCSD:
cc = pyscf.pbc.cc.CCSD(hf)
cc.kernel()

print("E(LDA)=            %+16.8f Ha" % lda.e_tot)
print("E(HF)=             %+16.8f Ha" % hf.e_tot)
print("E(HF @LDA)=        %+16.8f Ha" % emb_lda.e_mf)
print("E(Emb. CCSD @LDA)= %+16.8f Ha" % emb_lda.e_tot)
print("E(Emb. CCSD @HF)=  %+16.8f Ha" % emb_hf.e_tot)
print("E(CCSD)=           %+16.8f Ha" % cc.e_tot)
