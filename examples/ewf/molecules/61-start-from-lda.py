import pyscf
import pyscf.gto
import pyscf.dft
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
mol.output = 'pyscf.out'
mol.build()

# LDA
lda = pyscf.dft.RKS(mol)
lda.xc = 'svwn'
lda.kernel()
# The KS opbject needs to be converted to a HF object:
lda_as_hf = pyscf.scf.RHF(mol)
lda_as_hf.__dict__.update(lda.__dict__)
lda = lda_as_hf

emb_lda = vayesta.ewf.EWF(lda, bno_threshold=1e-6)
emb_lda.kernel()

# HF
hf = pyscf.scf.RHF(mol)
hf.kernel()

emb_hf = vayesta.ewf.EWF(hf, bno_threshold=1e-6)
emb_hf.kernel()

# Reference full system CCSD:
cc = pyscf.cc.CCSD(hf)
cc.kernel()

print("E(LDA)=            %+16.8f Ha" % lda.e_tot)
print("E(HF)=             %+16.8f Ha" % hf.e_tot)
print("E(HF @LDA)=        %+16.8f Ha" % emb_lda.e_mf)
print("E(Emb. CCSD @LDA)= %+16.8f Ha" % emb_lda.e_tot)
print("E(Emb. CCSD @HF)=  %+16.8f Ha" % emb_hf.e_tot)
print("E(CCSD)=           %+16.8f Ha" % cc.e_tot)
