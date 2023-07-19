import pyscf
import pyscf.gto
import pyscf.dft
import pyscf.scf
import pyscf.cc
import vayesta
import vayesta.edmet


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
lda = lda.density_fit()
lda.kernel()
# The KS opbject needs to be converted to a HF object:
lda = lda.to_rhf()

emb_lda = vayesta.edmet.EDMET(lda, solver="CCSD-S-1-1", bath_options=dict(dmet_threshold=1e-12), oneshot=True, make_dd_moments=False)
with emb_lda.iao_fragmentation() as f:
    f.add_all_atomic_fragments()
emb_lda.kernel()

# b3lyp
b3lyp = pyscf.dft.RKS(mol)
b3lyp.xc = 'b3lyp'
b3lyp = b3lyp.density_fit()
b3lyp.kernel()
# The KS opbject needs to be converted to a HF object:
b3lyp = b3lyp.to_rhf()

emb_b3lyp = vayesta.edmet.EDMET(b3lyp, solver="CCSD-S-1-1", bath_options=dict(dmet_threshold=1e-12), oneshot=True, make_dd_moments=False)
with emb_b3lyp.iao_fragmentation() as f:
    f.add_all_atomic_fragments()
emb_b3lyp.kernel()

# HF
hf = pyscf.scf.RHF(mol).density_fit()
hf.kernel()

emb_hf = vayesta.edmet.EDMET(hf, solver="CCSD-S-1-1", bath_options=dict(dmet_threshold=1e-12), oneshot=True, make_dd_moments=False)
with emb_hf.iao_fragmentation() as f:
    f.add_all_atomic_fragments()
emb_hf.kernel()

# Reference full system CCSD:
cc = pyscf.cc.CCSD(hf)
cc.kernel()

print("E(LDA)=              %+16.8f Ha" % lda.e_tot)
print("E(B3LYP)=            %+16.8f Ha" % b3lyp.e_tot)
print("E(HF)=               %+16.8f Ha" % hf.e_tot)
print("E(HF @LDA)=          %+16.8f Ha" % emb_lda.e_mf)
print("E(HF @B3LYP)=        %+16.8f Ha" % emb_b3lyp.e_mf)
print("E(Emb. CCSD @LDA)=   %+16.8f Ha" % emb_lda.e_tot)
print("E(Emb. CCSD @B3LYP)= %+16.8f Ha" % emb_b3lyp.e_tot)
print("E(Emb. CCSD @HF)=    %+16.8f Ha" % emb_hf.e_tot)
print("E(CCSD)=             %+16.8f Ha" % cc.e_tot)
