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
mol.basis = "cc-pVDZ"
mol.output = "pyscf.txt"
mol.build()

# Hartree-Fock
mf = pyscf.scf.UHF(mol)
mf.kernel()

# Embedded CCSD
emb = vayesta.ewf.EWF(mf, bath_options=dict(threshold=1e-6))
emb.kernel()
emb_ccsd_t = emb.get_ccsd_t_corr_energy()
emb_ccsd_tg = emb.get_ccsd_t_corr_energy(global_t1=True)
# Reference full system CCSD:
cc = pyscf.cc.CCSD(mf)
cc.kernel()
ccsd_t = cc.ccsd_t()
print("E(HF)=                           %+16.8f Ha" % mf.e_tot)
print("E(CCSD(T))=                      %+16.8f Ha" % cc.e_tot) 
print("E(Emb. CCSD)[WF]=                %+16.8f Ha" % emb.e_tot)
print("(T) Correction =                 %+16.8f Ha" % ccsd_t)
print("Emb. (T) Correction    =         %+16.8f Ha" % emb_ccsd_t)
print("Emb. (T) Correction (global t1)= %+16.8f Ha" % emb_ccsd_tg)