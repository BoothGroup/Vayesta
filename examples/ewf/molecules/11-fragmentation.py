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
mol.basis = 'cc-pVDZ'
mol.output = 'pyscf.out'
mol.build()

# Hartree-Fock
mf = pyscf.scf.RHF(mol)
mf.kernel()

# Embedded CCSD
emb_iao = vayesta.ewf.EWF(mf, bno_threshold=1e-6)
# If calling the kernel without initializing the fragmentation,
# IAO fragmentation is used automatically:
# It can be initialized manually by calling emb.iao_fragmentation()
emb_iao.kernel()

emb_iaopao = vayesta.ewf.EWF(mf, bno_threshold=1e-6)
# To use IAO+PAOs, call emb.iaopao_fragmentation() before the kernel:
with emb_iaopao.iaopao_fragmentation() as f:
    f.add_all_atomic_fragments()
emb_iaopao.kernel()

emb_sao = vayesta.ewf.EWF(mf, bno_threshold=1e-6)
# To use Lowdin AOs (SAOs), call emb.sao_fragmentation() before the kernel:
with emb_sao.sao_fragmentation() as f:
    f.add_all_atomic_fragments()
emb_sao.kernel()

# Reference full system CCSD:
cc = pyscf.cc.CCSD(mf)
cc.kernel()

print("E(HF)=      %+16.8f Ha" % mf.e_tot)
print("E(IAO)=     %+16.8f Ha" % emb_iao.e_tot)
print("E(IAO+PAO)= %+16.8f Ha" % emb_iaopao.e_tot)
print("E(SAO)=     %+16.8f Ha" % emb_sao.e_tot)
print("E(CCSD)=    %+16.8f Ha" % cc.e_tot)
