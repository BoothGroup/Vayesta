import pyscf
import pyscf.gto
import pyscf.scf
import pyscf.cc
import vayesta
import vayesta.ewf

mol = pyscf.gto.Mole()
mol.atom = 'N 0 0 0 ; N 0 0 1.4'
mol.basis = 'cc-pVDZ'
mol.build()

# Hartree-Fock
mf = pyscf.scf.RHF(mol)
mf.kernel()

# Reference full system CCSD:
cc = pyscf.cc.CCSD(mf)
cc.kernel()

# Embedded CCSD
emb = vayesta.ewf.EWF(mf, bath_options=dict(threshold=1e-4))
emb.kernel()

# Embedded Tailored CCSD (Embedded CCSD tailored with FCI in DMET cluster)
emb_tcc = vayesta.ewf.EWF(mf, solver='TCCSD', bath_options=dict(threshold=1e-4))
emb_tcc.kernel()

print("E(HF)=         %+16.8f Ha" % mf.e_tot)
print("E(CCSD)=       %+16.8f Ha" % cc.e_tot)
print("E(emb. CCSD)=  %+16.8f Ha" % emb.e_tot)
print("E(emb. TCCSD)= %+16.8f Ha" % emb_tcc.e_tot)
