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
mol.output = "pyscf.out"
mol.build()

# Hartree-Fock
mf = pyscf.scf.RHF(mol)
mf.kernel()

# Reference full system CCSD:
cc = pyscf.cc.CCSD(mf)
cc.kernel()

# Embedded CCSD
emb = vayesta.ewf.EWF(mf, bath_options=dict(threshold=1e-6), solver_options=dict(solve_lambda=True))
emb.kernel()

print("Total Energy")
print("E(HF)=        %+16.8f Ha" % mf.e_tot)
print("E(EWF-DPart)= %+16.8f Ha" % emb.get_dmet_energy())
print("E(EWF-Proj)=  %+16.8f Ha" % emb.e_tot)
print("E(EWF-DM)=    %+16.8f Ha" % emb.get_dm_energy())
print("E(CCSD)=      %+16.8f Ha" % cc.e_tot)

print("\nCorrelation Energy")
print("E(EWF-DPart)= %+16.8f Ha" % (emb.get_dmet_energy() - mf.e_tot))
print("E(EWF-Proj)=  %+16.8f Ha" % emb.e_corr)
print("E(EWF-DM)=    %+16.8f Ha" % emb.get_dm_corr_energy())
print("E(CCSD)=      %+16.8f Ha" % cc.e_corr)
