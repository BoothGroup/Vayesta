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

# Reference full system CCSD:
cc = pyscf.cc.CCSD(mf)
cc.kernel()

# Embedded CCSD
emb = vayesta.ewf.EWF(mf, bath_options=dict(threshold=1e-6),
        solver_options=dict(solve_lambda=True))
emb.kernel()

print("Total Energy")
print("E(HF)=        %+16.8f Ha" % mf.e_tot)
print("E(Proj)=      %+16.8f Ha" % emb.e_tot)
print("E(RDM2, gg)=  %+16.8f Ha" % emb.get_dm_energy_old(global_dm1=True, global_dm2=True))
print("E(RDM2, gl)=  %+16.8f Ha" % emb.get_dm_energy_old(global_dm1=True, global_dm2=False))
print("E(RDM2, lg)=  %+16.8f Ha" % emb.get_dm_energy_old(global_dm1=False, global_dm2=True))
print("E(RDM2, ll)=  %+16.8f Ha" % emb.get_dm_energy_old(global_dm1=False, global_dm2=False))

print("E(CCSD)=      %+16.8f Ha" % cc.e_tot)

print("\nCorrelation Energy")
print("E(Proj)=      %+16.8f Ha" % emb.e_corr)
print("E(RDM2, gg)=  %+16.8f Ha" % emb.get_dm_corr_energy_old(global_dm1=True, global_dm2=True))
print("E(RDM2, gl)=  %+16.8f Ha" % emb.get_dm_corr_energy_old(global_dm1=True, global_dm2=False))
print("E(RDM2, lg)=  %+16.8f Ha" % emb.get_dm_corr_energy_old(global_dm1=False, global_dm2=True))
print("E(RDM2, ll)=  %+16.8f Ha" % emb.get_dm_corr_energy_old(global_dm1=False, global_dm2=False))

print("E(CCSD)=      %+16.8f Ha" % cc.e_corr)
