import numpy as np

import pyscf
import pyscf.pbc
import pyscf.pbc.scf
import pyscf.pbc.cc

import vayesta
import vayesta.ewf


cell = pyscf.pbc.gto.Cell()
cell.a = 3.0 * np.eye(3)
cell.atom = 'He 0 0 0'
cell.basis = 'cc-pvdz'
cell.exp_to_discard=0.1
cell.build()

kmesh = [2,2,1]
kpts = cell.make_kpts(kmesh)

# --- Hartree-Fock
kmf = pyscf.pbc.scf.KUHF(cell, kpts)
kmf = kmf.rs_density_fit(auxbasis='cc-pvdz-ri')
kmf.kernel()

# --- Embedding
emb = vayesta.ewf.EWF(kmf, bath_options=dict(bathtype='full'), solver_options=dict(solve_lambda=True))
emb.kernel()
e_dm = emb.get_dm_energy()

# --- Reference full system CCSD
cc = pyscf.pbc.cc.KUCCSD(kmf)
cc.kernel()



print("Total Energy")
print("E(HF)=        %+16.8f Ha" % kmf.e_tot)
print("E(EWF-DPart)= %+16.8f Ha"% emb.get_dmet_energy())
print("E(EWF-Proj)=  %+16.8f Ha" % emb.e_tot)
print("E(EWF-DM)=    %+16.8f Ha"% emb.get_dm_energy())
print("E(CCSD)=      %+16.8f Ha" % cc.e_tot)

print("\nCorrelation Energy")
print("E(EWF-DPart)= %+16.8f Ha"% (emb.get_dmet_energy()-kmf.e_tot))
print("E(EWF-Proj)=  %+16.8f Ha" % emb.e_corr)
print("E(EWF-DM)=    %+16.8f Ha"% emb.get_dm_corr_energy())
print("E(CCSD)=      %+16.8f Ha" % cc.e_corr)


