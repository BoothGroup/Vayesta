import numpy as np

import pyscf
import pyscf.pbc
import pyscf.pbc.scf
import pyscf.pbc.cc

import vayesta
import vayesta.ewf
from vayesta.misc import solids


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
emb = vayesta.ewf.EWF(kmf, bath_type='full', solve_lambda=True)
emb.kernel()
e_dm = emb.get_dm_energy()

# --- Reference full system CCSD
cc = pyscf.pbc.cc.KUCCSD(kmf)
cc.kernel()


print("Total Energy")
print("E(HF)=        %+16.8f Ha" % kmf.e_tot)
print("E(Proj)=      %+16.8f Ha" % emb.e_tot)
print("E(RDM2, gl)=  %+16.8f Ha" % emb._get_dm_energy_old(global_dm1=True, global_dm2=False))
print("E(RDM2, ll)=  %+16.8f Ha" % emb._get_dm_energy_old(global_dm1=False, global_dm2=False))

print("E(CCSD)=      %+16.8f Ha" % (kmf.e_tot + cc.e_corr))

print("\nCorrelation Energy")
print("E(Proj)=      %+16.8f Ha" % emb.e_corr)
print("E(RDM2, gl)=  %+16.8f Ha" % emb._get_dm_corr_energy_old(global_dm1=True, global_dm2=False))
print("E(RDM2, ll)=  %+16.8f Ha" % emb._get_dm_corr_energy_old(global_dm1=False, global_dm2=False))

print("E(CCSD)=      %+16.8f Ha" % cc.e_corr)
