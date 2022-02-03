import numpy as np

import pyscf
import pyscf.pbc
import pyscf.pbc.scf
import pyscf.pbc.cc
import pyscf.pbc.tools

import vayesta
import vayesta.ewf




a = 5.4307 # Silicon equilibrium lattice const in Angstroms
cell = pyscf.pbc.gto.Cell()
cell.atom = ['Si 0.0 0.0 0.0', 'Si %f %f %f' % (a/4, a/4, a/4)]
cell.a = np.asarray([
    [a/2, a/2, 0],
    [0, a/2, a/2],
    [a/2, 0, a/2]])
#cell.basis = 'def2-svp'
cell.basis = 'gth-dzvp'
cell.pseudo = 'gth-pade'
cell.ke_cutoff = 80
cell.exp_to_discard=0.1

#cell.verbose=5
cell.max_memory = 190305596
cell.build()

kmesh = [2,2,2]
kpts = cell.make_kpts(kmesh)

print("Running RHF")
kmf = pyscf.pbc.scf.KRHF(cell, kpts)
kmf.exxdiv = None
print("No exxdiv")
kmf = kmf.density_fit()
kmf.kernel()
print(kmf.e_tot)

print("Running Vayesta")
#vayesta.new_log("vayesta-si-%.1f.log" % a)
emb = vayesta.ewf.EWF(kmf, solver="CCSD", bno_threshold=1e-4)
emb.iao_fragmentation()
#emb.add_atomic_fragment(0, sym_factor=1) # 2 Si-atoms per unit cell
#emb.add_atomic_fragment(1, sym_factor=1)
emb.add_all_atomic_fragments()
emb.kernel()
print("Vayesta")
e_rdm2 = emb.get_rdm2_energy()
print(e_rdm2)

print("Running CCSD")
cc = pyscf.pbc.cc.KCCSD(kmf)
cc.kernel()


print("Total Energy")
print("E(HF)=        %+16.8f Ha" % kmf.e_tot)
print("E(Proj)=      %+16.8f Ha" % emb.e_tot)
print("E(RDM2, gl)=  %+16.8f Ha" % emb.get_rdm2_energy(global_dm1=True, global_dm2=False))
print("E(RDM2, ll)=  %+16.8f Ha" % emb.get_rdm2_energy(global_dm1=True, global_dm2=False))

print("E(CCSD)=      %+16.8f Ha" % cc.e_tot)

print("\nCorrelation Energy")
print("E(Proj)=      %+16.8f Ha" % emb.e_corr)
print("E(RDM2, gl)=  %+16.8f Ha" % emb.get_rdm2_corr_energy(global_dm1=True, global_dm2=False))
print("E(RDM2, ll)=  %+16.8f Ha" % emb.get_rdm2_corr_energy(global_dm1=True, global_dm2=False))

print("E(CCSD)=      %+16.8f Ha" % cc.e_corr)
