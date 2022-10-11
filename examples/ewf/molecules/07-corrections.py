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
mol.basis = 'cc-pVTZ'
mol.output = 'pyscf.out'
mol.build()

# Hartree-Fock
mf = pyscf.scf.RHF(mol)
mf.kernel()

# Reference full system CCSD:
cc = pyscf.cc.CCSD(mf)
cc.kernel()

# Regular embedded CCSD:
emb = vayesta.ewf.EWF(mf, bath_options=dict(threshold=1e-4))
emb.kernel()

# A finite-bath correction (FBC) can be added to the energy.
# The FBC works best for large BNO thresholds.
e_fbc = emb.get_fbc_energy()

# The convergence with respect to the BNO threshold can be improved further by
# adding secondary MP2 fragments.
# Secondary fragments will both improve the energy and density-matrices:
emb_sec = vayesta.ewf.EWF(mf, bath_options=dict(threshold=1e-4))
with emb_sec.iao_fragmentation() as f:
    # The BNO threshold of the secondary fragments can be set directly, using `bno_threshold`,
    # or as a fraction of the CCSD threshold, using `bno_threshold_factor`:
    with f.secondary_fragments(solver='MP2', bno_threshold=1e-6):
        f.add_all_atomic_fragments()
emb_sec.kernel()
# The FBC can be combined with the secondary-fragment approach:
e_fbc_sec = emb_sec.get_fbc_energy()

print("E(Emb. CCSD)=            %+16.8f Ha  (error= %+.8f Ha)" % (emb.e_tot, emb.e_tot-cc.e_tot))
print("E(Emb. CCSD + FBC)=      %+16.8f Ha  (error= %+.8f Ha)" % (emb.e_tot+e_fbc, emb.e_tot+e_fbc-cc.e_tot))
print("E(Emb. CCSD/MP2)=        %+16.8f Ha  (error= %+.8f Ha)" % (emb_sec.e_tot, emb_sec.e_tot-cc.e_tot))
print("E(Emb. CCSD/MP2 + FBC)=  %+16.8f Ha  (error= %+.8f Ha)" % (emb_sec.e_tot+e_fbc_sec, emb_sec.e_tot+e_fbc_sec-cc.e_tot))
