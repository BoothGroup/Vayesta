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

# The convergence with respect to the BNO threshold can be improved by
# adding secondary MP2 fragments ("delta-MP2"):
emb1 = vayesta.ewf.EWF(mf, bath_options=dict(threshold=1e-4))
with emb1.iao_fragmentation() as f:
    # The BNO threshold can also be set as a factor of the CCSD threshold,
    # using the argument bno_threshold_factor
    with f.secondary_fragments(solver='MP2', bno_threshold=1e-8):
        f.add_all_atomic_fragments()
emb1.kernel()

# Reference embedded CCSD without secondary MP2 fragments:
emb0 = vayesta.ewf.EWF(mf, bath_options=dict(threshold=1e-4))
emb0.kernel()

# Reference full system CCSD:
cc = pyscf.cc.CCSD(mf)
cc.kernel()

print("E(Emb. CCSD)=      %+16.8f Ha  (error= %+.8f Ha)" % (emb0.e_tot, emb0.e_tot-cc.e_tot))
print("E(Emb. CCSD/MP2)=  %+16.8f Ha  (error= %+.8f Ha)" % (emb1.e_tot, emb1.e_tot-cc.e_tot))
