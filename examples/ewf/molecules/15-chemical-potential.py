import numpy as np
import pyscf
import pyscf.gto
import pyscf.scf
import vayesta
import vayesta.ewf

mol = pyscf.gto.Mole()
mol.atom = """
O  0.0000   0.0000   0.1173
H  0.0000   0.7572  -0.4692
H  0.0000  -0.7572  -0.4692
"""
mol.basis = 'cc-pVDZ'
mol.build()

# Hartree-Fock
mf = pyscf.scf.RHF(mol)
mf.kernel()

eta = 1e-4
# --- Standard embedded CCSD
emb = vayesta.ewf.EWF(mf, bath_options=dict(threshold=eta), solver_options=dict(solve_lambda=True))
with emb.sao_fragmentation() as frag:
    frag.add_all_atomic_fragments()
emb.kernel()
# Population analyis:
dm1 = emb.make_rdm1(ao_basis=True)
emb.pop_analysis(dm1, filename='pop.txt')

# --- Embedded CCSD with target number of electrons per fragment:
emb_nelec = vayesta.ewf.EWF(mf, bath_options=dict(threshold=eta), solver_options=dict(solve_lambda=True))
with emb_nelec.sao_fragmentation() as frag:
    frag.add_atomic_fragment(0, nelectron_target=8)
    frag.add_atomic_fragment(1, nelectron_target=1)
    frag.add_atomic_fragment(2, nelectron_target=1)
emb_nelec.kernel()
# Population analyis:
dm1 = emb_nelec.make_rdm1(ao_basis=True)
emb_nelec.pop_analysis(dm1, filename='pop-nelec.txt')

# --- Embedded CCSD with chemical potential, to ensure that democratically partitioned 1-DM has correct trace:
emb_cpt = vayesta.ewf.EWF(mf, bath_options=dict(threshold=eta), solver_options=dict(solve_lambda=True))
emb_cpt.optimize_chempot()
with emb_cpt.sao_fragmentation() as frag:
    frag.add_all_atomic_fragments()
emb_cpt.kernel()
# Population analyis:
dm1 = emb_cpt.make_rdm1(ao_basis=True)
emb_cpt.pop_analysis(dm1, filename='pop-cpt.txt')

print("E(HF)=                             %+16.8f Ha" % mf.e_tot)
print("E(emb. CCSD)=                      %+16.8f Ha" % emb.e_tot)
print("E(emb. CCSD + electron target)=    %+16.8f Ha" % emb_nelec.e_tot)
print("E(emb. CCSD + chemical potential)= %+16.8f Ha" % emb_cpt.e_tot)
