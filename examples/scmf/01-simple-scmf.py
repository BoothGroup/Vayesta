import numpy as np
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
mol.verbose = 4
mol.build()

# Hartree-Fock
mf = pyscf.scf.RHF(mol)
mf.kernel()

# CCSD
cc = pyscf.cc.CCSD(mf)
cc.kernel()

emb = vayesta.ewf.EWF(mf, bno_threshold=1e-4, solve_lambda=True)
with emb.sao_fragmentation() as f:
    f.add_all_atomic_fragments()
emb.pdmet_scmf()
# or:
#emb.brueckner_scmf()
emb.kernel()
assert emb.with_scmf.converged

print("E(HF)=           %+16.8f Ha" % mf.e_tot)
print("E(CCSD)=         %+16.8f Ha" % cc.e_tot)
print("E(Emb. CCSD)=    %+16.8f Ha" % emb.with_scmf.e_tot_oneshot)
print("E(sc Emb. CCSD)= %+16.8f Ha" % emb.e_tot)
