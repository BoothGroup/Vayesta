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
mol.build()

# Hartree-Fock
mf = pyscf.scf.RHF(mol)
mf.kernel()

# CCSD
cc = pyscf.cc.CCSD(mf)
cc.kernel()

# One shot
opts = dict(bath_options=dict(threshold=1e-4), solver_options=dict(solve_lambda=True))
emb = vayesta.ewf.EWF(mf, **opts)
with emb.sao_fragmentation() as f:
    f.add_all_atomic_fragments()
emb.kernel()

# p-DMET
emb_pdmet = vayesta.ewf.EWF(mf, **opts)
with emb_pdmet.sao_fragmentation() as f:
    f.add_all_atomic_fragments()
emb_pdmet.pdmet_scmf()
emb_pdmet.kernel()
assert emb_pdmet.with_scmf.converged

# Brueckner
emb_brueckner = vayesta.ewf.EWF(mf, **opts)
with emb_brueckner.sao_fragmentation() as f:
    f.add_all_atomic_fragments()
emb_brueckner.brueckner_scmf()
emb_brueckner.kernel()
assert emb_brueckner.with_scmf.converged


print("E(HF)=                    %+16.8f Ha" % mf.e_tot)
print("E(CCSD)=                  %+16.8f Ha" % cc.e_tot)
print("E(Emb. CCSD)=             %+16.8f Ha" % emb.e_tot)
print("E(Emb. CCSD + p-DMET)=    %+16.8f Ha" % emb_pdmet.e_tot)
print("E(Emb. CCSD + Brueckner)= %+16.8f Ha" % emb_brueckner.e_tot)
