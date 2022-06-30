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

# Parse options and calculate HF energy
emb_iao = vayesta.ewf.EWF(mf, bath_options=dict(threshold=1e-6),
                          solver_options=dict(solve_lambda=True))

# If calling the kernel without initializing the fragmentation,
# IAO fragmentation is used automatically:
emb_iao.kernel()

# Parse options and calculate HF energy
emb_iaopao = vayesta.ewf.EWF(mf, bath_options=dict(threshold=1e-6),
                          solver_options=dict(solve_lambda=True))

# To use IAO+PAOs, call emb.iaopao_fragmentation() before the kernel:
with emb_iaopao.iaopao_fragmentation() as f:
    f.add_all_atomic_fragments()
emb_iaopao.kernel()

# Parse options and calculate HF energy
emb_sao = vayesta.ewf.EWF(mf, bath_options=dict(threshold=1e-6),
                          solver_options=dict(solve_lambda=True))

# To use Lowdin AOs (SAOs), call emb.sao_fragmentation() before the kernel:

with emb_sao.sao_fragmentation() as f:
    f.add_all_atomic_fragments()
emb_sao.kernel()


# Parse options and calculate HF energy
emb = vayesta.ewf.EWF(mf, bath_options=dict(threshold=1e-6),
                          solver_options=dict(solve_lambda=True))

# Calculate IAOs of systems and map to atoms

with emb.iao_fragmentation() as frag:
    # Populate emb.fragments with 3 fragments, one for each atom
    # in water
    frag.add_atomic_fragment(0)
    frag.add_atomic_fragment(1)
    frag.add_atomic_fragment(2)

emb.kernel()

# Reference full system CCSD:
cc = pyscf.cc.CCSD(mf)
cc.kernel()

print("E(HF)=      %+16.8f Ha" % mf.e_tot)
print("E(IAO)=     %+16.8f Ha" % emb_iao.e_tot)
print("E(IAO+PAO)= %+16.8f Ha" % emb_iaopao.e_tot)
print("E(SAO)=     %+16.8f Ha" % emb_sao.e_tot)
print("E(IA_Frag)= %+16.8f Ha" % emb.e_tot)
print("E(CCSD)=    %+16.8f Ha" % cc.e_tot)
