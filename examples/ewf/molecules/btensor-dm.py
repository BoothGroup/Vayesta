from time import perf_counter

import numpy as np

import pyscf
import pyscf.gto
import pyscf.scf
import pyscf.cc
import vayesta
import vayesta.ewf

mol = pyscf.gto.Mole()

nwater = 4
use_sym = True

if nwater == 1:
    mol.atom = """
    O  0.0000   0.0000   0.1173
    H  0.0000   0.7572  -0.4692
    H  0.0000  -0.7572  -0.4692
    """
elif nwater == 2:
    mol.atom = """
    O  2.0000   0.0000   0.1173
    H  2.0000   0.7572  -0.4692
    H  2.0000  -0.7572  -0.4692
    O  6.0000   0.0000   0.1173
    H  6.0000   0.7572  -0.4692
    H  6.0000  -0.7572  -0.4692
    """
elif nwater == 4:
    mol.atom = """
    O  0.0000   0.0000   0.1173
    H  0.0000   0.7572  -0.4692
    H  0.0000  -0.7572  -0.4692
    O  2.0000   0.0000   0.1173
    H  2.0000   0.7572  -0.4692
    H  2.0000  -0.7572  -0.4692
    O  6.0000   0.0000   0.1173
    H  6.0000   0.7572  -0.4692
    H  6.0000  -0.7572  -0.4692
    O  8.0000   0.0000   0.1173
    H  8.0000   0.7572  -0.4692
    H  8.0000  -0.7572  -0.4692
    """

#mol.basis = 'cc-pVDZ'
#mol.basis = 'cc-pVTZ'
mol.basis = 'cc-pVQZ'
mol.output = 'pyscf.txt'
mol.build()

# Hartree-Fock
mf = pyscf.scf.RHF(mol)
mf.kernel()

# Embedded CCSD
eta = 1e-6
emb = vayesta.ewf.EWF(mf, bath_options=dict(threshold=eta))
with emb.fragmentation() as f:
    if nwater > 1 and use_sym:
        with f.mirror_symmetry(axis='x', center=(4, 0, 0)):
            for i in range(mol.natm//2):
                f.add_atomic_fragment(i)
    else:
        f.add_all_atomic_fragments()
emb.kernel()


# Density-matrix

#svd_tol = 1e-3
svd_tol = None
ovlp_tol = None
with_t1 = True

times = [perf_counter()]
dm1 = emb._make_rdm1_ccsd_global_wf(svd_tol=svd_tol, ovlp_tol=ovlp_tol, with_t1=with_t1, use_sym=use_sym)
times.append(perf_counter())

print(f"Vayesta time 1= {times[1]-times[0]:.3f}")

times = [perf_counter()]
dm1_bt1 = emb._make_rdm1_ccsd_global_wf_btensor(with_t1=with_t1, svd_tol=svd_tol, ovlp_tol=ovlp_tol, use_sym=use_sym)
times.append(perf_counter())
dm1_bt1 = emb._make_rdm1_ccsd_global_wf_btensor(with_t1=with_t1, svd_tol=svd_tol, ovlp_tol=ovlp_tol, use_sym=use_sym)
times.append(perf_counter())
dm1_bt1 = emb._make_rdm1_ccsd_global_wf_btensor(with_t1=with_t1, svd_tol=svd_tol, ovlp_tol=ovlp_tol, use_sym=use_sym)
times.append(perf_counter())

print(f"BTensor time 1= {times[1]-times[0]:.3f}")
print(f"BTensor time 2= {times[2]-times[1]:.3f}")
print(f"BTensor time 3= {times[3]-times[2]:.3f}")

print(f"Error in DM= {np.linalg.norm(dm1_bt1 - dm1)}")
