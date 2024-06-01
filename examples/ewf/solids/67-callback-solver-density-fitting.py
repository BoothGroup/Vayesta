import numpy as np

import pyscf
import pyscf.pbc
import pyscf.pbc.scf
import pyscf.pbc.cc
import pyscf.fci

import vayesta
import vayesta.ewf

# User defined FCI solver - takes pyscf mf as input and returns RDMs
# The mf argment contains the hamiltonain in the orthonormal cluster basis
# Pyscf or other solvers may be used to solve the cluster problem and may return RDMs, CISD amplitudes or CCSD amplitudes
def solver(mf):
    h1e = mf.get_hcore()
    # Need to convert cderis into standard 4-index tensor when using denisty fitting for the mean-field
    cderi = mf.with_df._cderi
    cderi = pyscf.lib.unpack_tril(cderi)
    h2e = np.einsum('Lpq,Lrs->pqrs', cderi, cderi)
    norb = mf.mo_coeff.shape[-1]
    nelec = mf.mol.nelec
    energy, civec = pyscf.fci.direct_spin0.kernel(h1e, h2e, norb, nelec, conv_tol=1.e-14)
    dm1, dm2 = pyscf.fci.direct_spin0.make_rdm12(civec, norb, nelec)
    results = dict(dm1=dm1, dm2=dm2, converged=True)
    return results

cell = pyscf.pbc.gto.Cell()
cell.a = 3.0 * np.eye(3)
cell.atom = "He 0 0 0"
cell.basis = "cc-pvdz"
#cell.exp_to_discard = 0.1
cell.build()

kmesh = [3, 3, 3]
kpts = cell.make_kpts(kmesh)

# --- Hartree-Fock
kmf = pyscf.pbc.scf.KRHF(cell, kpts)
kmf = kmf.rs_density_fit()
kmf.kernel()

# Vayesta options
nfrag = 1 
bath_opts = dict(bathtype="mp2", dmet_threshold=1e-15)   

# Run vayesta with user defined solver
emb = vayesta.ewf.EWF(kmf, solver="CALLBACK",  energy_functional='dmet', bath_options=bath_opts, solver_options=dict(callback=solver))
# Set up fragments
with emb.iao_fragmentation() as f:
        f.add_all_atomic_fragments()
emb.kernel()

print("Hartree-Fock energy          : %s"%kmf.e_tot)
print("DMET energy                  : %s"%emb.get_dmet_energy(part_cumulant=False, approx_cumulant=False))
print("DMET energy   (part-cumulant): %s"%emb.get_dmet_energy(part_cumulant=True, approx_cumulant=False))
print("DMET energy (approx-cumulant): %s"%emb.get_dmet_energy(part_cumulant=True, approx_cumulant=True))

