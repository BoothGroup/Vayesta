import numpy as np
import pyscf
import pyscf.gto
import pyscf.scf
import pyscf.fci
import vayesta
import vayesta.ewf
from vayesta.misc.molecules import ring

# User defined FCI solver - takes pyscf mf as input and returns RDMs
# The mf argment contains the hamiltonain in the orthonormal cluster basis
# Pyscf or other solvers may be used to solve the cluster problem and may return RDMs, CISD amplitudes or CCSD amplitudes
def solver(mf):
    h1e = mf.get_hcore()
    h2e = mf._eri
    norb = h1e[0].shape[-1]
    nelec = mf.mol.nelec
    energy, civec = pyscf.fci.direct_uhf.kernel(h1e, h2e, norb, nelec, conv_tol=1.e-14)
    dm1, dm2 = pyscf.fci.direct_uhf.make_rdm12s(civec, norb, nelec)
    results = dict(dm1=dm1, dm2=dm2, converged=True)
    return results

natom = 10
mol = pyscf.gto.Mole()
mol.atom = ring("H", natom, 1.5)
mol.basis = "sto-3g"
mol.output = "pyscf.out"
mol.verbose = 5
mol.symmetry = True
mol.build()

# Hartree-Fock
mf = pyscf.scf.UHF(mol)
mf.kernel()

# Vayesta options
use_sym = True
nfrag = 1
bath_opts = dict(bathtype="dmet")   

# Run vayesta with user defined solver
# NOTE: if the callback solver only returns RDMs, then energy_functional must be set to 'dmet'
emb = vayesta.ewf.EWF(mf, solver="CALLBACK",  energy_functional='dmet', bath_options=bath_opts, solver_options=dict(callback=solver))
# Set up fragments
with emb.iaopao_fragmentation() as f:
    if use_sym:
        # Add rotational symmetry
        with f.rotational_symmetry(order=natom//nfrag, axis=[0, 0, 1]):
            f.add_atomic_fragment(range(nfrag))
    else:
        # Add all atoms as separate fragments 
        f.add_all_atomic_fragments()
emb.kernel()

emb_cb = vayesta.ewf.EWF(mf, solver="CALLBACK",  energy_functional='dmet', bath_options=bath_opts, solver_options=dict(callback=solver))
# Set up fragments
with emb_cb.iaopao_fragmentation() as f:
    if use_sym:
        # Add rotational symmetry
        with f.rotational_symmetry(order=natom//nfrag, axis=[0, 0, 1]):
            f.add_atomic_fragment(range(nfrag))
    else:
        # Add all atoms as separate fragments 
        f.add_all_atomic_fragments()
emb_cb.kernel()


e1, ecb1 = emb.get_dmet_energy(part_cumulant=False, approx_cumulant=False), emb_cb.get_dmet_energy(part_cumulant=False, approx_cumulant=False)
e2, ecb2 = emb.get_dmet_energy(part_cumulant=True, approx_cumulant=False), emb_cb.get_dmet_energy(part_cumulant=True, approx_cumulant=False)
e3, ecb3 = emb.get_dmet_energy(part_cumulant=True, approx_cumulant=True), emb_cb.get_dmet_energy(part_cumulant=True, approx_cumulant=True)

print("Hartree-Fock energy          : %s"%mf.e_tot)
print("                                    Vayesta                Callback")
print("DMET energy                  : %s    %s"%(e1, ecb1))
print("DMET energy   (part-cumulant): %s    %s"%(e2, ecb2))
print("DMET energy (approx-cumulant): %s    %s"%(e3, ecb3))

