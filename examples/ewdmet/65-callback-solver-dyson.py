import numpy as np
import pyscf
import pyscf.gto
import pyscf.scf
import pyscf.fci
import vayesta
import vayesta.ewf
from vayesta.misc.molecules import ring
from dyson import FCI


# User defined FCI solver - takes pyscf mf as input and returns RDMs
# The mf argment contains the hamiltonain in the orthonormal cluster basis
# Pyscf or other solvers may be used to solve the cluster problem and may return RDMs, CISD amplitudes or CCSD amplitudes
# Green's function moments are also supported, and in are calculated via Dyson in this example.
def solver(mf):
    fci_1h = FCI["1h"](mf)
    fci_1p = FCI["1p"](mf)

    # Use MBLGF
    nmom_max = 4
    th = fci_1h.build_gf_moments(nmom_max)
    tp = fci_1p.build_gf_moments(nmom_max)

    norb = mf.mo_coeff.shape[-1]
    nelec = mf.mol.nelec
    civec= fci_1h.c_ci
    dm1, dm2 = pyscf.fci.direct_spin0.make_rdm12(civec, norb, nelec)
    results = dict(dm1=dm1, dm2=dm2, hole_moments=th, particle_moments=tp, converged=True)
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
mf = pyscf.scf.RHF(mol)
mf.kernel()

# Vayesta options
use_sym = True
nfrag = 1 
bath_opts = dict(bathtype="ewdmet", order=1, max_order=1)   

# Run vayesta with user defined solver
emb = vayesta.ewf.EWF(mf, solver="CALLBACK",  energy_functional='dmet', bath_options=bath_opts, solver_options=dict(callback=solver))
emb.qpewdmet_scmf(proj=2, maxiter=10)
# Set up fragments
with emb.iao_fragmentation() as f:
    if use_sym:
        # Add rotational symmetry
        with f.rotational_symmetry(order=natom//nfrag, axis=[0, 0, 1]):
            f.add_atomic_fragment(range(nfrag))
    else:
        # Add all atoms as separate fragments 
        f.add_all_atomic_fragments()
emb.kernel()

print("Hartree-Fock energy          : %s"%mf.e_tot)
print("DMET energy                  : %s"%emb.get_dmet_energy(part_cumulant=False, approx_cumulant=False))
print("DMET energy   (part-cumulant): %s"%emb.get_dmet_energy(part_cumulant=True, approx_cumulant=False))
print("DMET energy (approx-cumulant): %s"%emb.get_dmet_energy(part_cumulant=True, approx_cumulant=True))

