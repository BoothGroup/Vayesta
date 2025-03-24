import numpy as np
import pyscf
import pyscf.gto
import pyscf.scf
import pyscf.fci
import vayesta
import vayesta.egf
import vayesta.egf.self_energy
import vayesta.lattmod
from vayesta.misc.molecules import ring
from dyson import FCI, MBLGF


# User defined FCI solver - takes pyscf mf as input and returns RDMs
# The mf argment contains the hamiltonain in the orthonormal cluster basis
# Pyscf or other solvers may be used to solve the cluster problem and may return RDMs, CISD amplitudes or CCSD amplitudes
# Returning the cluster Green's function moments is also supported. They are calculated with Dyson in this example.
def solver(mf):
    fci_1h = FCI["1h"](mf)
    fci_1p = FCI["1p"](mf)

    # Use MBLGF
    nmom_max = 10
    th = fci_1h.build_gf_moments(nmom_max)
    tp = fci_1p.build_gf_moments(nmom_max)

    norb = mf.mo_coeff.shape[-1]
    nelec = mf.mol.nelec
    civec= fci_1h.c_ci
    dm1, dm2 = pyscf.fci.direct_spin0.make_rdm12(civec, norb, nelec)
    results = dict(dm1=dm1, dm2=dm2, gf_hole_moments=th, gf_particle_moments=tp, converged=True)
    return results


nsite = 10
u = 3
mol = vayesta.lattmod.Hubbard1D(nsite, nsite, hubbard_u=u)
mf = vayesta.lattmod.LatticeRHF(mol)
mf.kernel()


# momFCI
results = solver(mf)
th, tp = results["gf_hole_moments"], results["gf_particle_moments"]
se, gf = vayesta.egf.self_energy.gf_moments_block_lanczos((th, tp), shift='aux', nelec=mol.nelectron)


# Embedded momFCI

# Vayesta options
use_sym = True
nfrag = 2
bath_opts = dict(bathtype="dmet", order=1, max_order=1)   

# Run vayesta with user defined solver
emb = vayesta.egf.EGF(mf, solver="CALLBACK", proj=1, energy_functional='dmet', chempot_global='aux', bath_options=bath_opts, solver_options=dict(callback=solver))
emb.qsEGF(maxiter=20)
# Set up fragments
with emb.site_fragmentation() as f:
    if use_sym:
        # Add symmetry
        nimages = [nsite//nfrag, 1, 1]
        emb.symmetry.set_translations(nimages)
        f.add_atomic_fragment(range(nfrag))
    else:
        # Add all atoms as separate fragments 
        f.add_all_atomic_fragments()
emb.kernel()


ip = lambda gf: gf.physical().occupied().energies[-1]
ea = lambda gf: gf.physical().virtual().energies[0]
gap = lambda gf: gf.physical().virtual().energies[0] - gf.physical().occupied().energies[-1]
print('momFCI gap      = %f'%gap(gf))
print('Emb. momFCI gap = %f'%gap(emb.gf))
