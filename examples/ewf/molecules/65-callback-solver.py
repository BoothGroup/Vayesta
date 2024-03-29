import numpy as np
import pyscf
import pyscf.gto
import pyscf.scf
import pyscf.fci
import vayesta
import vayesta.ewf
from vayesta.misc.molecules import ring
from vayesta.core.types import Orbitals, FCI_WaveFunction, CISD_WaveFunction
import h5py


def solver(mf):
    h1e = mf.get_hcore()
    h2e = mf._eri
    norb = mf.mo_coeff.shape[-1]
    nelec = mf.mol.nelec
    energy, ci_vec = pyscf.fci.direct_spin0.kernel(h1e, h2e, norb, nelec, conv_tol=1.e-14)
    
    dm1, dm2 = pyscf.fci.direct_spin0.make_rdm12(ci_vec, norb, nelec)

    results = dict(dm1=dm1, dm2=dm2, converged=True, energy=energy, ci_vec=ci_vec)

    return results

natom = 2
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
print('Mean-Field (Hartree--Fock) energy = {}'.format(mf.e_tot))


# Vayesta options
use_sym = True
nfrag = 1 # When using symmetry, we have to specify the number of atoms in the fragment in this case.
#bath_opts = dict(bathtype="full")  # This is a complete bath space (should be exact)
bath_opts = dict(bathtype="dmet")   # This is the smallest bath size
# bath_opts = dict(bathtype='mp2', threshold=1.e-6)

# Run vayesta for comparison with FCI solver
emb = vayesta.ewf.EWF(mf, solver="CALLBACK",  energy_functional='dm', bath_options=bath_opts, solver_options=dict(callback=solver))
# Set up fragments
with emb.iao_fragmentation() as f:
    if use_sym:
        # Add rotational symmetry
        # Set order of rotation (2: 180 degrees, 3: 120 degrees, 4: 90: degrees,...),
        # axis along which to rotate and center of rotation (default units for axis and center are Angstroms):
        with f.rotational_symmetry(order=natom//nfrag, axis=[0, 0, 1]):
            f.add_atomic_fragment(range(nfrag))
    else:
        # Add all atoms as separate fragments 
        f.add_all_atomic_fragments()
emb.kernel()
print("DMET energy: %s"%emb.get_dmet_energy(part_cumulant=False, approx_cumulant=False))
print("DMET energy (part-cumulant): %s"%emb.get_dmet_energy(part_cumulant=True, approx_cumulant=False))
print("DMET energy (approx-cumulant): %s"%emb.get_dmet_energy(part_cumulant=True, approx_cumulant=True))
print("ERROR: %s"%emb.get_dmet_energy(part_cumulant=False, approx_cumulant=True))