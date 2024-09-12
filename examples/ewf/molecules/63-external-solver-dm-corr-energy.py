'''Use the cluster dump functionality (see 20-dump-clusters.py) to dump the clusters,
read them back in, solve them with pyscf's FCI solver, and show that the same result
is achieved as Vayesta's internal functionality. This computes various the energy via
paritioning and a partitioned cumulant.'''
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

# Create natom ring of minimal basis hydrogen atoms
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
print('Mean-Field (Hartree--Fock) energy = {}'.format(mf.e_tot))
assert mf.converged

# Vayesta options
use_sym = True
nfrag = 1 # When using symmetry, we have to specify the number of atoms in the fragment in this case.
#bath_opts = dict(bathtype="full")  # This is a complete bath space (should be exact)
bath_opts = dict(bathtype="dmet")   # This is the smallest bath size
# bath_opts = dict(bathtype='mp2', threshold=1.e-6)

# Run vayesta for comparison with FCI solver
emb = vayesta.ewf.EWF(mf, solver="FCI", bath_options=bath_opts, solver_options=dict(conv_tol=1.e-14))
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

# Run vayesta again, but just dump clusters, rather than solve them
emb_dump = vayesta.ewf.EWF(mf, solver="DUMP", bath_options=bath_opts, solver_options=dict(dumpfile="clusters-rhf.h5"))
# Set up fragments
with emb_dump.iao_fragmentation() as f:
    if use_sym:
        # Add rotational symmetry
        # Set order of rotation (2: 180 degrees, 3: 120 degrees, 4: 90: degrees,...),
        # axis along which to rotate and center of rotation (default units for axis and center are Angstroms):
        with f.rotational_symmetry(order=natom//nfrag, axis=[0, 0, 1]):
            f.add_atomic_fragment(range(nfrag))
    else:
        # Add all atoms as separate fragments 
        f.add_all_atomic_fragments()
print('Total number of fragments (inc. sym related): {}'.format(len(emb_dump.fragments)))
emb_dump.kernel()

class Cluster(object):

    def __init__(self, key, cluster):

        self.key = key
        self.c_frag = np.array(cluster["c_frag"]) # Fragment orbitals (AO,Frag)
        self.c_cluster = np.array(cluster["c_cluster"]) # Cluster orbitals (AO,Cluster)

        self.norb, self.nocc, self.nvir = cluster.attrs["norb"], cluster.attrs["nocc"], cluster.attrs["nvir"]
        self.fock = np.array(cluster["fock"]) # Fock matrix (Cluster,Cluster)
        self.h1e = np.array(cluster["heff"]) # 1 electron integrals (Cluster,Cluster)
        self.h2e = np.array(cluster["eris"]) # 2 electron integrals (Cluster,Cluster,Cluster,Cluster)

    def __repr__(self):
        return self.key

# Load clusters from file
with h5py.File("clusters-rhf.h5", "r") as f:
    clusters = [Cluster(key, cluster) for key, cluster in f.items()]
print("Clusters loaded: %d" % len(clusters))

# To complete the info we need the AO overlap matrix (also get the MO coefficients)
mo_coeff, ovlp = mf.mo_coeff, mf.get_ovlp()

def split_dm2(nocc, dm1, dm2, ao_repr=False):
    dm2_2 = dm2.copy()             # Approx cumulant
    dm2_1 = np.zeros_like(dm2)     # Correlated 1DM contribution
    dm2_0 = np.zeros_like(dm2)     # Mean-field 1DM contribution

    for i in range(nocc):
        dm2_1[i, i, :, :] += dm1 * 2
        dm2_1[:, :, i, i] += dm1 * 2
        dm2_1[:, i, i, :] -= dm1
        dm2_1[i, :, :, i] -= dm1.T

    for i in range(nocc):
        for j in range(nocc):
            dm2_0[i, i, j, j] += 4
            dm2_0[i, j, j, i] -= 2

    dm2_2 -= dm2_0 + dm2_1
    return dm2_0, dm2_1, dm2_2

# Full system Hamiltonian (AO basis)
h1e_ao = mf.get_hcore()
h2e_ao = mol.intor("int2e")

# Full-system FCI for comparison
fci = pyscf.fci.FCI(mf)
e_ci, c_ci = fci.kernel()
dm1_mo_fci, dm2_mo_fci = fci.make_rdm12(c_ci, h1e_ao.shape[0], mol.nelec)
dm1_ao_fci = mf.mo_coeff @ dm1_mo_fci @ mf.mo_coeff.T
dm2_ao_fci = np.einsum('pqrs,ip,jq,kr,ls->ijkl', dm2_mo_fci, mf.mo_coeff, mf.mo_coeff, mf.mo_coeff, mf.mo_coeff)
e1_fci = np.einsum('pq,pq->', h1e_ao, dm1_ao_fci)
e2_fci = 0.5 * np.einsum('pqrs,pqrs->', h2e_ao, dm2_ao_fci)

# Democratic Partitioning
# Go through each cluster, and solve it with pyscf's FCI module. Find its energy contribution by passing in the hamiltonian.
# Calculate democratically partitioned energy and 1DM
e1_dpart, e2_dpart = 0, 0
for ind, cluster in enumerate(clusters):
    # Solve, and return energy and FCI wave function
    energy, ci_vec = pyscf.fci.direct_spin0.kernel(cluster.h1e, cluster.h2e, cluster.norb, nelec=(cluster.nocc, cluster.nocc), conv_tol=1.e-14)
    
    # Build cluster denisty matrices
    orbs = Orbitals(cluster.c_cluster, occ=cluster.nocc)
    wf = FCI_WaveFunction(orbs, ci_vec)#.as_cisd(c0=1.0)
    dm1_cls, dm2_cls = wf.make_rdm1(), wf.make_rdm2()

    # Project DMs
    proj = cluster.c_frag.T @ ovlp @ cluster.c_cluster
    proj = proj.T @ proj
    dm1_cls = proj @ dm1_cls
    dm1_cls = 0.5 * (dm1_cls + dm1_cls.T)

    # Project 2DM
    dm2_cls = np.einsum('Ijkl,iI->ijkl', dm2_cls, proj)
    
    # Calculate effective 1 body Hamiltonian and subtract cluster contribution to Veff
    nocc = cluster.nocc
    heff = cluster.c_cluster.T @  (mf.get_hcore() + mf.get_veff()/2) @ cluster.c_cluster
    heff -= np.einsum("iipq->pq", cluster.h2e[:nocc, :nocc, :, :]) - np.einsum("iqpi->pq", cluster.h2e[:nocc, :, :, :nocc]) / 2

    # Calculate energy contributions
    e1_dpart+= np.einsum('pq,pq->', heff, dm1_cls)
    e2_dpart+= 0.5 * np.einsum('pqrs,pqrs->', cluster.h2e, dm2_cls)

if use_sym:
    e1_dpart *= natom // nfrag
    e2_dpart *= natom // nfrag

# Partitioned FCI
e1_pf, e22_pf = 0, 0
nocc = int(mf.mo_occ.sum()//2)
dm1_mo_fci[np.diag_indices(nocc)] -= 2 # Subtract HF contribtuion
h2e_mo = np.einsum('ijkl,ip,jq,kr,ls->pqrs', h2e_ao, mf.mo_coeff, mf.mo_coeff, mf.mo_coeff, mf.mo_coeff)
dm2_0, dm2_1, dm2_2 = split_dm2(nocc, dm1_mo_fci, dm2_mo_fci)
e22_pf = 0.5 * np.einsum('pqrs,pqrs->', h2e_mo, dm2_2)
fock_mo = mf.mo_coeff.T @ mf.get_fock() @ mf.mo_coeff
e1_pf = np.einsum('pq,pq->', fock_mo, dm1_mo_fci)


# Partitioned Cumulant
# Calculate 1DM energy contribution over full system from previously obtained democratically partitioned 1DM
e22_pc =  0
dm1_ao_pc = np.zeros_like(dm1_ao_fci)
for ind, cluster in enumerate(clusters):
    # Solve, and return energy and FCI wave function
    energy, ci_vec = pyscf.fci.direct_spin0.kernel(cluster.h1e, cluster.h2e, cluster.norb, nelec=(cluster.nocc, cluster.nocc), conv_tol=1.e-14)
    
    # Build cluster denisty matrices and split into seperate contributions
    orbs = Orbitals(cluster.c_cluster, occ=cluster.nocc)
    wf = FCI_WaveFunction(orbs, ci_vec)
    dm1_cls, dm2_cls = wf.make_rdm1(), wf.make_rdm2()
    nocc = cluster.nocc
    dm1_cls[np.diag_indices(cluster.nocc)] -= 2 # Subtract HF contribtuion
    dm2_0, dm2_1, dm2_2 = split_dm2(cluster.nocc, dm1_cls, dm2_cls) # Split into HF, correlated non-cumulant and cumulant contributions

    # Project 1DM and rotate to AO basis
    proj = cluster.c_frag.T @ ovlp @ cluster.c_cluster
    proj = proj.T @ proj
    dm1_cls = proj @ dm1_cls
    dm1_ao_pc += cluster.c_cluster @ dm1_cls @ cluster.c_cluster.T

    # Project 2DM and contract with 2-electron integrals within cluster
    dm2_2 = np.einsum('Ijkl,iI->ijkl', dm2_2, proj)
    e22_pc += 0.5 * np.einsum('pqrs,pqrs->', cluster.h2e, dm2_2)

# Calculate correlated non-cumulant contribution over full system
e1_pc = np.einsum('pq,pq->', mf.get_fock(), dm1_ao_pc) 

if use_sym:
    e1_pc *= natom // nfrag
    e22_pc *= natom // nfrag

print("Correlation Energies\n--------------------")
print("FCI energy from density matrix  = %f" % (e1_fci+e2_fci+mol.energy_nuc()-mf.e_tot))
print("External democratic paritioning = %f" % (e1_dpart+e2_dpart+mol.energy_nuc()-mf.e_tot))
print("Vayesta  democratic paritioning = %f" % (emb.get_dmet_energy(part_cumulant=False)-mf.e_tot))
print("FCI energy from cumulant        = %f" % (e1_pf+e22_pf))
print("External partitioned cumulant   = %f" % (e1_pc+e22_pc))
print("Vayesta  partitioned cumulant   = %f" % (emb.get_dmet_energy(part_cumulant=True)-mf.e_tot))

