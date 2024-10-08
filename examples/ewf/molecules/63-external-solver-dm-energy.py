'''Use the cluster dump functionality (see 20-dump-clusters.py) to dump the clusters,
read them back in, solve them with pyscf's FCI solver, and show that the same result
is achieved as Vayesta's internal functionality. Note that this just computes the
'projected' energy estimator over the clusters, from the C2 amplitudes.'''
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
natom = 6
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
bath_opts = dict(bathtype="full")   # This is the smallest bath size
# bath_opts = dict(bathtype='mp2', threshold=1.e-6)

# Run vayesta for comparison with FCI solver
emb = vayesta.ewf.EWF(mf, solver="FCI", bath_options=bath_opts, solver_options=dict(conv_tol=1.e-14))
# Set up fragments
with emb.iaopao_fragmentation() as f:
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
with emb_dump.iaopao_fragmentation() as f:
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
print(clusters)
print("Clusters loaded: %d" % len(clusters))

# To complete the info we need the AO overlap matrix (also get the MO coefficients)
mo_coeff, ovlp = mf.mo_coeff, mf.get_ovlp()


def split_dm2(nocc, dm1, dm2):
    dm2_2 = dm2.copy()             # Approx cumulant
    dm2_1 = np.zeros_like(dm2)     # Correlated 1DM contribution
    dm2_0 = np.zeros_like(dm2)     # Mean-field 1DM contribution

    dm1 = dm1.copy()
    dm1[np.diag_indices(nocc)] -= 2

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

h1e_ao = mf.get_hcore()
h2e_ao = mol.intor("int2e")
print(h2e_ao.dtype)
print("\nHartree-Fock\n------------")
dm1_ao_hf = mf.make_rdm1()
dm2_ao_hf = mf.make_rdm2()
e1_hf = np.einsum('pq,pq->', h1e_ao, dm1_ao_hf)
e2_hf = 0.5 * np.einsum('pqrs,pqrs->', h2e_ao, dm2_ao_hf)
print("E = %f" % (e1_hf+e2_hf+mol.energy_nuc()))
print("E1 = %f" % e1_hf)
print("E2 = %f" % e2_hf)
# Full-system FCI for comparison

print("\nFCI\n------------")
fci = pyscf.fci.FCI(mf)
e_ci, c_ci = fci.kernel()
dm1_mo_fci, dm2_mo_fci = fci.make_rdm12(c_ci, dm1_ao_hf.shape[0], mol.nelec)
dm1_ao_fci = mf.mo_coeff @ dm1_mo_fci @ mf.mo_coeff.T
dm2_ao_fci = np.einsum('pqrs,ip,jq,kr,ls->ijkl', dm2_mo_fci, mf.mo_coeff, mf.mo_coeff, mf.mo_coeff, mf.mo_coeff)
e1_fci = np.einsum('pq,pq->', h1e_ao, dm1_ao_fci)
e2_fci = 0.5 * np.einsum('pqrs,pqrs->', h2e_ao, dm2_ao_fci)
print("E = %f" % (e1_fci+e2_fci+mol.energy_nuc()))
print("E1 = %f" % e1_fci)
print("E2 = %f" % e2_fci)

print("\nDemocratic Partitioning\n------------")
# Go through each cluster, and solve it with pyscf's FCI module. Find its energy contribution by passing in the hamiltonian.
# Calculate democratically partitioned energy and 1DM
e1_dpart, e2_dpart = 0, 0
dm1_dpart = np.zeros_like(mf.mo_coeff)
for ind, cluster in enumerate(clusters):
    # Solve, and return energy and FCI wave function
    energy, ci_vec = pyscf.fci.direct_spin0.kernel(cluster.h1e, cluster.h2e, cluster.norb, nelec=(cluster.nocc, cluster.nocc), conv_tol=1.e-14)
    
    orbs = Orbitals(cluster.c_cluster, occ=cluster.nocc)
    wf = FCI_WaveFunction(orbs, ci_vec)#.as_cisd(c0=1.0)
    dm1_cls, dm2_cls = wf.make_rdm1(), wf.make_rdm2()

    #dm1_cls[np.diag_indices(cluster.nocc)] -= 2
    #dm2_0, dm2_1, dm2_2 = split_dm2(cluster.nocc, dm1_cls, dm2_cls)

    proj = cluster.c_frag.T @ ovlp @ cluster.c_cluster
    proj = proj.T @ proj
    # Project DMs
    dm1_cls = proj @ dm1_cls
    dm1_cls = 0.5 * (dm1_cls + dm1_cls.T)
    dm1_dpart += cluster.c_cluster @ dm1_cls @ cluster.c_cluster.T

    dm2_cls = np.einsum('Ijkl,iI->ijkl', dm2_cls, proj)

    e1_dpart+= np.einsum('pq,pq->', cluster.h1e, dm1_cls)
    e2_dpart+= 0.5 * np.einsum('pqrs,pqrs->', cluster.h2e, dm2_cls)
    
    #e2_dpart += 0.5 * np.einsum('pqrs,pqrs->', cluster.h2e, dm2_1 + dm2_2)

if use_sym:
    e1_dpart *= natom // nfrag
    e2_dpart *= natom // nfrag

print("E = %f" % (e1_dpart+e2_dpart+mol.energy_nuc()))
print("E1 = %f" % e1_dpart)
print("E2 = %f" % e2_dpart)

print("\nPartitioned FCI\n------------")
e1_pf, e20_pf, e21_pf, e22_pf = 0, 0, 0, 0
e1_pf = np.einsum('pq,pq->', h1e_ao, dm1_ao_fci)
SC = ovlp @ mf.mo_coeff
#dm2_mo_fci = np.einsum('ijkl,ip,jq,kr,ls->pqrs', dm2_ao_fci,SC,SC,SC,SC)
h2e_mo = np.einsum('ijkl,ip,jq,kr,ls->pqrs', h2e_ao, mf.mo_coeff, mf.mo_coeff, mf.mo_coeff, mf.mo_coeff)
dm2_0, dm2_1, dm2_2 = split_dm2(6, dm1_mo_fci, dm2_mo_fci)
e20_pf = 0.5 * np.einsum('pqrs,pqrs->', h2e_mo, dm2_0)
e21_pf = 0.5 * np.einsum('pqrs,pqrs->', h2e_mo, dm2_1)
e22_pf = 0.5 * np.einsum('pqrs,pqrs->', h2e_mo, dm2_2)
print("E = %f" % (e1_pf+e20_pf+e21_pf+e22_pf+mol.energy_nuc()))
print("E1 = %f" % e1_pf)
print("E2_0 = %f" % e20_pf)
print("E2_1 = %f" % e21_pf)
print("E2_2 = %f" % e22_pf)
print("Tr[gF] = %f"%np.einsum('pq,pq->', dm1_ao_fci, mf.get_fock()))
print("\nPartitioned Cumulant\n------------")
e1_pc, e20_pc, e21_pc, e22_pc = 0, 0, 0, 0
e1_pc = np.einsum('pq,pq->', mf.get_hcore(), dm1_dpart)

for ind, cluster in enumerate(clusters):
    # Solve, and return energy and FCI wave function
    energy, ci_vec = pyscf.fci.direct_spin0.kernel(cluster.h1e, cluster.h2e, cluster.norb, nelec=(cluster.nocc, cluster.nocc), conv_tol=1.e-14)
    
    orbs = Orbitals(cluster.c_cluster, occ=cluster.nocc)
    wf = FCI_WaveFunction(orbs, ci_vec)#.as_cisd(c0=1.0)
    dm1_cls, dm2_cls = wf.make_rdm1(), wf.make_rdm2()

    #dm1_cls[np.diag_indices(cluster.nocc)] -= 2
    
    dm1_cls = proj @ dm1_cls
    dm2_cls = np.einsum('Ijkl,iI->ijkl', dm2_cls, proj)
    dm2_0, dm2_1, dm2_2 = split_dm2(cluster.nocc, dm1_cls, dm2_cls)
    #dm2_cls = 0.125 * (dm2_cls + dm2_cls.transpose(1, 0, 3, 2))# + dm2_cls.transpose(2, 3, 0, 1) + dm2_cls.transpose(3, 2, 1, 0))

    e20_pc += 0.5 * np.einsum('pqrs,pqrs->', cluster.h2e, dm2_0)
    e21_pc += 0.5 * np.einsum('pqrs,pqrs->', cluster.h2e, dm2_1)
    e22_pc += 0.5 * np.einsum('pqrs,pqrs->', cluster.h2e, dm2_2)
    
    #e2_dpart += 0.5 * np.einsum('pqrs,pqrs->', cluster.h2e, dm2_1 + dm2_2)

if use_sym:
    e1_pc *= natom // nfrag
    e20_pc *= natom // nfrag
    e21_pc *= natom // nfrag
    e22_pc *= natom // nfrag
print("E = %f" % (e1_pc+e20_pc+e21_pc+e22_pc+mol.energy_nuc()))
print("E1 = %f" % e1_pc)
print("E2_0 = %f" % e20_pc)
print("E2_1 = %f" % e21_pc)
print("E2_2 = %f" % e22_pc)

exit()
print("Full system mean-field energy               = %f" % mf.e_tot)
print("Full system CCSD correlation energy         = %f" % cc.e_tot)
print("Full system FCI correlation energy          = %f" %(ci.e_tot))
print("Vayesta correlation energy                  = %f" % emb.e_corr)
print("Vayesta partitioned cumulant energy         = %f" % emb.get_dmet_energy(part_cumulant=True))
# print("Vayesta partitioned cumulant energy         = %f" % get_vayesta_dmet_energy(emb, part_cumulant=True, approx_cumulant=False))
# print("Vayesta partitioned approx cumulant energy  = %f" % emb.get_dmet_energy(part_cumulant=False))
# print("Vayesta partitioned approx cumulant energy  = %f" % get_vayesta_dmet_energy(emb, part_cumulant=True, approx_cumulant=True))
print("Correlation energy from external FCI solver = %f" % e_corr)


def get_energy(dm1, dm2, h1e, h2e):

    e1 = np.einsum('pq,pq->', h1e, dm1)
    e2 = np.einsum('pqrs,pqrs->', h2e, dm2) / 2

    return e1 + e2

