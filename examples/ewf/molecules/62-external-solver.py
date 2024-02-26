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

# Full-system FCI for comparison
ci = pyscf.fci.FCI(mf)
ci.kernel()
print('FCI total energy: {}'.format(ci.e_tot))

# Full-system coupled cluster
cc = pyscf.cc.CCSD(mf)
cc.kernel()
print('CCSD total energy: {}'.format(cc.e_tot))

# Vayesta options
use_sym = True
nfrag = 1 # When using symmetry, we have to specify the number of atoms in the fragment in this case.
#bath_opts = dict(bathtype="full")  # This is a complete bath space (should be exact)
bath_opts = dict(bathtype="dmet")   # This is the smallest bath size
# bath_opts = dict(bathtype='mp2', threshold=1.e-6)

# Run vayesta for comparison with FCI solver
emb = vayesta.ewf.EWF(mf, solver="FCI", bath_options=bath_opts, solver_options=dict(conv_tol=1.e-14)
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
print(clusters)
print("Clusters loaded: %d" % len(clusters))

# To complete the info we need the AO overlap matrix (also get the MO coefficients)
mo_coeff, ovlp = mf.mo_coeff, mf.get_ovlp()

# Go through each cluster, and solve it with pyscf's FCI module. Find its energy contribution by passing in the hamiltonian.
e_corr = 0
for ind, cluster in enumerate(clusters):
    # Solve, and return energy and FCI wave function
    energy, ci_vec = pyscf.fci.direct_spin0.kernel(cluster.h1e, cluster.h2e, cluster.norb, nelec=(cluster.nocc, cluster.nocc), conv_tol=1.e-14)
    orbs = Orbitals(cluster.c_cluster, occ=cluster.nocc)
    # Get the FCI wave function, and extract out the C0, C1 and C2 coefficients. Set C0 to 1 (intermediate normalization).
    wf = FCI_WaveFunction(orbs, ci_vec).as_cisd(c0=1.0)
    #assert(np.allclose(wf.c2,emb.fragments[ind].results.wf.as_cisd(c0=1.0).c2))
    #print("Error in C2 amplitudes: ",np.linalg.norm(emb.fragments[ind].results.wf.as_cisd(c0=1.0).c2-wf.c2))

    # Create a fragment projector of the occupied cluster orbitals onto the fragment space
    proj = cluster.c_frag.T @ ovlp @ cluster.c_cluster[:,:cluster.nocc]
    #assert(np.allclose(proj, emb.fragments[ind].get_overlap("frag|cluster-occ")))
    # Transform the first index of the C2 and C1 amplitudes to the fragment space
    wf_proj = wf.project(proj)

    # Calculate projected correlation energy contribution
    o, v = np.s_[:cluster.nocc], np.s_[cluster.nocc:]
    # NOTE: Formally, we should include a C1^2 contribution to the energy too, but this is likely to be small.
    # If using a non-canonical reference, there should also be a C1.F contribution too, which we are ignoring here.
    e_loc = 2*np.einsum('xi,xjab,iabj', proj, wf_proj.c2, cluster.h2e[o,v,v,o]) - np.einsum('xi,xjab,ibaj', proj, wf_proj.c2, cluster.h2e[o,v,v,o])
    #assert(np.isclose(emb.fragments[ind].get_fragment_energy(wf_proj.c1, wf_proj.c2)[2], e_loc))
    print('From cluster {}: FCI energy = {}, local correlation energy contribution = {}'.format(cluster, energy, e_loc))
    e_corr += e_loc
if use_sym:
    e_corr *= natom // nfrag

print("Full system mean-field energy = %f" % mf.e_tot)
print("Full system CCSD correlation energy = %f" % cc.e_corr)
print("Full system FCI correlation energy: {}".format(ci.e_tot-mf.e_tot))
print("Vayesta correlation energy = %f" % emb.e_corr)
print("Correlation energy from external FCI solver = %f" % e_corr)
