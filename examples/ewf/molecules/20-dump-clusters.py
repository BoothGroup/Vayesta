import pyscf
import pyscf.gto
import pyscf.scf
import pyscf.cc
import vayesta
import vayesta.ewf


# ---- Restricted spin symmetry

mol = pyscf.gto.Mole()
mol.atom = """
O1  0.0000   0.0000   0.1173
H2  0.0000   0.7572  -0.4692
H3  0.0000  -0.7572  -0.4692
"""
mol.basis = 'cc-pVDZ'
mol.output = 'pyscf.out'
mol.build()

# Hartree-Fock
mf = pyscf.scf.RHF(mol)
mf.kernel()

# Embedded CCSD
emb = vayesta.ewf.EWF(mf, solver='Dump', bath_options=dict(threshold=1e-6),
        solver_options=dict(dumpfile='clusters-rhf.h5'))
emb.kernel()

# ---- Unrestricted spin symmetry

mol = pyscf.gto.Mole()
mol.atom = """
O1  0.0000   0.0000   0.1173
H2  0.0000   0.7572  -0.4692
H3  0.0000  -0.7572  -0.4692
"""
mol.charge = mol.spin = 3
mol.basis = 'cc-pVDZ'
mol.output = 'pyscf.out'
mol.build()

# Hartree-Fock
mf = pyscf.scf.UHF(mol)
mf.kernel()

# Embedded CCSD
emb = vayesta.ewf.EWF(mf, solver='Dump', bath_options=dict(threshold=1e-6),
        solver_options=dict(dumpfile='clusters-uhf.h5'))
emb.kernel()


# --- Open restricted dump file:
print("\nOpening restricted dump file:")

import h5py
with h5py.File('clusters-rhf.h5', 'r') as f:
    # The HDF5 file contains a separate group for each fragment in the system:
    for key, frag in f.items():
        # The HDF5-group key for each fragment is constructed as 'fragment_%d' % id
        print("\nKey= %s" % key)
        # Name and ID of fragment:
        print("name= %s, id= %d" % (frag.attrs['name'], frag.attrs['id']))
        # Number of all/occupied/virtual orbitals:
        print("Full cluster:")
        norb, nocc, nvir = frag.attrs['norb'], frag.attrs['nocc'], frag.attrs['nvir']
        print("norb= %d, nocc= %d, nvir= %d" % (norb, nocc, nvir))
        # Orbital coefficients:
        # The first dimension corresponds to the atomic orbitals,
        # the second dimension to the cluster or fragment orbitals, respectively.
        print("c_cluster.shape= (%d, %d)" % frag['c_cluster'].shape)
        print("c_frag.shape=    (%d, %d)" % frag['c_frag'].shape)
        # Integral arrays:
        # hcore, fock, and eris are the 1-electron Hamiltonian, Fock, and 2-electron Hamiltonian
        # matrix elements in the cluster space.
        # heff is equal to hcore, plus an additional potential due to the Coulomb and exchange
        # interaction with the occupied environment orbitals outside the cluster.
        print("hcore.shape=     (%d, %d)" % frag['hcore'].shape)
        print("heff.shape=      (%d, %d)" % frag['heff'].shape)
        print("fock.shape=      (%d, %d)" % frag['fock'].shape)
        # The 2-electron integrals are in chemical ordering: eris[i,j,k,l] = (ij|kl)
        print("eris.shape=      (%d, %d, %d, %d)" % frag['eris'].shape)
        # DMET cluster:
        print("DMET cluster:")
        norb, nocc, nvir = [frag.attrs['%s_dmet_cluster' % x] for x in ('norb', 'nocc', 'nvir')]
        print("norb= %d, nocc= %d, nvir= %d" % (norb, nocc, nvir))
        print("c_dmet_cluster.shape=    (%d, %d)" % frag['c_dmet_cluster'].shape)

# --- Open unrestricted dump file:
print("\nOpening unrestricted dump file:")

with h5py.File('clusters-uhf.h5', 'r') as f:
    for key, frag in f.items():
        print("\nKey= %s" % key)
        # Name and ID:
        print("name= %s, id= %d" % (frag.attrs['name'], frag.attrs['id']))
        # The orbital sizes are now arrays of length 2, representing alpha and beta spin dimension
        norb, nocc, nvir = frag.attrs['norb'], frag.attrs['nocc'], frag.attrs['nvir']
        print("norb= (%d, %d), nocc= (%d, %d), nvir= (%d, %d)" % (*norb, *nocc, *nvir))
        # The suffixes _a and _b correspond to alpha and beta spin:
        print("c_cluster_a.shape= (%d, %d)" % frag['c_cluster_a'].shape)
        print("c_cluster_b.shape= (%d, %d)" % frag['c_cluster_b'].shape)
        print("c_frag_a.shape=    (%d, %d)" % frag['c_frag_a'].shape)
        print("c_frag_b.shape=    (%d, %d)" % frag['c_frag_b'].shape)
        # Integral arrays:
        print("hcore_a.shape=     (%d, %d)" % frag['hcore_a'].shape)
        print("hcore_b.shape=     (%d, %d)" % frag['hcore_b'].shape)
        print("heff_a.shape=      (%d, %d)" % frag['heff_a'].shape)
        print("heff_b.shape=      (%d, %d)" % frag['heff_b'].shape)
        print("fock_a.shape=      (%d, %d)" % frag['fock_a'].shape)
        print("fock_b.shape=      (%d, %d)" % frag['fock_b'].shape)
        # The spin blocks correspond to (aa|aa), (aa|bb), and (bb|bb)
        # (Note that (bb|aa) = (aa|bb).transpose(2,3,0,1) for real orbitals):
        print("eris_aa.shape=     (%d, %d, %d, %d)" % frag['eris_aa'].shape)
        print("eris_ab.shape=     (%d, %d, %d, %d)" % frag['eris_ab'].shape)
        print("eris_bb.shape=     (%d, %d, %d, %d)" % frag['eris_bb'].shape)
        # DMET cluster:
        print("DMET cluster:")
        norb, nocc, nvir = [frag.attrs['%s_dmet_cluster' % x] for x in ('norb', 'nocc', 'nvir')]
        print("norb= (%d, %d), nocc= (%d, %d), nvir= (%d, %d)" % (*norb, *nocc, *nvir))
        print("c_dmet_cluster_a.shape=    (%d, %d)" % frag['c_dmet_cluster_a'].shape)
        print("c_dmet_cluster_b.shape=    (%d, %d)" % frag['c_dmet_cluster_b'].shape)
