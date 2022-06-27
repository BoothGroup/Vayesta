import pyscf
import pyscf.gto
import pyscf.scf
import pyscf.cc
import vayesta
import vayesta.ewf


mol = pyscf.gto.Mole()
mol.atom = """
O  0.0000   0.0000   0.1173
H  0.0000   0.7572  -0.4692
H  0.0000  -0.7572  -0.4692
"""
mol.basis = 'cc-pVDZ'
mol.output = 'pyscf.out'
mol.build()

# Hartree-Fock
mf = pyscf.scf.RHF(mol)
mf.kernel()

# Embedded CCSD
dumpfile = 'cluster.h5'
emb = vayesta.ewf.EWF(mf, solver='Dump', bath_options=dict(threshold=1e-6),
        solver_options=dict(dumpfile=dumpfile))
emb.kernel()

# Open Dump file:
import h5py
with h5py.File(dumpfile, 'r') as f:
    for key, frag in f.items():
        print()
        print("Key= %s" % key)
        # Sizes:
        norb, nocc, nvir = frag.attrs['norb'], frag.attrs['nocc'], frag.attrs['nvir']
        print("norb= %d, nocc= %d, nvir= %d" % (norb, nocc, nvir))
        # Orbital coefficients:
        c_cluster, c_frag = frag['c_cluster'], frag['c_frag']
        print("c_cluster.shape= (%d, %d)" % c_cluster.shape)
        print("c_frag.shape= (%d, %d)" % c_frag.shape)
        # Integral arrays:
        hcore, heff, fock, eris = frag['hcore'], frag['heff'], frag['fock'], frag['eris']
        print("hcore.shape= (%d, %d)" % hcore.shape)
        print("heff.shape= (%d, %d)" % heff.shape)
        print("fock.shape= (%d, %d)" % fock.shape)
        print("eris.shape= (%d, %d, %d, %d)" % eris.shape)
