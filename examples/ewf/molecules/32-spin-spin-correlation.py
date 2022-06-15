import numpy as np
import pyscf
import pyscf.gto
import pyscf.scf

import vayesta
import vayesta.ewf

mol = pyscf.gto.Mole()
mol.atom = """
O  0.0000   0.0000   0.1173
H  0.0000   0.7572  -0.4692
H  0.0000  -0.7572  -0.4692
"""
mol.basis = 'cc-pVTZ'
mol.charge = mol.spin = 1
mol.output = 'pyscf.out'
mol.build()

# Hartree-Fock:
mf = pyscf.scf.UHF(mol)
mf.kernel()

# Embedded CCSD:
emb = vayesta.ewf.EWF(mf, bath_options=dict(threshold=1e-4),
        solver_options=dict(solve_lambda=True))
with emb.iao_fragmentation() as frag:
    frag.add_atomic_fragment(0)
    frag.add_atomic_fragment(1, bath_options=dict(threshold=-1))
    frag.add_atomic_fragment(2, bath_options=dict(threshold=+1))
emb.kernel()

# Lowdin orbitals, all atoms:
ssz = emb.get_atomic_ssz()
print("SAO <Sz(A)Sz(B)> values:")
for a in range(mol.natm):
    for b in range(mol.natm):
        print('A= %d, B= %d:  %+.5f' % (a, b, ssz[a,b]))
print("Total:       %+.5f" % ssz.sum())

# Subset of atoms:
atoms = [0,2]
ssz = emb.get_atomic_ssz(atoms=atoms)
print("SAO <Sz(A)Sz(B)> values:")
for i, a in enumerate(atoms):
    for j, b in enumerate(atoms):
        print('A= %d, B= %d:  %+.5f' % (a, b, ssz[i,j]))
print("Total:       %+.5f" % ssz.sum())

# IAO+PAO orbitals
ssz = emb.get_atomic_ssz(projection='iaopao')
print("IAO+PAO <Sz(A)Sz(B)> values:")
for a in range(mol.natm):
    for b in range(mol.natm):
        print('A= %d, B= %d:  %+.5f' % (a, b, ssz[a,b]))
print("Total:       %+.5f" % ssz.sum())
