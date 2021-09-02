# Compare to PySCF:
# pyscf/examples/scf/40-customizing_hamiltonian.py

import numpy as np

import vayesta
import vayesta.ewf
import vayesta.lattmod

nsite = 16
nimp = 2
hubbard_u = 6.0
boundary = 'APBC'
nelectron = nsite
mol = vayesta.lattmod.Hubbard1D(nsite, nelectron=nelectron, hubbard_u=hubbard_u, boundary=boundary, output='pyscf.out')
mf = vayesta.lattmod.LatticeMF(mol)
mf.kernel()

# Calculate each fragment:
ewf1 = vayesta.ewf.EWF(mf, solver='FCI', fragment_type='Site', bath_type=None)
for site in range(0, nsite, nimp):
    ewf1.make_atom_fragment(list(range(site, site+nimp)))
ewf1.kernel()

# Calculate a single fragment and use translational symmetry:
ewf2 = vayesta.ewf.EWF(mf, solver='FCI', fragment_type='Site', bath_type=None)
f = ewf2.make_atom_fragment(list(range(nimp)))
ewf2.kernel()

# Add fragments which are translationally symmetric to f - the results of the fragment f
# fill be automatically copied.
# Specify translation vectors as parts of the full system lattice vectors
# by passing a list with three integers, [n, m, l];
# the translation vectors will be set equal to the lattice vectors, divided
# by n, m, l in a0, a1, and a2 direction, repectively.
symfrags = f.make_tsymmetric_fragments(tvecs=[nsite//nimp, 1, 1])
print("%d symmetry equivalent fragments found" % len(symfrags))
# Check that every fragment has been identified!
assert (len(symfrags)+1 == nsite//nimp)

# Compare C1 in site basis
def site_basis_c1(f):
    c1 = f.results.c1
    c1 = np.einsum('ia,pi,qa->pq', c1, f.c_active_occ, f.c_active_vir)
    return c1

for i in range(len(ewf1.fragments)):
    err1 = np.linalg.norm(site_basis_c1(ewf1.fragments[i]) - site_basis_c1(ewf2.fragments[i]))
    err2 = np.linalg.norm(site_basis_c1(ewf1.fragments[i]) + site_basis_c1(ewf2.fragments[i]))
    err = min(err1, err2)
    print("Error of fragment %3d= %.3e" % (i, err))
