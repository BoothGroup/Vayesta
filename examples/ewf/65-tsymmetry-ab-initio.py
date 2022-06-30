# Compare to PySCF:
# pyscf/examples/scf/40-customizing_hamiltonian.py

import numpy as np

import pyscf
import pyscf.pbc
import pyscf.pbc.gto

import vayesta
import vayesta.ewf
import vayesta.lattmod

cell = pyscf.pbc.gto.Cell()
spacing = 1.8
shape = (4, 4)
impshape = (2, 2)
ncells = shape[0]*shape[1]
nimp = impshape[0] * impshape[1]
cell.a = np.zeros((3,3))
cell.a[0,0] = shape[0]*spacing
cell.a[1,1] = shape[1]*spacing
cell.a[2,2] = 20.0 # Vacuum
atoms = []
for row in range(shape[0]):
    for col in range(shape[1]):
        atoms.append('H %f %f 0' % (col*spacing, row*spacing))
cell.atom = atoms
cell.dimension = 2


# Reordering of atoms into (2, 2) tiles
from vayesta.lattmod.latt import Hubbard2D
order = Hubbard2D.get_tiles_order(shape, impshape)


cell.basis = '6-31G'
cell.build()
mf = pyscf.pbc.scf.RHF(cell)
mf = mf.density_fit(auxbasis='def2-universal-jkfit')
mf.conv_tol = 1e-14
mf.kernel()
assert mf.converged

# Calculate each fragment:
ewf1 = vayesta.ewf.EWF(mf, solver='FCI', bath_type=None)
for site in range(0, ncells, nimp):
    ewf1.add_atomic_fragment(order[site:site + nimp])
ewf1.kernel()

# Calculate a single fragment and use translational symmetry:
ewf2 = vayesta.ewf.EWF(mf, solver='FCI', bath_type=None)
f = ewf2.add_atomic_fragment(order[:nimp])
ewf2.kernel()

# Add fragments which are translationally symmetric to f - the results of the fragment f
# fill be automatically copied.
# Specify translation vectors as parts of the full system lattice vectors
# by passing a list with three integers, [n, m, l];
# the translation vectors will be set equal to the lattice vectors, divided
# by n, m, l in a0, a1, and a2 direction, repectively.
symfrags = f.make_tsymmetric_fragments(tvecs=[2, 2, 1])
assert len(symfrags) + 1 == ncells//nimp

# Check results:

def get_global_amplitudes(fragments):
    # Get combined T1 amplitudes
    nocc = np.count_nonzero(mf.mo_occ > 0)
    nvir = np.count_nonzero(mf.mo_occ == 0)
    occ = np.s_[:nocc]
    vir = np.s_[nocc:]
    c1 = np.zeros((nocc, nvir))
    c2 = np.zeros((nocc, nocc, nvir, nvir))
    for x in fragments:
        px = x.get_fragment_projector(x.c_active_occ)
        pc1 = np.dot(px, x.results.c1/x.results.c0)
        pc2 = np.einsum('xi,ijab->xjab', px, x.results.c2/x.results.c0)
        # Rotate from cluster basis to MO basis
        ro = np.dot(x.c_active_occ.T, mf.mo_coeff[:,occ])
        rv = np.dot(x.c_active_vir.T, mf.mo_coeff[:,vir])
        c1 += np.einsum('ia,ip,aq->pq', pc1, ro, rv)
        c2 += np.einsum('ijab,ip,jq,ar,bs->pqrs', pc2, ro, ro, rv, rv)
    return c1, c2

c1a, c2a = get_global_amplitudes(ewf1.fragments)
c1b, c2b = get_global_amplitudes(ewf2.fragments)

print("Error C1= %.2e" % np.linalg.norm(c1a - c1b))
print("Error C2= %.2e" % np.linalg.norm(c2a - c2b))
