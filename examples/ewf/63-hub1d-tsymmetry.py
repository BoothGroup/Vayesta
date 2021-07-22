# Compare to PySCF:
# pyscf/examples/scf/40-customizing_hamiltonian.py

import numpy as np

import vayesta
import vayesta.ewf
import vayesta.lattmod

nsite = 10
nelectron = nsite
nimp = 2
hubbard_u = 10.0
mol = vayesta.lattmod.Hubbard1D(nsite, nelectron=nelectron, hubbard_u=hubbard_u, output='pyscf.out')
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
# fill be automatically copied. There are two ways to pass the three translation vectors,
# which define the symmetric translations.
opt = 1
# Option 1)
# Specify translation vectors as parts of the full system lattice vectors
# by passing a list with three integers, [n, m, l];
# the translation vectors will be set equal to the lattice vectors, divided
# by n, m, l in a0, a1, and a2 direction, repectively.
if opt == 1:
    f.make_tsymmetric_fragments(tvecs=[nsite//nimp,1,1])
# Option 2) Specify translation vectors directly in units of Angstrom
# (Bohr is possible, by adding `unit='Bohr'`).
# Note that the Hubbard model class assumes a distance of 1A between neighboring sites.
if opt == 2:
    f.make_tsymmetric_fragments(tvecs=np.diag([nimp, 1, 1]))


# Check results:

def get_global_amplitudes(fragments):
    # Get combined T1 amplitudes
    nocc = nelectron//2
    nvir = nsite - nocc
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
