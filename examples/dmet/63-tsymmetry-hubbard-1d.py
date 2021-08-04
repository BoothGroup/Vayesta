# Compare to PySCF:
# pyscf/examples/scf/40-customizing_hamiltonian.py

import numpy as np

import vayesta.dmet
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
dmet1 = vayesta.dmet.DMET(mf, solver='FCI', fragment_type='Site')
for site in range(0, nsite, nimp):
    dmet1.make_atom_fragment(list(range(site, site+nimp)))
dmet1.kernel()

# Calculate a single fragment and use translational symmetry:
dmet2 = vayesta.dmet.DMET(mf, solver='FCI', fragment_type='Site')
f = dmet2.make_atom_fragment(list(range(nimp)))
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
dmet2.kernel()



# Compare converged correlation potential
print("L2 norm difference in converged correlation potentials with and without translational symmetry: {:6.4e}".format(
                    sum((dmet1.vcorr.ravel() - dmet2.vcorr.ravel())**2)**(0.5)))
