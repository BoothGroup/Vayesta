# Compare to PySCF:
# pyscf/examples/scf/40-customizing_hamiltonian.py

import numpy as np
import vayesta
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
dmet = vayesta.dmet.DMET(mf, solver='FCI')
with dmet.site_fragmentation() as f:
    for site in range(0, nsite, nimp):
        f.add_atomic_fragment(list(range(site, site+nimp)))
dmet.kernel()

# Calculate a single fragment and use translational symmetry:
dmet_sym = vayesta.dmet.DMET(mf, solver='FCI')
# Add translational symmetry:
# Specify translation vectors as parts of the full system lattice vectors
# by passing a list with three integers, [n, m, l];
# the translation vectors will be set equal to the lattice vectors, divided
# by n, m, l in a0, a1, and a2 direction, repectively.
dmet_sym.symmetry.set_translations([nsite//nimp, 1, 1])
with dmet_sym.site_fragmentation() as f:
    f.add_atomic_fragment(list(range(nimp)))
dmet_sym.kernel()

# Compare converged correlation potential
print("Difference in converged correlation potentials with and without translational symmetry:")
print("|d(V_corr)|= %.5e" % np.linalg.norm(dmet.vcorr - dmet_sym.vcorr))
