# Compare to PySCF:
# pyscf/examples/scf/40-customizing_hamiltonian.py

import numpy as np
import vayesta
import vayesta.dmet
import vayesta.lattmod


nsites = [6,6]
impshape = [2,2]
hubbard_u = 6.0
# Anti-PBC and mixed boundaries do not work with translational symmetry!
boundary = 'PBC'
nsite = nsites[0]*nsites[1]
nelectron = nsite-10
nimp = impshape[0]*impshape[1]
mol = vayesta.lattmod.Hubbard2D(nsites, nelectron=nelectron, hubbard_u=hubbard_u, boundary=boundary, tiles=impshape, output='pyscf.out')
mf = vayesta.lattmod.LatticeMF(mol)
mf.kernel()

# Calculate each fragment:
dmet = vayesta.dmet.DMET(mf, solver='FCI', maxiter=1)
with dmet.site_fragmentation() as f:
    for site in range(0, nsite, nimp):
        f.add_atomic_fragment(list(range(site, site + nimp)))
dmet.kernel()

# Calculate a single fragment and use translational symmetry:
dmet_sym = vayesta.dmet.DMET(mf, solver='FCI', maxiter=1)
dmet_sym.symmetry.set_translations([nsites[0]//impshape[0], nsites[1]//impshape[1], 1])
with dmet_sym.site_fragmentation() as f:
    frag = f.add_atomic_fragment(list(range(nimp)))
dmet_sym.kernel()

# Compare converged correlation potential
print("Difference in converged correlation potentials with and without translational symmetry:")
print("|d(V_corr)|= %.5e" % np.linalg.norm(dmet.vcorr - dmet_sym.vcorr))
