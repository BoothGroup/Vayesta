import numpy as np
import vayesta
import vayesta.dmet
import vayesta.lattmod


nsites = [6,6]
nimps = [2,2]
hubbard_u = 6.0
# Anti-PBC and mixed boundaries do not work with translational symmetry!
boundary = 'PBC'
nsite = nsites[0]*nsites[1]
nelectron = nsite-10
nimp = nimps[0]*nimps[1]
mol = vayesta.lattmod.Hubbard2D(nsites, nelectron=nelectron, hubbard_u=hubbard_u, boundary=boundary, tiles=nimps)
mf = vayesta.lattmod.LatticeMF(mol)
mf.kernel()

# Calculate each (2 x 2)-sites fragment:
# (Note that the Hubbard 2D class will reorder the sites, such that all 2 x 2 tiles contain continuous
# site indices)
dmet = vayesta.dmet.DMET(mf, solver='FCI', maxiter=1)
with dmet.site_fragmentation() as f:
    for site in range(0, nsite, nimp):
        f.add_atomic_fragment(list(range(site, site+nimp)))
dmet.kernel()

# Calculate a single fragment and use translational symmetry:
dmet_sym = vayesta.dmet.DMET(mf, solver='FCI', maxiter=1)
# Specify the number of translational copies in direction of the three lattice vectors by passing a list with three
# integers, [n0, n1, n2]. 1D or 2D systems have their periodic dimension along the first one or two axes.
nimages = [nsites[0]//nimps[0], nsites[1]//nimps[1], 1]
dmet_sym.symmetry.set_translations(nimages)
# Add only a single (2 x 2)-sites fragment:
with dmet_sym.site_fragmentation() as f:
    frag = f.add_atomic_fragment(list(range(nimp)))
dmet_sym.kernel()

print("Difference in converged solutions:")
print("  |d(E_tot)|=  %.5e" % abs(dmet.e_tot - dmet_sym.e_tot))
print("  |d(V_corr)|= %.5e" % np.linalg.norm(dmet.vcorr - dmet_sym.vcorr))
