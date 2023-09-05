import numpy as np
import vayesta
import vayesta.dmet
import vayesta.lattmod


nsite = 10
nimp = 2
hubbard_u = 6.0
boundary = "PBC"
nelectron = nsite
mol = vayesta.lattmod.Hubbard1D(nsite, nelectron=nelectron, hubbard_u=hubbard_u, boundary=boundary)
mf = vayesta.lattmod.LatticeMF(mol)
mf.kernel()

# Calculate each 2-sites fragment:
dmet = vayesta.dmet.DMET(mf, solver="FCI")
with dmet.site_fragmentation() as f:
    for site in range(0, nsite, nimp):
        f.add_atomic_fragment(list(range(site, site + nimp)))
dmet.kernel()

print(dmet.fragments)
# Calculate a single fragment and use translational symmetry:
dmet_sym = vayesta.dmet.DMET(mf, solver="FCI")
# Specify the number of translational copies in direction of the three lattice vectors by passing a list with three
# integers, [n0, n1, n2]. 1D or 2D systems have their periodic dimension along the first one or two axes.
nimages = [nsite // nimp, 1, 1]
dmet_sym.symmetry.set_translations(nimages)
# Add only a single 2-sites fragment:
with dmet_sym.site_fragmentation() as f:
    f.add_atomic_fragment(list(range(nimp)))
dmet_sym.kernel()

print("Difference in converged solutions:")
print("  |d(E_tot)|=  %.5e" % abs(dmet.e_tot - dmet_sym.e_tot))
print("  |d(V_corr)|= %.5e" % np.linalg.norm(dmet.vcorr - dmet_sym.vcorr))
