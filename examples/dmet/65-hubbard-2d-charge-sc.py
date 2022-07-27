import numpy as np
import vayesta
import vayesta.dmet
import vayesta.lattmod


# In the Hubbard model charge consistency should not change obtained results.
nsites = [6,6]
nimps = [2,2]
hubbard_u = 6.0
# Anti-PBC and mixed boundaries do not work with translational symmetry!
boundary = 'PBC'
nsite = nsites[0]*nsites[1]
nelectron = nsite-10
nimp = nimps[0]*nimps[1]
nimages = [nsites[0]//nimps[0], nsites[1]//nimps[1], 1]
mol = vayesta.lattmod.Hubbard2D(nsites, nelectron=nelectron, hubbard_u=hubbard_u, boundary=boundary, tiles=nimps)
mf = vayesta.lattmod.LatticeMF(mol)
mf.kernel()

# DMET with charge conistency (default):
dmet = vayesta.dmet.DMET(mf, solver='FCI')
dmet.symmetry.set_translations(nimages)
with dmet.site_fragmentation() as f:
    f.add_atomic_fragment(list(range(nimp)))
dmet.kernel()

# DMET without charge conistency:
dmet_nocc = vayesta.dmet.DMET(mf, solver='FCI', charge_consistent=False)
dmet_nocc.symmetry.set_translations(nimages)
with dmet_nocc.site_fragmentation() as f:
    f_nocc = f.add_atomic_fragment(list(range(nimp)))
dmet_nocc.kernel()

print("Difference in converged solutions:")
print("  |d(E_tot)|=  %.5e" % abs(dmet.e_tot - dmet_nocc.e_tot))
print("  |d(V_corr)|= %.5e" % np.linalg.norm(dmet.vcorr - dmet_nocc.vcorr))
