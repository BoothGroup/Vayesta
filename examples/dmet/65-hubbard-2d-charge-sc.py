# Compare to PySCF:
# pyscf/examples/scf/40-customizing_hamiltonian.py

import numpy as np
import vayesta
import vayesta.dmet
import vayesta.lattmod


# In the Hubbard model charge consistency shouldn't change obtained results.
nsites = [6,6]
impshape = [2,2]
hubbard_u = 6.0
# For the 2D Hubbard model, APBC and mixed boundaries do not work at the moment!
boundary = 'PBC'
nsite = nsites[0]*nsites[1]
nelectron = nsite-10
nimp = impshape[0]*impshape[1]
mol = vayesta.lattmod.Hubbard2D(nsites, nelectron=nelectron, hubbard_u=hubbard_u, boundary=boundary, tiles=impshape, output='pyscf.out')
mf = vayesta.lattmod.LatticeMF(mol)
mf.kernel()

# Calculate a single fragment and use translational symmetry:
dmet_cc = vayesta.dmet.DMET(mf, solver='FCI', charge_consistent=True)
with dmet_cc.site_fragmentation() as f:
    f_cc = f.add_atomic_fragment(list(range(nimp)))
# Add fragments which are translationally symmetric to f - the results of the fragment f
# fill be automatically copied.
symfrags_cc = f_cc.make_tsymmetric_fragments(tvecs=[nsites[0]//impshape[0], nsites[1]//impshape[1], 1])
# Check that every fragment has been identified!
assert (len(symfrags_cc)+1 == nsite//nimp)
dmet_cc.kernel()

# Calculate a single fragment and use translational symmetry:
dmet_nocc = vayesta.dmet.DMET(mf, solver='FCI', charge_consistent=False)
with dmet_nocc.site_fragmentation() as f:
    f_nocc = f.add_atomic_fragment(list(range(nimp)))
# Add fragments which are translationally symmetric to f - the results of the fragment f
# fill be automatically copied.
symfrags_nocc = f_nocc.make_tsymmetric_fragments(tvecs=[nsites[0]//impshape[0], nsites[1]//impshape[1], 1])
# Check that every fragment has been identified!
assert (len(symfrags_nocc)+1 == nsite//nimp)
dmet_nocc.kernel()

# Compare converged correlation potential
print("Difference in converged energy with and without charge consistency: {:6.4e}".format(
                    abs(dmet_cc.e_dmet - dmet_nocc.e_dmet)))
