import numpy as np

import vayesta
import vayesta.dmet
from vayesta.misc.molecules import ring


mol = pyscf.gto.Mole()
mol.atom = ring('H', 6, 2.0)
mol.basis = 'sto-3g'
mol.output = 'pyscf.out'
mol.verbose = 4
mol.build()

# Hartree-Fock
mf = pyscf.scf.RHF(mol)
mf.kernel()

# DMET calculation with DIIS extrapolation of the high-level correlation potential.
dmet_diis = vayesta.dmet.DMET(mf, solver='FCI', charge_consistent=False, diis=True,
                              max_elec_err=1e-6)
with dmet_diis.iao_fragmentation() as f:
    f.add_atomic_fragment([0, 1])
    f.add_atomic_fragment([2, 3])
    f.add_atomic_fragment([4, 5])
dmet_diis.kernel()

# DMET calculation without DIIS, using
dmet_mix = vayesta.dmet.DMET(mf, solver='FCI', charge_consistent=False, diis=False,
                             max_elec_err=1e-6)
with dmet_mix.iao_fragmentation() as f:
    f.add_atomic_fragment([0, 1])
    f.add_atomic_fragment([2, 3])
    f.add_atomic_fragment([4, 5])
dmet_mix.kernel()

ediff = (dmet_diis.e_dmet - dmet_mix.e_dmet)
vdiff = np.linalg.norm(dmet_diis.vcorr - dmet_mix.vcorr)

print("Difference between DIIS and mixing solution:")
print("delta(Energy)=   %.8f" % ediff)
print("|delta(V_corr)|= %.8f" % vdiff)
