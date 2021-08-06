import numpy as np

import pyscf.cc
import pyscf.tools
import pyscf.tools.ring

import vayesta.dmet

natom = 6
d = 2.0
ring = pyscf.tools.ring.make(natom, d)
atom = [('H %f %f %f' % xyz) for xyz in ring]

mol = pyscf.gto.Mole()
mol.atom = atom
mol.basis = 'sto-3g'
mol.verbose = 10
mol.output = 'pyscf_out.txt'
mol.build()

# Hartree-Fock
mf = pyscf.scf.RHF(mol)
mf.kernel()

# DMET calculation with DIIS extrapolation of the high-level correlation potential.
dmet_diis = vayesta.dmet.DMET(mf, solver='FCI', fragment_type='IAO', charge_consistent=False, diis=True,
                              max_elec_err = 1e-6)
dmet_diis.make_atom_fragment([0,1]); dmet_diis.make_atom_fragment([2,3]); dmet_diis.make_atom_fragment([4,5])
dmet_diis.kernel()

# DMET calculation without DIIS, using
dmet_mix = vayesta.dmet.DMET(mf, solver='FCI', fragment_type='IAO', charge_consistent=False, diis=False,
                             max_elec_err = 1e-6)
dmet_mix.make_atom_fragment([0,1]); dmet_mix.make_atom_fragment([2,3]); dmet_mix.make_atom_fragment([4,5])
dmet_mix.kernel()

ediff = abs(dmet_diis.e_dmet - dmet_mix.e_dmet)
vdiff = sum(np.ravel(dmet_diis.vcorr - dmet_mix.vcorr)**2)**(0.5)

print("Difference between DIIS and mixing solution in          Energy          = {:6.4e}".format(ediff))
print("                                               Correlation Potential L2 = {:6.4e}".format(vdiff))