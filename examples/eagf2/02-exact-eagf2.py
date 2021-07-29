import numpy as np

import pyscf
import pyscf.gto
import pyscf.scf

import vayesta
import vayesta.eagf2

mol = pyscf.gto.Mole()
mol.atom = 'Li 0 0 0; H 0 0 1.4'
mol.basis = 'aug-cc-pvdz'
mol.verbose = 10
mol.build()

mf = pyscf.scf.RHF(mol)
mf.kernel()

gf2 = vayesta.eagf2.RAGF2(mf)
gf2.kernel()

egf2 = vayesta.eagf2.EAGF2(mf, bno_threshold=-1)
egf2.make_atom_fragment(0)
egf2.make_atom_fragment(1)
egf2.kernel()

print("%-3s  = %+16.8f Ha" % ('E', egf2.e_tot))
print("%-3s  = %+16.8f Ha" % ('IP', egf2.e_ip))
print("%-3s  = %+16.8f Ha" % ('EA', egf2.e_ea))

# Energies will not be exact due to the definition of the two-body
# energy in the Galitskii-Migdal formula, but IP and EA will be.
assert np.allclose(gf2.e_ip, egf2.e_ip)
assert np.allclose(gf2.e_ea, egf2.e_ea)
