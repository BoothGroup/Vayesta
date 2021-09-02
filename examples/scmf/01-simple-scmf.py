import numpy as np

import pyscf
import pyscf.gto
import pyscf.scf
import pyscf.cc

import vayesta
import vayesta.ewf


mol = pyscf.gto.Mole()
mol.atom = """
O  0.0000   0.0000   0.1173
H  0.0000   0.7572  -0.4692
H  0.0000  -0.7572  -0.4692
"""
mol.basis = 'cc-pVDZ'
mol.build()

# Hartree-Fock
mf = pyscf.scf.RHF(mol)
mf.kernel()

# CCSD
cc = pyscf.cc.CCSD(mf)
cc.kernel()

ecc = vayesta.ewf.EWF(mf, make_rdm1=True, fragment_type='Lowdin-AO', bath_type=None)
ecc.make_all_atom_fragments()

ecc = ecc.pdmet_scmf()
#ecc = ecc.brueckner_scmf()
ecc.kernel()
assert ecc.with_scmf.converged
e_0 = ecc.with_scmf.e_tot_oneshot
e_sc = ecc.with_scmf.e_tot

print("E%-14s %+16.8f Ha" % ('(HF)=', mf.e_tot))
print("E%-14s %+16.8f Ha" % ('(EWF-CCSD)=', e_0))
print("E%-14s %+16.8f Ha" % ('(SC-EWF-CCSD)=', e_sc))

with open("energies.txt", 'a') as f:
    f.write('%.2f  %+16.8f  %+16.8f  %+16.8f  %+16.8f\n' % (a, mf.e_tot, cc.e_tot, e_0, e_sc))
