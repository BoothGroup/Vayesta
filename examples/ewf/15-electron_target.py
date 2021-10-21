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
mol.output = 'pyscf-01.out'
mol.build()

# Hartree-Fock
mf = pyscf.scf.RHF(mol)
mf.conv_tol = 1e-12
mf.kernel()
assert mf.converged

# CCSD
cc = pyscf.cc.CCSD(mf)
cc.kernel()

bno = 1e-6
ecc = vayesta.ewf.EWF(mf, bno_threshold=bno, make_rdm1=True)
ecc.sao_fragmentation()
ecc.add_all_atomic_fragments()
ecc.kernel()

dm1 = mf.make_rdm1()
ecc.pop_analysis(dm1, filename='pop-mf.txt')

dm1 = cc.make_rdm1(ao_repr=True)
ecc.pop_analysis(dm1, filename='pop-cc.txt')

dm1 = ecc.make_rdm1_demo(ao_basis=True)
ecc.pop_analysis(dm1, filename='pop-pdm.txt')

dm1 = ecc.make_rdm1_ccsd(ao_basis=True)
ecc.pop_analysis(dm1, filename='pop-pwf.txt')


print("E%-11s %+16.8f Ha" % ('(HF)=', mf.e_tot))
print("E%-11s %+16.8f Ha" % ('(EWF-CCSD)=', ecc.e_tot))

ecc2 = vayesta.ewf.EWF(mf, bno_threshold=bno, make_rdm1=True)
ecc2.sao_fragmentation()
ecc2.add_atomic_fragment(0, nelectron_target=8)
ecc2.add_atomic_fragment(1, nelectron_target=1)
ecc2.add_atomic_fragment(2, nelectron_target=1)
ecc2.kernel()

dm1 = ecc2.make_rdm1_demo(ao_basis=True)
ecc2.pop_analysis(dm1, filename='pop-pdm-cpt.txt')

dm1 = ecc2.make_rdm1_ccsd(ao_basis=True)
ecc2.pop_analysis(dm1, filename='pop-pwf-cpt.txt')

print("E%-11s %+16.8f Ha" % ('(HF)=', mf.e_tot))
print("E%-11s %+16.8f Ha" % ('(EWF-CCSD)=', ecc2.e_tot))
