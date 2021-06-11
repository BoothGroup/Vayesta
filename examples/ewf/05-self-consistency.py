import pyscf
import pyscf.gto
import pyscf.scf
import pyscf.cc

import vayesta
import vayesta.ewf

mol = pyscf.gto.Mole()
#mol.atom = ['Li 0.0 0.0 0.0', 'H 0.0, 0.0, 2.0']
mol.atom = ['N 0.0 0.0 0.0', 'N 0.0, 0.0, 1.1']
mol.basis = 'aug-cc-pvtz'
mol.verbose = 10
mol.output = 'pyscf_out.txt'
mol.build()

# Hartree-Fock
mf = pyscf.scf.RHF(mol)
mf.kernel()

# Reference full system CCSD:
#cc = pyscf.cc.CCSD(mf)
#cc.kernel()
#e_cc = cc.e_tot

bno_thr = 1e-4

ecc = vayesta.ewf.EWF(mf, bno_threshold=bno_thr)
ecc.make_all_atom_fragments()
ecc.kernel()
e_ewf = ecc.e_tot

ecc = vayesta.ewf.EWF(mf, bno_threshold=bno_thr, sc_mode=True)
f0 = ecc.make_atom_fragment(0)
f1 = ecc.make_atom_fragment(1)

# Fragments tailor each other:
f0.add_tailor_fragment(f1)
f1.add_tailor_fragment(f0)

ecc.kernel()

print("E%-14s %+16.8f Ha" % ('(HF)=', mf.e_tot))
#print("E%-14s %+16.8f Ha" % ('(CCSD)=', e_cc))
print("E%-14s %+16.8f Ha" % ('(EWF-CCSD)=', e_ewf))
print("E%-14s %+16.8f Ha" % ('(SC-EWF-CCSD)=', ecc.e_tot))
