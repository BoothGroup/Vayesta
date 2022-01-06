# BROKEN

import pyscf
import pyscf.gto
import pyscf.scf
import pyscf.cc

import vayesta
import vayesta.ewf

mol = pyscf.gto.Mole()
mol.atom = ['N 0.0 0.0 0.0', 'N 0.0, 0.0, 1.1']
mol.basis = 'aug-cc-pvtz'
mol.verbose = 10
mol.output = 'pyscf_out.txt'
mol.build()

# Hartree-Fock
mf = pyscf.scf.RHF(mol)
mf.kernel()

# Reference full system CCSD:
cc = pyscf.cc.CCSD(mf)
cc.kernel()

# Reference one-shot EWF-CCSD
ecc = vayesta.ewf.EWF(mf, bno_threshold=1e-4)
ecc.kernel()

# Enable self-consistency (SC) mode by setting sc_mode:
# sc_mode = 1: Project both first and second occupied index of other's fragment T2-amplitude onto it's fragment space
# (conservative external correction)
# sc_mode = 2: Project only the first occupied index of other's fragment T2-amplitude onto it's fragment space
# (more substantional external correction)
scecc = vayesta.ewf.EWF(mf, bno_threshold=1e-4, sc_mode=1)
scecc.iao_fragmentation()
f0 = scecc.add_atomic_fragment(0)
f1 = scecc.add_atomic_fragment(1)
# Additional we need to define which fragment is considered in the tailoring of each other fragment:
f0.add_tailor_fragment(f1)
f1.add_tailor_fragment(f0)
# If each fragment should be tailored by all others (like here), you can also call ecc.tailor_all_fragments()
scecc.kernel()

print("E%-14s %+16.8f Ha" % ('(HF)=', mf.e_tot))
print("E%-14s %+16.8f Ha" % ('(CCSD)=', cc.e_tot))
print("E%-14s %+16.8f Ha" % ('(EWF-CCSD)=', ecc.e_tot))
print("E%-14s %+16.8f Ha" % ('(SC-EWF-CCSD)=', scecc.e_tot))
