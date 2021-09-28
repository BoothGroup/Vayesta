import pyscf
import pyscf.gto
import pyscf.scf
import pyscf.mcscf

import vayesta
import vayesta.ewf

mol = pyscf.gto.Mole()
mol.atom = ['N 0.0 0.0 0.0', 'N 0.0, 0.0, 2.0']
mol.basis = 'aug-cc-pvdz'
mol.verbose = 10
mol.output = 'pyscf_out.txt'
mol.build()

# Hartree-Fock
mf = pyscf.scf.RHF(mol)
mf.kernel()

# Reference CASCI
casci = pyscf.mcscf.CASCI(mf, 8, 10)
casci.kernel()

# Reference CASSCF
casscf = pyscf.mcscf.CASSCF(mf, 8, 10)
casscf.kernel()

ecc = vayesta.ewf.EWF(mf, solver='FCI', bno_threshold=100)
ecc.iao_fragmentation()
ecc.add_atomic_fragment(0, sym_factor=2)
ecc.kernel()

print("E(HF)=       %+16.8f Ha" % mf.e_tot)
print("E(CASCI)=    %+16.8f Ha" % casci.e_tot)
print("E(CASSCF)=   %+16.8f Ha" % casscf.e_tot)
print("E(EWF-CCSD)= %+16.8f Ha" % ecc.e_tot)
