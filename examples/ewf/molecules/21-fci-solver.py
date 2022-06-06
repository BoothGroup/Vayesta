import pyscf
import pyscf.gto
import pyscf.scf
import pyscf.mcscf

import vayesta
import vayesta.ewf

mol = pyscf.gto.Mole()
mol.atom = ['N 0 0 0', 'N 0 0 2']
mol.basis = 'aug-cc-pvdz'
mol.output = 'pyscf.out'
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

# FCI with DMET bath orbitals only
emb = vayesta.ewf.EWF(mf, solver='FCI', bath_options=dict(bathtype='dmet'))
with emb.iao_fragmentation() as f:
    f.add_atomic_fragment(0, sym_factor=2)
emb.kernel()

print("E(HF)=        %+16.8f Ha" % mf.e_tot)
print("E(CASCI)=     %+16.8f Ha" % casci.e_tot)
print("E(CASSCF)=    %+16.8f Ha" % casscf.e_tot)
print("E(Emb. CCSD)= %+16.8f Ha" % emb.e_tot)
