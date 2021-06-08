import pyscf
import pyscf.gto
import pyscf.scf
import pyscf.cc

import vayesta
import vayesta.ewf

mol = pyscf.gto.Mole()
mol.atom = """
O 0 0      0
H1 0 -2.757 2.587
H2 0  2.757 2.587"""
mol.basis = 'aug-cc-pvdz'
mol.verbose = 10
mol.output = 'pyscf_out.txt'
mol.build()

# Hartree-Fock
mf = pyscf.scf.RHF(mol)
mf.kernel()
print("E%-11s %+16.8f Ha" % ('(HF)=', mf.e_tot))

# IAO fragmentations:

# Atomic fragments
ecc = vayesta.ewf.EWF(mf, bno_threshold=1e-6)
ecc.make_atom_fragment([0, 1])  # OH-group
ecc.make_atom_fragment(2)       # H2 atom
ecc.kernel()
print("E%-11s %+16.8f Ha" % ('(EWF-CCSD)=', ecc.e_tot))

# Atomic fragments using atom labels
ecc = vayesta.ewf.EWF(mf, bno_threshold=1e-6)
ecc.make_atom_fragment(["O", "H1"])     # OH-group
ecc.make_atom_fragment("H2")            # H2 atom
ecc.kernel()
print("E%-11s %+16.8f Ha" % ('(EWF-CCSD)=', ecc.e_tot))

# Atomic orbital fragments
ecc = vayesta.ewf.EWF(mf, bno_threshold=1e-6)
ecc.make_ao_fragment([1])                       # 2s at O
ecc.make_ao_fragment(["O 1s", "H.*1s"])         # 1s at O,H1,H2
ecc.make_ao_fragment("O 2p.*")                 # 2px, 2py, 2pz at O
ecc.kernel()
print("E%-11s %+16.8f Ha" % ('(EWF-CCSD)=', ecc.e_tot))

# Lowdin-AO fragmentation

# Atomic fragments
ecc = vayesta.ewf.EWF(mf, bno_threshold=1e-6, fragment_type='Lowdin-AO')
ecc.make_atom_fragment([0, 1])  # OH-group
ecc.make_atom_fragment(2)       # H2 atom
ecc.kernel()
print("E%-11s %+16.8f Ha" % ('(EWF-CCSD)=', ecc.e_tot))
