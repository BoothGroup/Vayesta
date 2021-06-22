import pyscf
import pyscf.gto
import pyscf.scf

import vayesta
import vayesta.ewf

mol = pyscf.gto.Mole()
mol.atom = """
Se 0 0      0
H 0 -2.757 2.587
H 0  2.757 2.587"""
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
ecc.make_atom_fragment([0, 1])  # Se-H in one fragment
ecc.make_atom_fragment(2)       # Other H atom
ecc.kernel()
print("E%-11s %+16.8f Ha" % ('(EWF-CCSD)=', ecc.e_tot))

# Using atom labels
ecc = vayesta.ewf.EWF(mf, bno_threshold=1e-6)
ecc.make_atom_fragment('Se')
ecc.make_atom_fragment('H')     # Both H atoms!
ecc.kernel()
print("E%-11s %+16.8f Ha" % ('(EWF-CCSD)=', ecc.e_tot))

# Atomic orbital fragments
ecc = vayesta.ewf.EWF(mf, bno_threshold=1e-6)
ecc.make_ao_fragment([0,1])                         # 1s and 2s at Se
ecc.make_ao_fragment(["0 Se 3dz", "0 Se 3dx2-y2"])  # 3dz^2 and 3dx^2-y^2 at Se
ecc.kernel()
print("E%-11s %+16.8f Ha" % ('(EWF-CCSD)=', ecc.e_tot))

# Lowdin-AO fragmentation

# Atomic fragments
ecc = vayesta.ewf.EWF(mf, bno_threshold=1e-6, fragment_type='Lowdin-AO')
ecc.make_atom_fragment([0, 1])  # Se-H
ecc.make_atom_fragment(2)       # Other H atom
ecc.kernel()
print("E%-11s %+16.8f Ha" % ('(EWF-CCSD)=', ecc.e_tot))
