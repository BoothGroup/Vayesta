import numpy as np
import pyscf
import pyscf.gto
import pyscf.scf
import pyscf.cc
import vayesta
import vayesta.ewf


mol = pyscf.gto.Mole()
mol.atom = 'Li 0 0 0 ; Li 0 0 3.0'
mol.basis = 'cc-pVTZ'
mol.verbose = 10
mol.output = 'pyscf_out.txt'
mol.build()

# Hartree-Fock
mf = pyscf.scf.RHF(mol)
mf.kernel()

# Reference full system CCSD:
cc = pyscf.cc.CCSD(mf)
cc.kernel()

# Test exact limit by using the full_bath
emb = vayesta.ewf.EWF(mf, bath_options=dict(bathtype='full'))
emb.kernel()
assert abs(cc.e_tot-emb.e_tot) < 1e-8

# Use MP2 bath orbitals [default]:
# threshold is the BNO occupation threshold
emb_mp2 = vayesta.ewf.EWF(mf, bath_options=dict(bathtype='mp2', threshold=1e-4))
emb_mp2.kernel()

# Use maximally R^2-localized bath orbitals:
# rcut is the cutoff distance in Angstrom
emb_r2 = vayesta.ewf.EWF(mf, bath_options=dict(bathtype='r2', rcut=2.0))
emb_r2.kernel()

# Occupied and virtual bath can be different:
bath = dict(bathtype_occ='r2', rcut_occ=2.0, bathtype_vir='mp2', threshold_vir=1e-4)
emb_mix = vayesta.ewf.EWF(mf, bath_options=bath)
emb_mix.kernel()


print("E(CCSD)=       %+16.8f Ha" % cc.e_tot)
print("Embedded CCSD")
print("E(full bath)=  %+16.8f Ha" % emb.e_tot)
print("E(MP2 bath)=   %+16.8f Ha" % emb_mp2.e_tot)
print("E(R2 bath)=    %+16.8f Ha" % emb_r2.e_tot)
print("E(mixed bath)= %+16.8f Ha" % emb_mix.e_tot)
