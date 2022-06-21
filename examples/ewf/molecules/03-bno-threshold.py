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
mol.basis = 'aug-cc-pVDZ'
mol.output = 'pyscf.out'
mol.build()

# Hartree-Fock
mf = pyscf.scf.RHF(mol)
mf.kernel()

# Embedded CCSD
# We can perform multiple calculations on the same EWF object:
eta_list = [1e-5, 1e-6, 1e-7, 1e-8]
energies = []
emb = vayesta.ewf.EWF(mf)
for eta in eta_list:
    # Change options allows to change options between calculations
    emb.change_options(bath_options=dict(threshold=eta))
    # When running multiple calculations with the same embedding object,
    # it is important to reset the calculations between runs.
    # This will be done automatically, if not done by the user, however
    # it may delete some parts of the calculations which we may want to reuse
    # to save time. In this case we can call `emb.reset()` ourselves, and specifiy to keep
    # the DMET bath and occupied/virtual bath constructors by setting `reset_bath=False`.
    # Other options are `reset_cluster` and `reset_eris`.
    emb.reset(reset_bath=False)
    emb.kernel()
    energies.append(emb.e_tot)

print("E(HF)=        %+16.8f Ha" % mf.e_tot)
for i, eta in enumerate(eta_list):
    print("E(eta=%.0e)= %+16.8f Ha" % (eta, energies[i]))

# Reference full system CCSD:
cc = pyscf.cc.CCSD(mf)
cc.kernel()
print("E(CCSD)=      %+16.8f Ha" % cc.e_tot)
