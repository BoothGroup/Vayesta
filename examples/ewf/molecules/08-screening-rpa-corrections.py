import pyscf.gto
import pyscf.scf
import pyscf.cc
import vayesta.ewf


mol = pyscf.gto.Mole()
mol.atom = """
O  0.0000   0.0000   0.1173
H  0.0000   0.7572  -0.4692
H  0.0000  -0.7572  -0.4692
"""
mol.basis = 'cc-pVTZ'
mol.output = 'pyscf.out'
mol.build()

# Hartree-Fock
mf = pyscf.scf.RHF(mol).density_fit()
mf.kernel()

# Reference full system CCSD:
cc = pyscf.cc.CCSD(mf)
cc.kernel()

eta = 1e-6

# Embedded CCSD calculation with bare interactions and no energy correction.
emb_bare = vayesta.ewf.EWF(mf, bath_options=dict(threshold=eta))
emb_bare.kernel()

# Embedded CCSD with mRPA screened interactions and RPA cumulant approximation for nonlocal correlations.
emb = vayesta.ewf.EWF(mf, bath_options=dict(threshold=eta), screening="mrpa", ext_rpa_correction="cumulant")
emb.kernel()
e_nonlocal_cumulant = emb.e_nonlocal
# Embedded CCSD with mRPA screened interactions and delta RPA approximation for nonlocal correlations.
emb = vayesta.ewf.EWF(mf, bath_options=dict(threshold=eta), screening="mrpa", ext_rpa_correction="erpa")
emb.kernel()
e_nonlocal_erpa = emb.e_nonlocal

# Note that mRPA screening and external corrections often cancel with each other in the case of the energy.
print("E(CCSD)=                              %+16.8f Ha" % cc.e_tot)
print("E(RPA)=                               %+16.8f Ha  (error= %+.8f Ha)" % (emb.e_mf + emb.e_rpa,
                                                                               emb.e_mf + emb.e_rpa - cc.e_tot))
print("E(Emb. CCSD)=                         %+16.8f Ha  (error= %+.8f Ha)" % (emb_bare.e_tot, emb_bare.e_tot-cc.e_tot))
print("E(Emb. Screened CCSD)=                %+16.8f Ha  (error= %+.8f Ha)" % (emb.e_tot, emb.e_tot-cc.e_tot))
print("E(Emb. Screened CCSD + \Delta E_k)=   %+16.8f Ha  (error= %+.8f Ha)" % (emb.e_tot+e_nonlocal_cumulant,
                                                                               emb.e_tot+e_nonlocal_cumulant-cc.e_tot))
print("E(Emb. Screened CCSD + \Delta RPA)=   %+16.8f Ha  (error= %+.8f Ha)" % (emb.e_tot+e_nonlocal_erpa,
                                                                               emb.e_tot+e_nonlocal_erpa-cc.e_tot))
