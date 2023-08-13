import pyscf.gto
import pyscf.scf
import pyscf.cc
import pyscf.mcscf
import vayesta.ewf
from vayesta.misc import molecules

mol = pyscf.gto.Mole()
mol.atom = molecules.arene(6)
mol.basis = 'cc-pVDZ'
mol.output = 'pyscf.out'
mol.build()

mf = pyscf.scf.RHF(mol).density_fit()
mf.kernel()

ncas = (4,4)

# Reference CASCI
mycasci = pyscf.mcscf.CASCI(mf, *ncas)
mycasci.kernel()

# Equivalent CAS calculation with bare interactions
casci_bare = vayesta.ewf.EWF(mf, bath_options=dict(bathtype="dmet"), screening=None, solver="FCI")
with casci_bare.cas_fragmentation() as f:
    f.add_cas_fragment(*ncas)
casci_bare.kernel()

# CAS calculation with cRPA screening
casci_crpa = vayesta.ewf.EWF(mf, bath_options=dict(bathtype="dmet"), screening="crpa", solver="FCI")
with casci_crpa.cas_fragmentation() as f:
    f.add_cas_fragment(*ncas)
casci_crpa.kernel()

# CAS calculation with mRPA screening
casci_mrpa = vayesta.ewf.EWF(mf, bath_options=dict(bathtype="dmet"), screening="mrpa", solver="FCI")
with casci_mrpa.cas_fragmentation() as f:
    f.add_cas_fragment(*ncas)
casci_mrpa.kernel()

print("E(HF)=                              %+16.8f Ha" % mf.e_tot)
print("ΔE(CASCI)=                          %+16.8f Ha" % (mycasci.e_tot-mf.e_tot))
print("ΔE(Emb. CASCI, bare interactions)=  %+16.8f Ha  (diff= %+.8f Ha)" % (casci_bare.e_tot-mf.e_tot, casci_bare.e_tot-mycasci.e_tot))
print("ΔE(Emb. CASCI, cRPA screening)=     %+16.8f Ha  (diff= %+.8f Ha)" % (casci_crpa.e_tot-mf.e_tot, casci_crpa.e_tot-mycasci.e_tot))
print("ΔE(Emb. CASCI, mRPA screening)=     %+16.8f Ha  (diff= %+.8f Ha)" % (casci_mrpa.e_tot-mf.e_tot, casci_mrpa.e_tot-mycasci.e_tot))
