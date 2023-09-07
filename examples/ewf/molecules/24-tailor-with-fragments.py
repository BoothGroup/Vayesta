import numpy as np
import pyscf
from pyscf.scf import UHF
from pyscf.gto import Mole
from pyscf.cc import CCSD
import vayesta
from vayesta.ewf import EWF


mol = Mole()
mol.atom = """
    Fe 0 0 0
    O 0 0 1.616
"""
mol.spin = 4  # 3d6 Fe(II)
mol.basis = "6-31g"
mol.build()

mf = UHF(mol)
mf.max_cycle = 300
mf.kernel()
dm1 = mf.make_rdm1()
while True:
    mo1 = mf.stability()[0]
    stable = mo1 is mf.mo_coeff
    dm1 = mf.make_rdm1(mo_coeff=mo1)
    if stable:
        break
    mf.kernel(dm1)
assert mf.converged
# Reference full system CCSD:
cc = CCSD(mf)
cc.kernel()

output = [f"Hartree Fock Energy={mf.e_tot}, CCSD Energy={cc.e_tot if cc.converged else np.nan}"]
for projectors in (0, 1, 2):
    # Tailor CCSD with an atomic FCI fragment (projected onto fragment space)
    # T1 and T2 amplitudes of the FCI fragment are used to tailor the CCSD amplitudes. setting auxilary to true,
    # makes sure that the FCI fragment is solved first, but does not contribute to expectation values. Note that
    # projectors = 0 is only suitable for calculations with only a single FCI fragment. Because of overlapping bath
    # spaces, for multiple constraining fragments proj=0 will double-count and therefore projectors = 1 or 2 should be used.
    emb = EWF(mf)
    fci_frags = []
    with emb.iao_fragmentation() as f:
        fci_frags.append(
            f.add_atomic_fragment(
                ["Fe"],
                orbital_filter=["Fe 3d"],
                solver="FCI",
                store_wf_type="CCSDTQ",
                bath_options=dict(bathtype="dmet"),
                auxiliary=True,
            )
        )
        ccsd = f.add_full_system(solver="extCCSD", bath_options=dict(bathtype="full"))
    ccsd.add_external_corrections(fci_frags, correction_type="tailor", projectors=projectors)
    emb.kernel()
    output.append(f"Projectors={projectors}, Tailored CC Energy={emb.e_tot if emb.converged else np.nan} ")

for line in output:
    print(line)
