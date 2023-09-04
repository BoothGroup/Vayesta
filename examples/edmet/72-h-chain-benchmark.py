import numpy as np
import pyscf.cc
import vayesta.dmet
import vayesta.edmet


natom = 10

for d in np.arange(1.0, 3.7, 0.4):
    pos = [(d * x, 0, 0) for x in range(natom)]
    atom = [("H %f %f %f" % xyz) for xyz in pos]

    mol = pyscf.gto.Mole()
    mol.atom = atom
    mol.basis = "cc-pvdz"
    mol.unit = "Bohr"
    mol.build()

    # Hartree-Fock
    mf = pyscf.scf.RHF(mol)
    mf.kernel()

    # Reference full system CCSD:
    mycc = pyscf.cc.CCSD(mf)
    mycc.kernel()

    # Single-shot
    dmet_oneshot = vayesta.dmet.DMET(mf, solver="FCI", max_elec_err=1e-6, maxiter=1)
    with dmet_oneshot.iao_fragmentation() as f:
        f.add_all_atomic_fragments()
    dmet_oneshot.kernel()
    # Single-shot EDMET
    edmet_oneshot = vayesta.edmet.EDMET(
        mf, solver="FCI", max_elec_err=1e-6, oneshot=True, solver_options=dict(max_boson_occ=4)
    )
    with edmet_oneshot.iao_fragmentation() as f:
        f.add_all_atomic_fragments()
    edmet_oneshot.kernel()

    e_cc = mycc.e_tot if mycc.converged else np.NaN
    print("E%-14s %+16.8f Ha" % ("(HF)=", mf.e_tot))
    print("E%-14s %+16.8f Ha" % ("(CCSD)=", e_cc))
    print("E%-14s %+16.8f Ha" % ("(DMET-FCI-Oneshot)=", dmet_oneshot.e_tot))
    print("E%-14s %+16.8f Ha" % ("(EDMET-FCI-Oneshot)=", edmet_oneshot.e_tot))

    with open("energies_h10_ccpvdz.txt", "a") as f:
        f.write(
            "%.2f  % 16.8f  % 16.8f  %16.8f  %16.8f\n" % (d, mf.e_tot, e_cc, dmet_oneshot.e_tot, edmet_oneshot.e_tot)
        )
