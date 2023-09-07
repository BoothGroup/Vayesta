import numpy as np
import pyscf.cc
import pyscf.fci
import pyscf.tools
import pyscf.tools.ring
import vayesta.dmet


natom = 6

for d in np.arange(0.5, 3.0001, 0.25):
    ring = pyscf.tools.ring.make(natom, d)
    atom = [("H %f %f %f" % xyz) for xyz in ring]

    mol = pyscf.gto.Mole()
    mol.atom = atom
    mol.basis = "6-31G"
    mol.output = "pyscf.out"
    mol.verbose = 4
    mol.build()

    # Hartree-Fock:
    mf = pyscf.scf.RHF(mol)
    mf.kernel()

    # Reference CCSD:
    cc = pyscf.cc.CCSD(mf)
    cc.kernel()

    # Reference FCI
    fci = pyscf.fci.FCI(mf)
    fci.kernel()

    # Single-shot
    dmet_oneshot = vayesta.dmet.DMET(mf, solver="FCI", max_elec_err=1e-6, maxiter=1)
    with dmet_oneshot.iao_fragmentation() as f:
        f.add_atomic_fragment([0, 1])
        f.add_atomic_fragment([2, 3])
        f.add_atomic_fragment([4, 5])
    dmet_oneshot.kernel()
    # Full DMET
    dmet_diis = vayesta.dmet.DMET(mf, solver="FCI", max_elec_err=1e-6, charge_consistent=True, diis=True)
    with dmet_diis.iao_fragmentation() as f:
        f.add_atomic_fragment([0, 1])
        f.add_atomic_fragment([2, 3])
        f.add_atomic_fragment([4, 5])
    dmet_diis.kernel()

    print("E%-14s %+16.8f Ha" % ("(HF)=", mf.e_tot))
    print("E%-14s %+16.8f Ha" % ("(CCSD)=", cc.e_tot))
    print("E%-14s %+16.8f Ha" % ("(FCI)=", fci.e_tot))
    print("E%-14s %+16.8f Ha" % ("(Oneshot DMET-FCI)=", dmet_oneshot.e_tot))
    print("E%-14s %+16.8f Ha" % ("(DMET-FCI)=", dmet_diis.e_tot))

    with open("energies.txt", "a") as f:
        f.write(
            "%.2f  % 16.8f  % 16.8f  % 16.8f  %16.8f  %16.8f\n"
            % (d, mf.e_tot, cc.e_tot, fci.e_tot, dmet_oneshot.e_tot, dmet_diis.e_tot)
        )
