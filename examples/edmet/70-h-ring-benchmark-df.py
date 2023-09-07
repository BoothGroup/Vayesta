import numpy as np
import pyscf.cc
import pyscf.fci
import pyscf.tools
import pyscf.tools.ring
import vayesta.dmet
import vayesta.edmet


natom = 6
filename = "energies_scEDMET_h{:d}_compare_df.txt".format(natom)

with open(filename, "a") as f:
    f.write(
        ("%6s" + "  %16s  " * 8)
        % ("d", "HF", "CCSD", "FCI", "DMET (Oneshot)", "DMET", "EDMET (Oneshot)", "EDMET (old)", "EDMET (new)")
    )

import numpy as np
import pyscf.cc
import pyscf.fci
import pyscf.tools
import pyscf.tools.ring
import vayesta.dmet
import vayesta.edmet


natom = 6
filename = "energies_h{:d}_compare_df.txt".format(natom)

with open(filename, "a") as f:
    f.write(("%6s" + "  %16s  " * 6 + "\n") % ("d", "HF", "CCSD", "FCI", "DMET (Oneshot)", "DMET", "EDMET (Oneshot)"))

for d in np.arange(0.5, 3.0001, 0.25):
    ring = pyscf.tools.ring.make(natom, d)
    atom = [("H %f %f %f" % xyz) for xyz in ring]

    mol = pyscf.gto.Mole()
    mol.atom = atom
    mol.basis = "STO-6G"
    # mol.verbose = 10
    # mol.output = 'pyscf_out.txt'
    mol.build()

    # Hartree-Fock
    # Replace with
    # mf = pyscf.scf.RHF(mol)
    # to run without density fitting.
    mf = pyscf.scf.RHF(mol).density_fit()
    mf.kernel()

    # Reference full system CCSD:
    mycc = pyscf.cc.CCSD(mf)
    mycc.kernel()

    myfci = pyscf.fci.FCI(mf)
    myfci.kernel()

    # Single-shot
    dmet_oneshot = vayesta.dmet.DMET(mf, solver="FCI", max_elec_err=1e-4, maxiter=1)
    with dmet_oneshot.iao_fragmentation() as f:
        for i in range(0, natom, 2):
            f.add_atomic_fragment([i, i + 1])
    dmet_oneshot.kernel()
    # Full DMET
    dmet_diis = vayesta.dmet.DMET(mf, solver="FCI", charge_consistent=True, diis=True, max_elec_err=1e-4)
    with dmet_diis.iao_fragmentation() as f:
        for i in range(0, natom, 2):
            f.add_atomic_fragment([i, i + 1])
    dmet_diis.kernel()
    # Single-shot EDMET
    edmet_oneshot = vayesta.edmet.EDMET(
        mf, solver="FCI", max_elec_err=1e-4, maxiter=1, solver_options=dict(max_boson_occ=2), oneshot=True
    )
    with edmet_oneshot.iao_fragmentation() as f:
        for i in range(0, natom, 2):
            f.add_atomic_fragment([i, i + 1])
    edmet_oneshot.kernel()

    e_cc = mycc.e_tot if mycc.converged else np.NaN
    e_dmet = dmet_diis.e_tot if dmet_diis.converged else np.NaN
    print("E%-14s %+16.8f Ha" % ("(HF)=", mf.e_tot))
    print("E%-14s %+16.8f Ha" % ("(CCSD)=", e_cc))
    print("E%-14s %+16.8f Ha" % ("(FCI)=", myfci.e_tot))
    print("E%-14s %+16.8f Ha" % ("(DMET-FCI)=", dmet_oneshot.e_tot))
    print("E%-14s %+16.8f Ha" % ("(EDMET-FCI-Oneshot)=", edmet_oneshot.e_tot))

    with open(filename, "a") as f:
        f.write(
            "%.2f  % 16.8f  % 16.8f  % 16.8f  %16.8f  %16.8f  %16.8f\n"
            % (d, mf.e_tot, e_cc, myfci.e_tot, dmet_oneshot.e_tot, e_dmet, edmet_oneshot.e_tot)
        )
