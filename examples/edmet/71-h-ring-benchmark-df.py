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
    f.write(("%6s" + "  %16s  " * 8) % (
    "d", "HF", "CCSD", "FCI", "DMET (Oneshot)", "DMET", "EDMET (Oneshot)", "EDMET (old)", "EDMET (new)"))

for d in np.arange(0.5, 3.0001, 0.25):

    ring = pyscf.tools.ring.make(natom, d)
    atom = [('H %f %f %f' % xyz) for xyz in ring]

    mol = pyscf.gto.Mole()
    mol.atom = atom
    mol.basis = 'STO-6G'
    # mol.verbose = 10
    # mol.output = 'pyscf_out.txt'
    mol.build()

    # Hartree-Fock
    mf = pyscf.scf.RHF(mol)
    dfmf = mf.density_fit()
    dfmf.kernel()

    # Reference full system CCSD:
    mycc = pyscf.cc.CCSD(dfmf)
    mycc.kernel()

    myfci = pyscf.fci.FCI(dfmf)
    myfci.kernel()

    # Single-shot
    dmet_oneshot = vayesta.dmet.DMET(dfmf, solver='FCI', max_elec_err=1e-4, maxiter=1)
    with dmet_oneshot.iao_fragmentation() as f:
        for i in range(0, natom, 2):
            f.add_atomic_fragment([i, i + 1])
    dmet_oneshot.kernel()
    # Full DMET
    dmet_diis = vayesta.dmet.DMET(dfmf, solver='FCI', charge_consistent=True, diis=True,
                                  max_elec_err=1e-4)
    with dmet_diis.iao_fragmentation() as f:
        for i in range(0, natom, 2):
            f.add_atomic_fragment([i, i + 1])
    dmet_diis.kernel()
    # Single-shot EDMET
    edmet_oneshot = vayesta.edmet.EDMET(dfmf, solver='EBFCI', max_elec_err=1e-4, maxiter=1, max_boson_occ=2)
    with edmet_oneshot.iao_fragmentation() as f:
        for i in range(0, natom, 2):
            f.add_atomic_fragment([i, i + 1])
    edmet_oneshot.kernel()
    # Full DMET
    edmet_orig = vayesta.edmet.EDMET(dfmf, solver='EBFCI', charge_consistent=True, max_elec_err=1e-4, maxiter=40,
                                     max_boson_occ=2, old_sc_condition=True)
    with edmet_orig.iao_fragmentation() as f:
        for i in range(0, natom, 2):
            f.add_atomic_fragment([i, i + 1])
    edmet_orig.kernel()

    edmet_new = vayesta.edmet.EDMET(dfmf, solver='EBFCI', charge_consistent=True, max_elec_err=1e-4, maxiter=40,
                                    max_boson_occ=2)
    with edmet_new.iao_fragmentation() as f:
        for i in range(0, natom, 2):
            f.add_atomic_fragment([i, i + 1])
    edmet_new.kernel()

    e_sc_edmet1 = edmet_orig.e_tot if edmet_orig.converged else np.NaN
    e_sc_edmet2 = edmet_new.e_tot if edmet_new.converged else np.NaN
    e_cc = mycc.e_tot if mycc.converged else np.NaN
    e_dmet = dmet_diis.e_tot if dmet_diis.converged else np.NaN
    print("E%-14s %+16.8f Ha" % ('(HF)=', dfmf.e_tot))
    print("E%-14s %+16.8f Ha" % ('(CCSD)=', e_cc))
    print("E%-14s %+16.8f Ha" % ('(FCI)=', myfci.e_tot))
    print("E%-14s %+16.8f Ha" % ('(DMET-FCI)=', dmet_oneshot.e_tot))
    print("E%-14s %+16.8f Ha" % ('(EDMET-FCI-Oneshot)=', edmet_oneshot.e_tot))
    print("E%-14s %+16.8f Ha" % ('(EDMET1-FCI)=', e_sc_edmet1))
    print("E%-14s %+16.8f Ha" % ('(EDMET2-FCI)=', e_sc_edmet2))

    with open(filename, "a") as f:
        f.write("%.2f  % 16.8f  % 16.8f  % 16.8f  %16.8f  %16.8f  %16.8f  %16.8f  %16.8f\n" % (d, dfmf.e_tot, e_cc,
                                                                                               myfci.e_tot,
                                                                                               dmet_oneshot.e_tot,
                                                                                               e_dmet,
                                                                                               edmet_oneshot.e_tot,
                                                                                               e_sc_edmet1,
                                                                                               e_sc_edmet2))
