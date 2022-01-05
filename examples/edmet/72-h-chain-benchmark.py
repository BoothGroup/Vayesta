import numpy as np

import pyscf.cc
import pyscf.fci
import pyscf.tools

import vayesta.dmet
import vayesta.edmet

natom = 10

for d in np.arange(1.0, 3.7, 0.2):

    pos = [(d * x, 0, 0) for x in range(natom)]
    atom = [('H %f %f %f' % xyz) for xyz in pos]

    mol = pyscf.gto.Mole()
    mol.atom = atom
    mol.basis = 'cc-pvdz'
    mol.unit = "Bohr"
    #mol.verbose = 10
    #mol.output = 'pyscf_out.txt'
    mol.build()

    # Hartree-Fock
    mf = pyscf.scf.RHF(mol)
    mf.kernel()

    # Reference full system CCSD:
    mycc = pyscf.cc.CCSD(mf)
    mycc.kernel()

    #myfci = pyscf.fci.FCI(mf)
    #myfci.kernel()

    # Single-shot
    dmet_oneshot = vayesta.dmet.DMET(mf, solver='FCI', max_elec_err=1e-6, maxiter=1)
    dmet_oneshot.iao_fragmentation()
    dmet_oneshot.add_all_atomic_fragments()
    dmet_oneshot.kernel()
    # Full DMET
    #dmet_diis = vayesta.dmet.DMET(mf, solver='FCI', charge_consistent=True, diis=True,
    #                              max_elec_err = 1e-6)
    #dmet_diis.iao_fragmentation()
    #dmet_diis.add_all_atomic_fragments()
    #dmet_diis.kernel()
    # Single-shot EDMET
    edmet_oneshot = vayesta.edmet.EDMET(mf, solver='EBFCI', max_elec_err=1e-6, maxiter=1, bos_occ_cutoff=4)
    edmet_oneshot.iao_fragmentation()
    edmet_oneshot.add_all_atomic_fragments()
    edmet_oneshot.kernel()
    # Full DMET
    edmet_diis = vayesta.edmet.EDMET(mf, solver='EBFCI', charge_consistent=True, max_elec_err=1e-6, maxiter=40, bos_occ_cutoff=4)
    edmet_diis.iao_fragmentation()
    edmet_diis.add_all_atomic_fragments()
    edmet_diis.kernel()

    e_sc_edmet = edmet_diis.e_tot if edmet_diis.converged else np.NaN
    e_cc = mycc.e_tot if mycc.converged else np.NaN
    print("E%-14s %+16.8f Ha" % ('(HF)=', mf.e_tot))
    print("E%-14s %+16.8f Ha" % ('(CCSD)=', e_cc))
    print("E%-14s %+16.8f Ha" % ('(DMET-FCI)=', dmet_oneshot.e_tot))
    #print("E%-14s %+16.8f Ha" % ('(DMET-FCI)=', dmet_diis.e_tot))
    print("E%-14s %+16.8f Ha" % ('(EDMET-FCI-Oneshot)=', edmet_oneshot.e_tot))
    print("E%-14s %+16.8f Ha" % ('(EDMET-FCI)=', e_sc_edmet))

    with open("energies_h10_ccpvdz.txt", "a") as f:
        f.write("%.2f  % 16.8f  % 16.8f  %16.8f  %16.8f  %16.8f\n" % (d, mf.e_tot, e_cc, dmet_oneshot.e_tot,
                                                               edmet_oneshot.e_tot, e_sc_edmet))
