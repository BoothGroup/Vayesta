import numpy as np

import pyscf.cc
import pyscf.fci
import pyscf.tools
import pyscf.tools.ring

import vayesta.dmet
import vayesta.edmet
import vayesta.rpa
import logging

natom = 10
frags = [[x,x+1] for x in range(0, natom, 2)]
basis = "STO-6G"
efile = "energies_h{:d}_{:s}.txt".format(natom, basis)



with open(efile, "w") as f:
    f.write("%s  % 16s  % 16s  % 16s  %16s  %16s  %16s  %16s\n" % (
    "d", "HF", "dRPA", "CCSD", "FCI", "DMET Oneshot", "DMET", "EDMET oneshot"))


for d in np.arange(0.5, 3.0001, 0.25):

    ring = pyscf.tools.ring.make(natom, d)
    atom = [('H %f %f %f' % xyz) for xyz in ring]

    mol = pyscf.gto.Mole()
    mol.atom = atom
    mol.basis = basis
    mol.verbose = 10
    mol.output = 'pyscf_out.txt'
    mol.build()

    # Hartree-Fock
    mf = pyscf.scf.RHF(mol)
    mf.kernel()

    rpa = vayesta.rpa.dRPA(mf, logging.Logger("mylog"))
    rpa.kernel()

    # Reference full system CCSD:
    mycc = pyscf.cc.CCSD(mf)
    mycc.kernel()

    efci = np.nan
    myfci = pyscf.fci.FCI(mf)
    myfci.kernel()
    efci = myfci.e_tot

    # Single-shot
    dmet_oneshot = vayesta.dmet.DMET(mf, solver='FCI', fragment_type='IAO', max_elec_err=1e-6, maxiter=1, bath_type=None)
    # Full DMET
    dmet_diis = vayesta.dmet.DMET(mf, solver='FCI', fragment_type='IAO', charge_consistent=True, diis=True,
                                  max_elec_err=1e-6)
    edmet_oneshot = vayesta.edmet.EDMET(mf, solver='EBFCI', fragment_type='IAO', max_elec_err=1e-6, maxiter=1,
                                       bos_occ_cutoff=2)

    for f in frags:
        dmet_oneshot.make_atom_fragment(f)
        dmet_diis.make_atom_fragment(f)
        edmet_oneshot.make_atom_fragment(f)

    dmet_oneshot.kernel()
    dmet_diis.kernel()
    dmet_energy = np.nan
    if dmet_diis.converged: dmet_energy = dmet_diis.e_tot
    edmet_oneshot.kernel()

    print("E%-14s %+16.8f Ha" % ('(HF)=', mf.e_tot))
    print("E%-14s %+16.8f Ha" % ('(CCSD)=', mycc.e_tot))
    print("E%-14s %+16.8f Ha" % ('(FCI)=', efci))
    print("E%-14s %+16.8f Ha" % ('(Oneshot DMET-FCI)=', dmet_oneshot.e_tot))
    print("E%-14s %+16.8f Ha" % ('(DMET-FCI)=', dmet_energy))

    with open(efile, "a") as f:
        f.write("%.2f  % 16.8f  % 16.8f  % 16.8f  %16.8f  %16.8f  %16.8f  %16.8f\n" % (d, mf.e_tot, rpa.ecorr + mf.e_tot,
                                                    mycc.e_tot, efci, dmet_oneshot.e_tot, dmet_energy, edmet_oneshot.e_tot))
