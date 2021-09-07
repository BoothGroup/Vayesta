import numpy as np

import pyscf.cc
import pyscf.fci
import pyscf.tools
import pyscf.tools.ring

import vayesta.dmet
import vayesta.edmet
import vayesta.rpa
import logging
from pyscf import ao2mo

natom = 10
frags = [[x,x+1] for x in range(0, natom, 2)]
basis = "STO-6G"
efile = "energies_h{:d}_{:s}.txt".format(natom, basis)
doccfile = "docc_h{:d}_{:s}.txt".format(natom, basis)
nnfile = "nn_h{:d}_{:s}.txt".format(natom, basis)


for filename in [efile, doccfile, nnfile]:
    with open(filename, "w") as f:
        f.write("%s  % 16s  % 16s  % 16s  %16s  %16s  %16s  %16s\n" % (
        "d", "HF", "dRPA", "CCSD", "FCI", "DMET Oneshot", "DMET", "EDMET oneshot"))

def get_correlators(qemb):
    f = qemb.fragments[0]
    c = np.linalg.multi_dot([f.c_active.T, mf.get_ovlp(), f.c_frag])
    return rdm_to_correlators(f.results.dm2, c)

def rdm_to_correlators(dm2, c):
    correlators = np.einsum("ijkl,xi,xj,yk,yl->xy", dm2, *(4*[c.T]))/
    docc = correlators[0, 0]
    nn = correlators[0, 1]
    return docc, nn

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
    os_docc, os_nn = get_correlators(dmet_oneshot)
    dmet_diis.kernel()

    dmet_energy, dmet_docc, dmet_nn = np.nan, np.nan, np.nan
    if dmet_diis.converged:
        dmet_energy = dmet_diis.e_tot
        dmet_docc, dmet_nn = get_correlators(dmet_diis)
    edmet_oneshot.kernel()
    edmet_docc, edmet_nn = get_correlators(edmet_oneshot)

    c = np.linalg.multi_dot([mf.mo_coeff, mf.get_ovlp(), dmet_oneshot.fragments[0].c_frag])

    mf_dm1 = mf.make_rdm1()
    mf_docc, mf_nn = rdm_to_correlators(
        np.einsum("pq,rs->pqrs", mf_dm1, mf_dm1) - np.einsum("pq,rs->psrq", mf_dm1, mf_dm1), dmet_oneshot.fragments[0].c_frag)
    ccsd_docc, ccsd_nn = rdm_to_correlators(mycc.make_rdm2(), c)
    fci_docc, fci_nn = rdm_to_correlators(myfci.make_rdm2(myfci.ci, myfci.norb, myfci.nelec), c)


    print("E%-14s %+16.8f Ha" % ('(HF)=', mf.e_tot))
    print("E%-14s %+16.8f Ha" % ('(CCSD)=', mycc.e_tot))
    print("E%-14s %+16.8f Ha" % ('(FCI)=', efci))
    print("E%-14s %+16.8f Ha" % ('(Oneshot DMET-FCI)=', dmet_oneshot.e_tot))
    print("E%-14s %+16.8f Ha" % ('(DMET-FCI)=', dmet_energy))

    with open(efile, "a") as f:
        f.write("%.2f  % 16.8f  % 16.8f  % 16.8f  %16.8f  %16.8f  %16.8f  %16.8f\n" % (d, mf.e_tot, rpa.ecorr + mf.e_tot,
                                                    mycc.e_tot, efci, dmet_oneshot.e_tot, dmet_energy, edmet_oneshot.e_tot))

    with open(doccfile, "a") as f:
        f.write("%.2f  % 16.8f  % 16.8f  % 16.8f  %16.8f  %16.8f  %16.8f  %16.8f\n" % (d, mf_docc, np.nan,
                                                    ccsd_docc, fci_docc, os_docc, dmet_docc, edmet_docc))

    with open(nnfile, "a") as f:
        f.write("%.2f  % 16.8f  % 16.8f  % 16.8f  %16.8f  %16.8f  %16.8f  %16.8f\n" % (d, mf_nn, np.nan,
                                                    ccsd_nn, fci_nn, os_nn, dmet_nn, edmet_nn))


