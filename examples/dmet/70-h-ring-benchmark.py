import numpy as np

import pyscf.cc
import pyscf.fci
import pyscf.tools
import pyscf.tools.ring

import vayesta.dmet

natom = 6

for d in np.arange(0.5, 3.0001, 0.25):

    ring = pyscf.tools.ring.make(natom, d)
    atom = [('H %f %f %f' % xyz) for xyz in ring]

    mol = pyscf.gto.Mole()
    mol.atom = atom
    mol.basis = '6-31G'
    mol.verbose = 10
    mol.output = 'pyscf_out.txt'
    mol.build()

    # Hartree-Fock
    mf = pyscf.scf.RHF(mol)
    mf.kernel()

    # Reference full system CCSD:
    mycc = pyscf.cc.CCSD(mf)
    mycc.kernel()

    myfci = pyscf.fci.FCI(mf)
    myfci.kernel()

    # Single-shot
    dmet_oneshot = vayesta.dmet.DMET(mf, solver='FCI', fragment_type='IAO', max_elec_err=1e-6, maxiter=1)
    dmet_oneshot.make_atom_fragment([0, 1])
    dmet_oneshot.make_atom_fragment([2, 3])
    dmet_oneshot.make_atom_fragment([4, 5])
    dmet_oneshot.kernel()
    # Full DMET
    dmet_diis = vayesta.dmet.DMET(mf, solver='FCI', fragment_type='IAO', charge_consistent=True, diis=True,
                                  max_elec_err = 1e-6)
    dmet_diis.make_atom_fragment([0,1]); dmet_diis.make_atom_fragment([2,3]); dmet_diis.make_atom_fragment([4,5])
    dmet_diis.kernel()

    print("E%-14s %+16.8f Ha" % ('(HF)=', mf.e_tot))
    print("E%-14s %+16.8f Ha" % ('(CCSD)=', mycc.e_tot))
    print("E%-14s %+16.8f Ha" % ('(FCI)=', myfci.e_tot))
    print("E%-14s %+16.8f Ha" % ('(Oneshot DMET-FCI)=', dmet_oneshot.e_tot))
    print("E%-14s %+16.8f Ha" % ('(DMET-FCI)=', dmet_diis.e_tot))

    with open("energies.txt", "a") as f:
        f.write("%.2f  % 16.8f  % 16.8f  % 16.8f  %16.8f  %16.8f\n" % (d, mf.e_tot, mycc.e_tot, myfci.e_tot, dmet_oneshot.e_tot,
                                                               dmet_diis.e_tot))
