# tailoring broken

import numpy as np

import pyscf
import pyscf.gto
import pyscf.scf
import pyscf.cc
import pyscf.tools
import pyscf.tools.ring

import vayesta
import vayesta.ewf

natom = 6

for d in np.arange(0.5, 3.0001, 0.25):

    ring = pyscf.tools.ring.make(natom, d)
    atom = [('H %f %f %f' % xyz) for xyz in ring]

    mol = pyscf.gto.Mole()
    mol.atom = atom
    mol.basis = 'aug-cc-pvdz'
    mol.verbose = 10
    mol.output = 'pyscf_out.txt'
    mol.build()

    # Hartree-Fock
    mf = pyscf.scf.RHF(mol)
    mf.kernel()

    # Reference full system CCSD:
    cc = pyscf.cc.CCSD(mf)
    cc.kernel()

    # One-shot EWF-CCSD
    ecc = vayesta.ewf.EWF(mf, bno_threshold=1e-4)
    ecc.add_all_atomic_fragments()
    ecc.kernel()

    # Self-consistent EWF-CCSD
    scecc = vayesta.ewf.EWF(mf, bno_threshold=1e-4, sc_mode=1)
    scecc.add_all_atomic_fragments()
    scecc.tailor_all_fragments()
    scecc.kernel()

    print("E%-14s %+16.8f Ha" % ('(HF)=', mf.e_tot))
    print("E%-14s %+16.8f Ha" % ('(CCSD)=', cc.e_tot))
    print("E%-14s %+16.8f Ha" % ('(EWF-CCSD)=', ecc.e_tot))
    print("E%-14s %+16.8f Ha" % ('(SC-EWF-CCSD)=', scecc.e_tot))

    with open("energies.txt", "a") as f:
        f.write("%.2f  % 16.8f  % 16.8f  % 16.8f  %16.8f\n" % (d, mf.e_tot, cc.e_tot, ecc.e_tot, scecc.e_tot))
