import numpy as np

import pyscf
import pyscf.pbc
import pyscf.pbc.gto
import pyscf.pbc.scf
import pyscf.pbc.cc

import vayesta
import vayesta.ewf

kmesh = [4, 4, 1]

#for d in np.arange(0.5, 3.0001, 0.25):
for d in np.arange(0.7, 3.0001, 0.25):

    cell = pyscf.pbc.gto.Cell()
    cell.atom = 'He 0.0 0.0 0.0'
    cell.a = d*np.eye(3)
    cell.a[2,2] = 20.0
    cell.basis = '6-31g'
    cell.verbose = 10
    cell.output = 'pyscf_out.txt'
    cell.dimension = 2
    cell.build()

    kpts = cell.make_kpts(kmesh)

    # Hartree-Fock
    kmf = pyscf.pbc.scf.KRHF(cell, kpts)
    kmf = kmf.density_fit()
    kmf.kernel()

    # Reference full system CCSD:
    kcc = pyscf.pbc.cc.KCCSD(kmf)
    kcc.kernel()

    # One-shot EWF-CCSD
    ecc = vayesta.ewf.EWF(kmf, bno_threshold=1e-4)
    ecc.make_all_atom_fragments()
    ecc.kernel()

    # Self-consistent EWF-CCSD
    scecc = vayesta.ewf.EWF(kmf, bno_threshold=1e-4, sc_mode=1)
    scecc.make_all_atom_fragments()
    scecc.tailor_all_fragments()
    scecc.kernel()

    print("E%-14s %+16.8f Ha" % ('(HF)=', kmf.e_tot))
    print("E%-14s %+16.8f Ha" % ('(CCSD)=', kcc.e_tot))
    print("E%-14s %+16.8f Ha" % ('(EWF-CCSD)=', ecc.e_tot))
    print("E%-14s %+16.8f Ha" % ('(SC-EWF-CCSD)=', scecc.e_tot))

    with open("energies.txt", "a") as f:
        f.write("%.2f  % 16.8f  % 16.8f  % 16.8f  %16.8f\n" % (d, kmf.e_tot, kcc.e_tot, ecc.e_tot, scecc.e_tot))
        #f.write("%.2f  % 16.8f  % 16.8f  % 16.8f  %16.8f\n" % (d, kmf.e_tot, ecc.e_tot, scecc.e_tot))
