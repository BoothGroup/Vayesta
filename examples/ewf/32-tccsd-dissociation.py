import numpy as np

import pyscf
import pyscf.gto
import pyscf.scf
import pyscf.cc
import pyscf.mcscf
import pyscf.mrpt

import vayesta
import vayesta.ewf

dmin = 1.2
dmax = 3.0
dstep = 0.1
dimer = ['F', 'F']
cas = (14, 8)
cas_sym = {
        'A1g' : 2,
        'A1u' : 2,
        'E1gx' : 1,
        'E1gy' : 1,
        'E1ux' : 1,
        'E1uy' : 1,
        }
basis = 'cc-pVTZ'


#for d in [3.0]:
for d in np.arange(dmin, dmax+1e-12, dstep):
    mol = pyscf.gto.Mole()
    mol.atom = ['%s 0.0 0.0 0.0' % dimer[0], '%s 0.0 0.0 %f' % (dimer[1], d)]
    mol.basis = basis
    mol.verbose = 10
    mol.output = 'pyscf_out.txt'
    mol.build()

    mol_sym = pyscf.gto.Mole()
    mol_sym.atom = ['%s 0.0 0.0 0.0' % dimer[0], '%s 0.0 0.0 %f' % (dimer[1], d)]
    mol_sym.basis = basis
    mol_sym.verbose = 10
    mol_sym.output = 'pyscf_out_sym.txt'
    mol_sym.symmetry = True
    mol_sym.build()


    # Hartree-Fock
    mf = pyscf.scf.RHF(mol)
    mf.kernel()
    # Stability
    mo = mf.stability()[0]
    if not np.allclose(mo, mf.mo_coeff):
        print("HF unstable at d=%f A" % d)
        dm1 = mf.make_rdm1(mo, mf.mo_occ)
        mf.kernel(dm1)
        mo = mf.stability()[0]
        assert np.allclose(mo, mf.mo_coeff)

    # Hartree-Fock
    mf_sym = pyscf.scf.RHF(mol_sym)
    mf_sym.kernel()
    # Stability
    mo = mf_sym.stability()[0]
    if not np.allclose(mo, mf_sym.mo_coeff):
        print("HF unstable at d=%f A" % d)
        dm1 = mf_sym.make_rdm1(mo, mf_sym.mo_occ)
        mf_sym.kernel(dm1)
        mo = mf_sym.stability()[0]
        assert np.allclose(mo, mf_sym.mo_coeff)

    if not np.isclose(mf.e_tot, mf_sym.e_tot):
        print("E(MF) != E(sym-MF)")

    #mf.analyze()
    #1/0

    # Reference CASCI
    casci = pyscf.mcscf.CASCI(mf_sym, cas[1], cas[0])
    if cas_sym is not None:
        mo = pyscf.mcscf.sort_mo_by_irrep(casci, mf_sym.mo_coeff, cas_sym)
    else:
        mo = None
    casci.kernel(mo)

    # Reference CASCI
    casscf = pyscf.mcscf.CASSCF(mf_sym, cas[1], cas[0])
    if cas_sym is not None:
        mo = pyscf.mcscf.sort_mo_by_irrep(casscf, mf_sym.mo_coeff, cas_sym)
    else:
        mo = None
    casscf.kernel(mo)

    nevpt2 = pyscf.mrpt.nevpt2.NEVPT(casscf)
    e_nevpt2 = nevpt2.kernel()

    # Reference full system CCSD:
    cc = pyscf.cc.CCSD(mf)
    try:
        cc.kernel()
    except:
        cc.e_tot = np.nan

    # Reference one-shot EWF-CCSD
    ecc = vayesta.ewf.EWF(mf, bno_threshold=-1)
    if dimer[0] == dimer[1]:
        ecc.add_atomic_fragment(1, sym_factor=2)
    else:
        ecc.add_all_atomic_fragments()
    try:
        ecc.kernel()
    except:
        ecc.e_tot = np.nan

    etcc = vayesta.ewf.EWF(mf, solver='TCCSD', bno_threshold=-1)
    if dimer[0] == dimer[1]:
        etcc.add_atomic_fragment(1, sym_factor=2)
    else:
        etcc.add_all_atomic_fragments()
    etcc.kernel()

    with open('energies.txt', 'a') as f:
        if d == dmin:
            f.write("# Energies for %s-%s dimer\n" % (dimer[0], dimer[1]))
        f.write('%3f  %16.8f  %16.8f  %16.8f  %16.8f  %16.8f  %16.8f  %16.8f\n' % (d, mf.e_tot, casci.e_tot, casscf.e_tot, casscf.e_tot + e_nevpt2, cc.e_tot, ecc.e_tot, etcc.e_tot))
