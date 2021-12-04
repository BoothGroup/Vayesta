import numpy as np

import pyscf
import pyscf.gto
import pyscf.scf
import pyscf.cc
import pyscf.ao2mo

import vayesta
import vayesta.ewf

spacing = 10.0
basis = '6-31G'
#basis = 'cc-pvdz'
bno = 1e-6
#bno = 1e-8
#stretch = 1.0
stretch = 1.5

for n in range(1, 10):
    mol = pyscf.gto.Mole()
    atom = n*["O  0.0000   0.0000  %f", "H  0.0000   %f  %f", "H  0.0000  %f  %f"]
    #atom = n*["O  %f   0.0000  %f", "H  %f   %f  %f", "H  %f  %f  %f"]
    for i in range(0, len(atom), 3):
        atom[i] = atom[i] % ((i//3)*spacing + stretch*0.1173)
        #atom[i] = atom[i] % ((i//3)*spacing,  stretch*0.1173)
    for i in range(1, len(atom), 3):
        atom[i] = atom[i] % (stretch*0.7572, (i//3)*spacing - stretch*0.4692)
        #atom[i] = atom[i] % ((i//3)*spacing, stretch*0.7572, -stretch*0.4692)
    for i in range(2, len(atom), 3):
        atom[i] = atom[i] % (-stretch*0.7572, (i//3)*spacing - stretch*0.4692)
        #atom[i] = atom[i] % ((i//3)*spacing, -stretch*0.7572,- stretch*0.4692)
    mol.atom = atom
    mol.basis = basis
    mol.verbose = 10
    mol.output = 'pyscf_out.txt'
    mol.build()

    # Hartree-Fock
    mf = pyscf.scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()

    # Exact CCSD
    ccsd = pyscf.cc.ccsd.CCSD(mf)
    ccsd.conv_tol = 1e-10
    eris = ccsd.ao2mo()
    ccsd.kernel(eris=eris)

    ecc = vayesta.ewf.EWF(mf, bno_threshold=bno, make_rdm1=True)
    # Alternative:
    ecc.make_all_atom_fragments()
    ecc.kernel()

    print("E%-16s %+16.8f Ha" % ('(HF)=', mf.e_tot / n))
    print("E%-16s %+16.8f Ha" % ('(CCSD)=', ccsd.e_tot / n))
    print("E%-16s %+16.8f Ha" % ('(EWF-CCSD)=', ecc.e_tot / n))

    # New energy
    t1, t2 = ecc.get_t12()
    e_corr2 = ccsd.energy(t1=t1, t2=t2, eris=eris)
    print("E%-16s %+16.8f Ha" % ('(EWF-CCSD2)=', (mf.e_tot + e_corr2) / n))

    # RDM energy
    #dm1 = ecc.make_rdm1_ccsd()
    #dm2 = ecc.make_rdm2_ccsd()

    dm1 = ecc.make_rdm1_ccsd(t_as_lambda=True)
    dm2 = ecc.make_rdm2_ccsd(t_as_lambda=True)

    h1 = np.einsum('pi,pq,qj->ij', mf.mo_coeff.conj(), mf.get_hcore(), mf.mo_coeff)
    nmo = mf.mo_coeff.shape[1]
    eri = pyscf.ao2mo.kernel(mol, mf.mo_coeff, compact=False).reshape([nmo]*4)
    E = np.einsum('pq,qp', h1, dm1)
    # Note dm2 is transposed to simplify its contraction to integrals
    E+= np.einsum('pqrs,pqrs', eri, dm2) * .5
    E+= mol.energy_nuc()
    e_corr3 = (E-mf.e_tot)
    print("E%-16s %+16.8f Ha" % ('(EWF-CCSD-DM)=', (E / n)))

    #perc1 = 100*ecc.e_corr/ccsd.e_corr
    #perc2 = 100*e_corr2/ccsd.e_corr
    #perc3 = 100*e_corr3/ccsd.e_corr
    #dperc = abs(e_corr2 - ecc.e_corr) / abs(ccsd.e_corr - ecc.e_corr)
    with open('e-corr.txt', 'a') as f:
        f.write('%2d  %.10f  %.10f  %.10f\n' % (n,
            100*ecc.e_corr/ccsd.e_corr,
            100*e_corr2/ccsd.e_corr,
            100*e_corr3/ccsd.e_corr))
