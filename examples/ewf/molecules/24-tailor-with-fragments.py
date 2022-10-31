import numpy as np
import pyscf
import pyscf.gto
import pyscf.scf
import pyscf.cc
import pyscf.fci
import vayesta
import vayesta.ewf


dm1 = None
for d in np.arange(1.0, 3.1, 0.25):

    mol = pyscf.gto.Mole()
    mol.atom = 'N 0 0 0 ; N 0 0 %f' % d
    mol.basis = 'STO-6G'
    mol.build()

    # Hartree-Fock
    mf = pyscf.scf.RHF(mol)
    mf.kernel(dm1)
    while True:
        mo1 = mf.stability()[0]
        stable = mo1 is mf.mo_coeff
        dm1 = mf.make_rdm1(mo_coeff=mo1)
        if stable:
            break
        mf.kernel(dm1)

    # Reference full system CCSD:
    cc = pyscf.cc.CCSD(mf)
    cc.kernel()

    # Reference full system FCI:
    fci = pyscf.fci.FCI(mf)
    fci.kernel()

    # Tailor single CCSD with two atomic FCI fragments (projected in first index onto fragment space)
    tcc = vayesta.ewf.EWF(mf, solver='CCSD', bath_options=dict(bathtype='full'), solve_lambda=True)
    # store_wf_type determines the type of wave function which will be stored.
    # For the tailoring below, only the T1 and T2 amplitudes of the FCI fragments are needed,
    # and the FCI wave function can thus be converted + truncated to a CCSD wave function:
    with tcc.fragmentation() as f:
        fci_x1 = f.add_atomic_fragment(0, solver='FCI', bath_options=dict(bathtype='dmet'), store_wf_type='CCSD')
        fci_x2 = f.add_atomic_fragment(1, solver='FCI', bath_options=dict(bathtype='dmet'), store_wf_type='CCSD')
        ccsd = f.add_atomic_fragment([0, 1], active=False)
    # Solve FCI
    tcc.kernel()
    fci_x1.active = fci_x2.active = False
    ccsd.add_external_corrections([fci_x1, fci_x2], correction_type='tailor')
    ccsd.active = True
    # Solve CCSD
    tcc.kernel()

    energies = [mf.e_tot, cc.e_tot if cc.converged else np.nan, fci.e_tot, tcc.e_tot, tcc.get_dm_energy()]
    with open('energies.txt', 'a') as f:
        fmt = '%.2f' + (len(energies)*'  %+16.8f') + '\n'
        f.write(fmt % (d, *energies))
