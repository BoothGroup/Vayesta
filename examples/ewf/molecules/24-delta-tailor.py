import numpy as np
import pyscf
import pyscf.gto
import pyscf.scf
import pyscf.cc
import pyscf.fci
import vayesta
import vayesta.ewf


dm1 = None
for d in np.arange(1.0, 1.5, 0.2):

    mol = pyscf.gto.Mole()
    mol.atom = 'N 0 0 0 ; N 0 0 %f' % d
    cassize = (6,6)
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

    fci = pyscf.fci.FCI(mf)
    fci.kernel()

    def make_tcc(correction_type):
        tcc = vayesta.ewf.EWF(mf, solver='CCSD', bath_options=dict(bathtype='full'), solver_options=dict(solve_lambda=True))
        with tcc.cas_fragmentation() as f:
            cas = f.add_cas_fragment(*cassize, solver='FCI', store_wf_type='CCSD', bath_options=dict(bathtype='dmet'), auxiliary=True)
        with tcc.sao_fragmentation() as f:
            ccsd = f.add_atomic_fragment([0, 1])
        ccsd.add_external_corrections([cas], correction_type=correction_type, projectors=0)
        tcc.kernel()
        return tcc

    # Conventional Tailored CCSD
    tcc = make_tcc('tailor')
    # "delta" TCCSD
    dtcc = make_tcc('delta-tailor')

    def energy(obj):
        if obj.converged:
            return obj.e_tot
        return np.nan

    energies = [mf.e_tot, energy(cc), energy(fci), energy(tcc), energy(dtcc)]
    with open('energies.txt', 'a') as f:
        fmt = '%.2f' + (len(energies)*'  %+16.8f') + '\n'
        f.write(fmt % (d, *energies))
