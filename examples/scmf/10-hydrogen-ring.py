import numpy as np

import pyscf
import pyscf.gto
import pyscf.scf
import pyscf.fci
import pyscf.tools.ring

import vayesta
import vayesta.ewf

natom = 10
nfrag = 2
basis = 'STO-6G'

for a in np.arange(0.8, 3.001, 0.2):
    mol = pyscf.gto.Mole()
    atom = pyscf.tools.ring.make(natom, a)
    atom = ['H %f %f %f' % xyz for xyz in atom]
    mol.atom = atom
    mol.basis = basis
    mol.output = 'pyscf_out.txt'
    mol.build()

    # Hartree-Fock
    mf = pyscf.scf.RHF(mol)
    mf.conv_tol = 1e-12
    mf.kernel()
    stable = False
    while not stable:
        mo1 = mf.stability()[0]
        stable = (mo1 is mf.mo_coeff)
        if stable:
            print("HF stable!")
            break
        print("HF unstable...")
        dm1 = mf.make_rdm1(mo_coeff=mo1)
        mf.kernel(dm0=dm1)
    e_hf = mf.e_tot / natom

    fci = pyscf.fci.FCI(mf)
    fci.kernel()
    e_fci = fci.e_tot /  natom

    ewf = vayesta.ewf.EWF(mf, solver='FCI', fragment_type='Lowdin-AO', bath_type=None, make_rdm1=True, make_rdm2=True)
    for n in range(0, natom, nfrag):
        ewf.make_atom_fragment(list(range(n, n+nfrag)), nelectron_target=nfrag)

    # --- One-shot calculation:
    ewf.kernel()
    e_pwf_1 = ewf.get_total_energy() / natom
    e_pdm_1 = ewf.get_dmet_energy() / natom

    # --- Self-consistent MF calculations:
    # These set the attribute ewf.with_scmf
    # Check vayesta/core/qemb/scmf.py for arguments
    ewf = ewf.pdmet_scmf()
    #ewf = ewf.brueckner_scmf()
    ewf.kernel()
    if ewf.with_scmf.converged:
        e_pwf_sc = ewf.get_total_energy() / natom
        e_pdm_sc = ewf.get_dmet_energy() / natom
    else:
        print("SCMF not converged!")
        e_pwf_sc = e_pdm_sc = np.nan

    print("E(HF)=           %+16.8f Ha" % (e_hf))
    print("E(FCI)=          %+16.8f Ha" % (e_fci))
    print("E(p-WF)[1-shot]= %+16.8f Ha" % (e_pwf_1))
    print("E(p-WF)[sc]=     %+16.8f Ha" % (e_pwf_sc))
    print("E(p-DM)[1-shot]= %+16.8f Ha" % (e_pdm_1))
    print("E(p-DM)[sc]=     %+16.8f Ha" % (e_pdm_sc))

    with open("10-energies.txt", 'a') as f:
        energies = (e_hf, e_fci, e_pwf_1, e_pwf_sc, e_pdm_1, e_pdm_sc)
        fmt = '%.2f' + len(energies)*'  %+16.8f' + '\n'
        f.write(fmt % (a, *energies))
