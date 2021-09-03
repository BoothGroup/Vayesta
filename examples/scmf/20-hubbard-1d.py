import numpy as np

import pyscf
import pyscf.gto
import pyscf.scf
import pyscf.fci
import pyscf.tools.ring

import vayesta
import vayesta.ewf
import vayesta.lattmod

nsite = 10
nelectron = nsite
nfrag = 2

for hubbard_u in np.arange(0, 12.1, 1):
    mol = vayesta.lattmod.Hubbard1D(nsite, nelectron=nelectron, hubbard_u=hubbard_u, output='pyscf.out')
    mf = vayesta.lattmod.LatticeMF(mol)
    mf.kernel()

    e_hf = mf.e_tot / nsite

    fci = vayesta.ewf.EWF(mf, solver='FCI', fragment_type='Site', bath_type=None)
    fci.make_atom_fragment(list(range(nsite)))
    fci.kernel()
    e_fci = fci.e_tot / nsite

    ewf = vayesta.ewf.EWF(mf, solver='FCI', fragment_type='Site', bath_type=None, make_rdm1=True, make_rdm2=True)
    for n in range(0, nsite, nfrag):
        ewf.make_atom_fragment(list(range(n, n+nfrag)), nelectron_target=nfrag)

    # --- One-shot calculation:
    ewf.kernel()
    e_pwf_1 = ewf.get_total_energy() / nsite
    e_pdm_1 = ewf.get_dmet_energy() / nsite

    # --- Self-consistent MF calculations:
    # These set the attribute ewf.with_scmf
    # Check vayesta/core/qemb/scmf.py for arguments
    ewf = ewf.pdmet_scmf()
    #ewf = ewf.brueckner_scmf()
    ewf.kernel()
    if ewf.with_scmf.converged:
        e_pwf_sc = ewf.get_total_energy() / nsite
        e_pdm_sc = ewf.get_dmet_energy() / nsite
    else:
        print("SCMF not converged!")
        e_pwf_sc = e_pdm_sc = np.nan

    print("E(HF)=           %+16.8f Ha" % (e_hf))
    print("E(FCI)=          %+16.8f Ha" % (e_fci))
    print("E(p-WF)[1-shot]= %+16.8f Ha" % (e_pwf_1))
    print("E(p-WF)[sc]=     %+16.8f Ha" % (e_pwf_sc))
    print("E(p-DM)[1-shot]= %+16.8f Ha" % (e_pdm_1))
    print("E(p-DM)[sc]=     %+16.8f Ha" % (e_pdm_sc))

    with open("20-energies.txt", 'a') as f:
        energies = (e_hf, e_fci, e_pwf_1, e_pwf_sc, e_pdm_1, e_pdm_sc)
        fmt = '%4.1f' + len(energies)*'  %+16.8f' + '\n'
        f.write(fmt % (hubbard_u, *energies))
