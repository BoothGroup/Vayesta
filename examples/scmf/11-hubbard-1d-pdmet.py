import numpy as np

import vayesta
import vayesta.ewf
import vayesta.lattmod

nimp = 2
nsite = 10
nelectron = nsite

for hubbard_u in range(0, 13):
    print("U = %f" % hubbard_u)

    mol = vayesta.lattmod.Hubbard1D(nsite, nelectron=nelectron, hubbard_u=hubbard_u, output='pyscf.out')
    rhf = vayesta.lattmod.LatticeRHF(mol)
    rhf.kernel()

    # DMET
    vayesta.new_log("rdmet-U-%.0f.log" % hubbard_u)
    rdmet = vayesta.ewf.EWF(rhf, solver='FCI', bath_type=None, make_rdm1=True, make_rdm2=True)
    rdmet.site_fragmentation()
    imp = rdmet.add_atomic_fragment(list(range(nimp)))
    imp.add_tsymmetric_fragments(tvecs=[nsite//nimp, 1, 1])
    rdmet.pdmet_scmf()
    rdmet.kernel()
    e_rdmet = rdmet.get_dmet_energy()

    fmt = '%4.1f' + 3*'  %+16.8f' + '\n'
    energies = np.asarray([rhf.e_tot, rdmet.e_tot, e_rdmet]) / nelectron
    with open("energies-imp-%d.txt", 'a') as f:
        f.write(fmt % (hubbard_u, *energies))
