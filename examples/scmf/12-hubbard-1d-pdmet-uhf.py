import numpy as np

import vayesta
import vayesta.ewf
import vayesta.lattmod

nimp = 2
nsite = 10
nelectron = nsite

dm = None
for hubbard_u in range(0, 13):
    print("U = %f" % hubbard_u)
    mol = vayesta.lattmod.Hubbard1D(nsite, nelectron=nelectron, hubbard_u=hubbard_u, output='pyscf.out')

    uhf = vayesta.lattmod.LatticeUHF(mol)
    uhf.conv_tol = 1e-12
    # Break spin symmetry
    if dm is not None:
        delta = 0.1*dm[1]
        delta[0::2,0::2] += 0.1
        delta[1::2,1::2] -= 0.1
        dm = (dm[0]+delta, dm[1]-delta)
    uhf.kernel(dm0=dm)
    dm = uhf.make_rdm1()

    vayesta.new_log("udmet-U-%.0f.log" % hubbard_u)
    udmet = vayesta.ewf.EWF(uhf, solver='FCI', bath_type=None, make_rdm1=True, make_rdm2=True)
    udmet.site_fragmentation()
    if nimp % 2 == 0:
        imp = udmet.add_atomic_fragment(list(range(nimp)))
        imp.add_tsymmetric_fragments(tvecs=[nsite//nimp, 1, 1])
    else:
        imp1 = udmet.add_atomic_fragment(list(range(nimp)))
        imp2 = udmet.add_atomic_fragment(list(range(nimp, 2*nimp)))
        imp1.add_tsymmetric_fragments(tvecs=[nsite//(2*nimp), 1, 1])
        imp2.add_tsymmetric_fragments(tvecs=[nsite//(2*nimp), 1, 1])
    udmet.pdmet_scmf()
    udmet.kernel()
    e_udmet = udmet.get_dmet_energy()

    fmt = '%4.1f' + 3*'  %+16.8f' + '\n'
    energies = np.asarray([uhf.e_tot, udmet.e_tot, e_udmet]) / nelectron
    with open("energies-imp-%d-uhf.txt", 'a') as f:
        f.write(fmt % (hubbard_u, *energies))
