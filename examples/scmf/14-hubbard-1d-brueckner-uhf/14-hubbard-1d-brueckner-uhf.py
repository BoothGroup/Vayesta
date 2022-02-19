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
    vayesta.new_log('U%.0f.log' % hubbard_u)

    mol = vayesta.lattmod.Hubbard1D(nsite, nelectron=nelectron, hubbard_u=hubbard_u, output='pyscf.out')
    mf = vayesta.lattmod.LatticeUHF(mol)
    mf.conv_tol = 1e-12
    # Break spin symmetry
    if dm is not None and np.allclose(dm[0], dm[1], rtol=0, atol=0.1):
        a = list(range(0, nsite, 2))
        b = list(range(1, nsite, 2))
        dm[0][a,a] += 0.1
        dm[0][b,b] -= 0.1
        dm[1][b,b] += 0.1
        dm[1][a,a] -= 0.1
    mf.kernel(dm0=dm)
    dm = mf.make_rdm1()

    dmet = vayesta.ewf.EWF(mf, solver='FCI', bath_type=None, make_rdm1=True, make_rdm2=True)
    dmet.site_fragmentation()
    if nimp % 2 == 0:
        imp = dmet.add_atomic_fragment(list(range(nimp)))
        imp.add_tsymmetric_fragments(tvecs=[nsite//nimp, 1, 1])
    else:
        imp1 = dmet.add_atomic_fragment(list(range(nimp)))
        imp2 = dmet.add_atomic_fragment(list(range(nimp, 2*nimp)))
        imp1.add_tsymmetric_fragments(tvecs=[nsite//(2*nimp), 1, 1])
        imp2.add_tsymmetric_fragments(tvecs=[nsite//(2*nimp), 1, 1])

    # One-shot results
    vayesta.new_log('os-U%.0f.log' % hubbard_u)
    dmet.kernel()
    e_wf = dmet.e_tot
    e_dm = dmet.get_dmet_energy()

    # Self-consistent results
    vayesta.new_log('sc-U%.0f.log' % hubbard_u)
    dmet.brueckner_scmf()
    dmet.kernel()
    assert dmet.with_scmf.converged
    e_wf_sc = dmet.e_tot
    e_dm_sc = dmet.get_dmet_energy()

    energies = np.asarray([mf.e_tot, e_wf, e_wf_sc, e_dm, e_dm_sc]) / nelectron
    fmt = '%4.1f' + len(energies)*'  %+16.8f' + '\n'
    with open("energies.txt", 'a') as f:
        f.write(fmt % (hubbard_u, *energies))
