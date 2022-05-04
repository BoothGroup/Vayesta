import numpy as np

import vayesta
import vayesta.ewf
import vayesta.lattmod

from scmf import SelfConsistentMF

nsites = (8,8)
nsite = nsites[0]*nsites[1]
nimp = 4
doping = 0
u_min = 12
u_max = 12
u_step = 1
do_fci = (nsite <= 10)
#tvecs = [nsites[0]//2, nsites[1]//2, 1]
tvecs = None

mo_pdmet = None
mo_brueck = None

for uidx, hubbard_u in enumerate(range(u_min, u_max+1, u_step)):
    print("Hubbard-U= %4.1f" % hubbard_u)
    print("===============")
    mol = vayesta.lattmod.Hubbard2D(nsites, nelectron=nsite+doping, hubbard_u=hubbard_u, boundary=('PBC', 'APBC'), tiles=(2,2))
    mf = vayesta.lattmod.LatticeMF(mol)
    mf.kernel()

    # Full system FCI
    if do_fci:
        fci = vayesta.ewf.EWF(mf, solver='FCI', make_rdm1=True, make_rdm2=True, bath_type=None, fragment_type='Site')
        f = fci.make_atom_fragment(list(range(nsite)))
        fci.kernel()
        if f.results.converged:
            e_exact = fci.e_tot / mol.nelectron
            docc_exact = np.einsum("ijkl,i,j,k,l->", f.results.dm2, *(4*[f.c_active[0]]))/2
        else:
            print("Full-system FCI not converged.")
            e_exact = docc_exact = np.nan
        print("E(exact)= %.8f" % e_exact)
        print("Docc(exact)= %.8f" % docc_exact)
    else:
        e_exact = docc_exact = np.nan

    oneshot = SelfConsistentMF(mf, sc_type=None)
    e_oneshot, docc_oneshot = oneshot.kernel(nimp=nimp)

    sc_pdmet = SelfConsistentMF(mf, sc_type='pdmet', tvecs=tvecs)
    e_pdmet, docc_pdmet = sc_pdmet.kernel(nimp=nimp, mo_coeff=mo_pdmet)
    mo_pdmet = sc_pdmet.mf.mo_coeff

    sc_brueck = SelfConsistentMF(mf, sc_type='brueckner', tvecs=tvecs)
    e_brueck, docc_brueck = sc_brueck.kernel(nimp=nimp, mo_coeff=mo_brueck)
    mo_brueck = sc_brueck.mf.mo_coeff

    with open('energies.txt', 'a') as f:
        if uidx == 0:
            f.write("%12s  %16s  %16s  %16s  %16s\n" % ("# Hubbard-U", "Exact", "Oneshot", "PDMET", "Brueckner"))
        f.write('%12.1f  %16.8f  %16.8f  %16.8f  %16.8f\n' % (hubbard_u, e_exact, e_oneshot, e_pdmet, e_brueck))

    with open('docc.txt', 'a') as f:
        if uidx == 0:
            f.write("%12s  %16s  %16s  %16s  %16s\n" % ("# Hubbard-U", "Exact", "Oneshot", "PDMET", "Brueckner"))
        s = nimp // 2 # Take site at the center
        f.write('%12.1f  %16.8f  %16.8f  %16.8f  %16.8f\n' % (hubbard_u, docc_exact, docc_oneshot[s], docc_pdmet[s], docc_brueck[s]))
