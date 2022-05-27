from timeit import default_timer as timer
import numpy as np
import pyscf
import pyscf.gto
import pyscf.scf
import pyscf.cc
import vayesta
import vayesta.ewf
from vayesta.misc import molecules

solver 'MP2'
eta = 1e-6

for n in range(1, 14):
    mol = pyscf.gto.Mole()
    mol.atom = molecules.alkane(n)
    mol.basis = 'cc-pVDZ'
    mol.output = 'pyscf.out'
    mol.build()

    # Hartree-Fock
    mf = pyscf.scf.RHF(mol)
    mf.kernel()

    # Embedded CCSD
    if solver == 'MP2':
        emb = vayesta.ewf.EWF(mf, bno_threshold=eta, solver='MP2')
    else:
        emb = vayesta.ewf.EWF(mf, bno_threshold=eta, t_as_lambda=True)
    emb.kernel()

    t0 = timer()
    dm1_slow = emb.make_rdm1(slow=True)
    t_slow = (timer() - t0)

    t0 = timer()
    dm1_nosvd = emb.make_rdm1(svd_tol=None)
    t_nosvd = (timer() - t0)

    t0 = timer()
    dm1_svd = emb.make_rdm1(svd_tol=1e-3)
    t_svd = (timer() - t0)

    natom = mol.natm
    nao = mol.nao
    with open('timings.txt', 'a') as f:
        f.write('%2d  %4d  %12.6f  %12.6f  %12.6f\n' % (natom, nao, t_slow, t_nosvd, t_svd))

    err_nosvd = np.linalg.norm(dm1_nosvd - dm1_slow)
    err_svd = np.linalg.norm(dm1_svd - dm1_slow)
    with open('errors.txt', 'a') as f:
        f.write('%2d  %4d  %12.6e  %12.6e\n' % (natom, nao, err_nosvd, err_svd))
