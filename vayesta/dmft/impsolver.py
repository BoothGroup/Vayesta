import numpy as np

import pyscf
import pyscf.fci

from helper import *

__all__ = ["__kernel__"]


def make_cluster_ham(h1e, eri, nimp, bathpol, bathcpl):
    nbath = len(bathpol)
    ncluster = nimp + nbath
    h1e_cl = np.zeros((ncluster, ncluster))
    imp = np.s_[:nimp]
    bath = np.s_[nimp:]
    h1e_cl[imp,imp] = h1e[imp,imp]
    h1e_cl[bath,bath] = np.diag(bathpol)
    h1e_cl[imp,bath] = bathcpl
    h1e_cl[bath,imp] = bathcpl.T
    eri_cl = np.zeros(4*[ncluster])
    eri_cl[imp,imp,imp,imp] = eri

    return h1e_cl, eri_cl

def canonicalize_h1e(h1e, eri):
    e, c = np.linalg.eigh(h1e)
    h1e = np.diag(e)
    eri = einsum("ijkl,ia,jb,kc,ld->abcd", eri, c, c, c, c)
    return h1e, eri


def solve_ground_state(h1e, eri, nelec=None):

    ncluster = h1e.shape[-1]
    if nelec is None:
        nelec = ncluster
        print(nelec)

    fci = pyscf.fci.direct_spin0.FCI()
    fci.verbose = 100

    en0, wf0 = fci.kernel(h1e, eri, ncluster, nelec)
    dm1, dm2 = fci.make_rdm12(wf0, ncluster, nelec)
    docc = np.asarray([dm2[i,i,i,i]/2 for i in range(ncluster)])

    return en0, wf0, dm1, docc



def kernel(h1e, eri, nimp, bathpol, bathcpl, nelec=None, canonicalize=True):

    h1e_cl, eri_cl = make_cluster_ham(h1e, eri, nimp, bathpol, bathcpl)

    if canonicalize:
        h1e_cl, eri_cl = canonicalize_h1e(h1e_cl, eri_cl)

    en0, wf0, dm1, docc = solve_ground_state(h1e_cl, eri_cl)

    print(dm1)
    print(docc)


