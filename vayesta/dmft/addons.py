import numpy as np


def make_1D_hubbard_model(nsites, t=1.0, boundary=None):
    h1e = np.zeros((nsites, nsites))

    if np.isscalar(boundary):
        bphase = boundary
    elif boundary in ("periodic", "pbc"):
        bphase = 1
    elif boundary in ("anti-periodic", "anti-pbc", "apbc"):
        bphase = -1
    else:
        bphase = -1 if (nsites % 4) == 0 else 1

    for i in range(nsites-1):
        h1e[i,i+1] = h1e[i+1,i] = -t
    h1e[nsites-1,0] = h1e[0,nsites-1] = bphase * -t

    return h1e
