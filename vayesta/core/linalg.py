import logging

import numpy as np

log = logging.getLogger(__name__)

def recursive_block_svd(a, n, tol=1e-10, maxblock=100):
    """Perform SVD of rectangular, offdiagonal blocks of a matrix recursively.

    Parameters
    ----------
    a : (m, m) array
        Input matrix.
    n : int
        Number of rows of the first offdiagonal block.
    tol : float, optional
        Singular values below the tolerance are considered uncoupled. Default: 1e^10.
    maxblock : int, optional
        Maximum number of recursions. Default: 100.

    Returns
    -------
    coeff : (m-n, m-n) array
        Coefficients.
    sv : (m-n) array
        Singular values.
    order : (m-n) array
        Orders.
    """
    size = a.shape[-1]
    log.debugv("Recursive block SVD of %dx%d matrix" % a.shape)
    coeff = np.eye(size)
    sv = np.full((size-n,), 0.0)
    orders = np.full((size-n,), -1)

    ndone = 0
    low = np.s_[:n]
    env = np.s_[n:]

    for order in range(1, maxblock+1):
        blk = np.linalg.multi_dot((coeff.T, a, coeff))[low,env]
        u, s, vh = np.linalg.svd(blk)
        rot = vh.T.conj()
        ncpl = np.count_nonzero(s >= tol)
        log.debugv("Found %d bath orbitals with singular values %r" % (ncpl, s[:ncpl].tolist()))
        coeff[:,env] = np.dot(coeff[:,env], rot)
        # Set singular value and order array
        sv[ndone:(ndone+ncpl)] = s[:ncpl]
        orders[ndone:(ndone+ncpl)] = order

        # Update spaces
        ndone = n
        low = np.s_[ndone:(ndone+ncpl)]
        env = np.s_[(ndone+ncpl):]
        ndone += ncpl
        if ndone == size:
            log.debugv("All bath orbitals found; exiting.")
            break
        if ncpl == 0:
            log.debugv("Remaining environment orbitals are decoupled; exiting.")
        if ndone > size:
            raise RuntimeError()
            break
    else:
        log.debugv("Not all bath orbitals found in %d recusions", maxblock)

    coeff = coeff[n:,n:]
    assert np.allclose(np.dot(coeff.T, coeff)-np.eye(coeff.shape[-1]), 0)
    return coeff, sv, orders
