import scipy.sparse.linalg
import numpy as np


def cg_inv(A, B, log, inplace=False, M=None, tol=1e-12, **kwargs):
    """
    Using `scipy.sparse.linalg.cg` solve Ax = B for x, where B can be matrix-valued.

    Parameters
    ----------
        A : Sparse representation of NxN array.
        B : Dense NxK array.
        log : logger to write any warnings to.
        inplace : bool, default False.
            Whether result should be stored in-place in array B to minimise memory requirements.
        M : preconditioner approximating inverse of A.
        tol : float.
            convergence threshold for cg solver.
        All other keyword arguments are passed directly to `scipy.sparse.linalg.cg`.
    Returns
    -------
        x : NxK array, solving Ax = B
    """
    n, k = B.shape
    if inplace:
        res = B
    else:
        res = np.zeros_like(B)
    for i in range(k):
        res[:, i], convinfo = scipy.sparse.linalg.cg(A, B[:, i], M=M, tol=tol, **kwargs)
        if convinfo < 0:
            log.critical("Conjugate gradient solver encountered error in iterative inversion!")
            raise RuntimeError("Conjugate gradient solver failed to iterative inversion!")
        elif convinfo > 0:
            log.warning("Iterative inversion not converged to within tolerance in given number of steps.")
    return res
