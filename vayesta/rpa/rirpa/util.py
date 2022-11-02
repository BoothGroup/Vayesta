import scipy.sparse.linalg
import numpy as np


def cg_inv(funcA, B, log, inplace=False, funcM=None, tol=1e-12, **kwargs):
    """
    Using `scipy.sparse.linalg.cg` solve Ax = B for x, where B can be matrix-valued.

    Parameters
    ----------
        funcA : function applying NxN matrix A to arbitrary matrix or vector.
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
    N, k = B.shape

    def linop_factory(func):
        return scipy.sparse.linalg.LinearOperator(shape=(N, N),
                                                  matvec=lambda v: func(v.reshape((N, -1))).reshape(-1))

    linopA = linop_factory(funcA)
    linopM = None if funcM is None else linop_factory(funcM)

    res, convinfo = scipy.sparse.linalg.cg(linopA, B.reshape(-1), M=linopM, tol=tol, **kwargs)
    if convinfo < 0:
        log.critical("Conjugate gradient solver encountered error in iterative inversion!")
        raise RuntimeError("Conjugate gradient solver failed to iterative inversion!")
    elif convinfo > 0:
        log.warning("Iterative inversion not converged to within tolerance in given number of steps.")

    return res.reshape(B.shape)
