import logging

import numpy as np

from vayesta.core.util import *

log = logging.getLogger(__name__)


def orthogonalize_mo(c, s, tol=1e-6):
    """Orthogonalize MOs, such that C^T S C = I (identity matrix).

    Parameters
    ----------
    c : ndarray
        MO orbital coefficients.
    s : ndarray
        AO overlap matrix.
    tol : float, optional
        Tolerance.

    Returns
    -------
    c_out : ndarray
        Orthogonalized MO coefficients.
    """
    assert np.all(c.imag == 0)
    assert np.allclose(s, s.T)
    l = np.linalg.cholesky(s)
    c2 = np.dot(l.T, c)
    #chi = np.linalg.multi_dot((c.T, s, c))
    chi = np.dot(c2.T, c2)
    chi = (chi + chi.T)/2
    e, v = np.linalg.eigh(chi)
    assert np.all(e > 0)
    r = einsum("ai,i,bi->ab", v, 1/np.sqrt(e), v)
    c_out = np.dot(c, r)
    chi_out = np.linalg.multi_dot((c_out.T, s, c_out))
    # Check orthogonality within tol
    nonorth = abs(chi_out - np.eye(chi_out.shape[-1])).max()
    if tol is not None and nonorth > tol:
        log.error("Orbital non-orthogonality= %.1e", nonorth)

    return c_out

def get_ssz(dm1, dm2, proj1=None, proj2=None):
    dm1a = dm1/2
    dm2aa = (dm2 - dm2.transpose(0,3,2,1)) / 6
    dm2ab = (dm2/2 - dm2aa)

    if proj1 is None:
        ssz = (einsum('iijj->', dm2aa) - einsum('iijj->', dm2ab))/2
        ssz += einsum('ii->', dm1a)/2
        return ssz
    if proj2 is None:
        proj2 = proj1
    ssz = (einsum('ijkl,ij,kl->', dm2aa, proj1, proj2)
         - einsum('ijkl,ij,kl->', dm2ab, proj1, proj2))/2
    ssz += einsum('ij,ik,jk->', dm1a, proj1, proj2)/2
    return ssz
