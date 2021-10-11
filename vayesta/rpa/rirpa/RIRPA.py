import numpy as np
from vayesta.core.util import *
from vayesta.rpa.rirpa import opt_NI_grid


def eval_eta0(D, cderi, ri_xc, target_rot):
    """Given the RI decompositions for our ERIs and XC kernel, along with a 1D representation of the irreducible
    polarisation propagator, evaluate the zeroth moment of the dd response. This involves performing a numerical
    integration with an optimised quadrature grid."""

    rik_MP_L, rik_MP_R, rik_PP_L, rik_PP_R = cderi


    grid, weights = opt_NI_grid.get_grid(rik_MP_L, rik_MP_R, rik_PP_L, rik_PP_R, D, target_rot)

    res = np.zeros(D.shape*2)
    for point, weight in zip(grid, weights):
        res += weight * eval_eta0_contrib(point, rik_MP_L, rik_MP_R, rik_PP_L, rik_PP_R, D, target_rot)

    # Now need to multiply by the inverse of P, using low-rank expression.
    res = res
    # Now have our resulting estimate of the zeroth moment!
    return res + np.eye(D.shape[0])

def eval_eta0_contrib(freq, rik_MP_L, rik_MP_R, rik_PP_L, rik_PP_R, D, target_rot):
    """Evaluate contribution to RI integral at a particular frequency point.
    This step scales with the grid size, so should be efficiently parallelised.
    We only obtain a contribution to a target rotation of the excitation space, so reducing the scaling of this
    procedure.
    """
    G = construct_G(freq, D)
    n_aux_MP = rik_MP_L.shape[0]
    n_aux_PP = rik_PP_L.shape[0]

    # Construction of these intermediates is one of the limiting steps of the entire procedure (scales as O(N^4) at
    # each frequency point, due to summing over all excitations). Could be alleviated with pre-screening, but not for
    # now.
    Q_MP = einsum("np,p,mp->nm", rik_MP_R, D, rik_MP_L)
    Q_PP = einsum("np,p,mp->nm", rik_PP_R, D, rik_PP_L)
    # Construct rotation required for LHS of our expression; the number of contributions we seek determines the
    # computational scaling of evaluating our contributions from their initial low-rank representations.
    # If the full space is required evaluation scales as O(N^5), while for reduced scaling it only requires O(N^4) or
    # O(N^3) for a linear or constant number of degrees of freedom.
    lrot = einsum("pn,n,n->pn", target_rot, D**(-1), G)
    rrot = np.multiply(G, D**-1)
    # Don't use multidot or similar here to ensure index spanning full space is contracted last, as otherwise more
    # expensive.
    return (2 / np.pi) * (freq ** 2) * (
        np.dot(np.dot(np.dot(lrot, rik_MP_L.T), np.linalg.inv(np.eye(n_aux_MP) + 2 * Q_MP)), einsum("np,p->np", rik_MP_R, rrot)) -
        np.dot(np.dot(np.dot(lrot, rik_PP_L.T), np.linalg.inv(np.eye(n_aux_PP) + 2 * Q_PP)), einsum("np,p->np", rik_PP_R, rrot)))

def construct_G(freq, D):
    """Evaulate G = D (D**2 + \omega**2 I)**(-1), given frequency and diagonal of D."""
    return np.multiply(D, (D ** 2 + freq ** 2) ** (-1))


def get_RI_MP_PP(D, cderi, ri_xc):
    """Given D and low-rank expressions for eris and xc kernel, construct low-rank expressions for MP and PP."""
    # First construct low-rank representation for M and P, then for products.
    # Only have eri contrib to P.
    pass

def construct_product_RI(D, cderi_1, cderi_2):
    """Given two matrices expressed as low-rank modifications, cderi_1 and cderi_2, of some full-rank matrix D,
    construct the RI expression for the deviation of their product from D**2.
    The rank of the resulting deviation is at most the sum of the ranks of the original modifications."""
    # Construction of this matrix is the computationally limiting step of this construction (O(N^4)) in our usual use,
    # but we only need to perform it once per calculation since it's frequency-independent.
    U = np.dot(cderi_1, cderi_2.T)

    RI_L = np.concatenate([cderi_1, einsum("p,np->np", D, cderi_2) + np.dot(U.T, cderi_1) / 2], axis=0)
    RI_R = np.concatenate([einsum("p,np->np", D, cderi_1) + np.dot(U, cderi_2) / 2, cderi_2], axis=0)
    return RI_L, RI_R