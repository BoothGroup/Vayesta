
import numpy as np
import scipy
from vayesta.core.util import *
from vayesta.rpa.rirpa import opt_NI_grid

from vayesta.rpa.rirpa import opt_NI_grid


def eval_eta0(D, ri_P, ri_M, target_rot, npoints = 100, ainit = 1.0):
    """Given a 1D representation of the irreducible polarisation propagator and the low-rank decompositions for
    the difference of A+B and A-B from this, evaluate the zeroth moment of the dd response. This involves performing a
    numerical integration with an optimised quadrature grid.
    We have tried to specify the expected computational scaling of each step in this procedure, for the sake of future
    optimisation."""

    # Some steps of this construction are O(N^4), but only need to be performed once.
    rik_MP_L, rik_MP_R = construct_product_RI(D, ri_M, ri_P)
    rik_PP_L, rik_PP_R = construct_product_RI(D, ri_P, ri_P)
    # Construct P^{-1} here, to use in both constructing quadrature and obtaining final estimate.
    ri_Pinv = construct_inverse_RI(D, ri_P)

    #print("MP:")
    #print(np.diag(D**2) + dot(rik_MP_L.T, rik_MP_R))
    #print("PP:")
    #print(np.diag(D ** 2) + dot(rik_PP_L.T, rik_PP_R))
    #print("Pinv:")
    #print(np.diag(D ** -1) + dot(ri_Pinv.T, ri_Pinv))
    # This only optimises wrt the trace of the desired quantity, so should have a lower scaling than other steps.
    #grid, weights = opt_NI_grid.get_grid(rik_MP_L, rik_MP_R, rik_PP_L, rik_PP_R, D, target_rot, npoints, ainit)

#    grid, weights = opt_NI_grid.get_grid_opt_integral_diff1(rik_MP_L, rik_MP_R, rik_PP_L, rik_PP_R, D, target_rot, npoints, ainit)
    grid, weights = opt_NI_grid.get_grid_opt_integral_nodiff(rik_MP_L, rik_MP_R, D, target_rot, npoints, ainit)



    integral = np.zeros(D.shape*2)
    for point, weight in zip(grid, weights):
        # Evaluating each component will scale as O(N^4) thanks to construction of the Q intermediates.
        contrib = eval_eta0_contrib(point, rik_MP_L, rik_MP_R, rik_PP_L, rik_PP_R, D, target_rot)
        #print("!",point)
        #print(contrib)
        integral += weight * contrib

    # Now need to multiply by the inverse of P, using low-rank expression we've already constructed.
    # Note ri contrib is negative.
    #print(integral.shape, ri_Pinv.shape, ri_P.shape)
    mom0 = einsum("pq,q->pq",integral,D**(-1)) - np.dot(np.dot(integral,ri_Pinv.T), ri_Pinv)
    # Now have our resulting estimate of the zeroth moment!
    return mom0 + np.eye(D.shape[0]), integral

def eval_eta0_contrib(freq, rik_MP_L, rik_MP_R, rik_PP_L, rik_PP_R, D, target_rot):
    """Evaluate contribution to RI integral at a particular frequency point.
    This step scales with the grid size, so should be efficiently parallelised.
    We only obtain a contribution to a target rotation of the excitation space, so reducing the scaling of this
    procedure.
    """
    G = construct_G(freq, D)
    n_aux_MP = rik_MP_L.shape[0]
    n_aux_PP = rik_PP_L.shape[0]

    # Construct rotation required for LHS of our expression; the number of contributions we seek determines the
    # computational scaling of evaluating our contributions from their initial low-rank representations.
    # If the full space is required evaluation scales as O(N^5), while for reduced scaling it only requires O(N^4) or
    # O(N^3) for a linear or constant number of degrees of freedom.
    lrot = einsum("pn,n,n->pn", target_rot, D**(-1), G)
    rrot = np.multiply(G, D**(-1))

    # Construction of these intermediates is one of the limiting steps of the entire procedure (scales as O(N^4) at
    # each frequency point, due to summing over all excitations). Could be alleviated with pre-screening, but not for
    # now.
    Q_MP = einsum("np,p,mp->nm", rik_MP_R, rrot, rik_MP_L)
    Q_PP = einsum("np,p,mp->nm", rik_PP_R, rrot, rik_PP_L)

    # Don't use multidot or similar here to ensure index spanning full space is contracted last, as otherwise more
    # expensive.
    return (1 / np.pi) * (freq ** 2) * (
        np.dot(np.dot(np.dot(lrot, rik_MP_L.T), np.linalg.inv(np.eye(n_aux_MP) + Q_MP)), einsum("np,p->np", rik_MP_R, rrot)) -
        np.dot(np.dot(np.dot(lrot, rik_PP_L.T), np.linalg.inv(np.eye(n_aux_PP) + Q_PP)), einsum("np,p->np", rik_PP_R, rrot)))

def construct_G(freq, D):
    """Evaluate G = D (D**2 + \omega**2 I)**(-1), given frequency and diagonal of D."""
    return np.multiply(D, (D ** 2 + freq ** 2) ** (-1))

def construct_product_RI(D, ri_1, ri_2):
    """Given two matrices expressed as low-rank modifications, cderi_1 and cderi_2, of some full-rank matrix D,
    construct the RI expression for the deviation of their product from D**2.
    The rank of the resulting deviation is at most the sum of the ranks of the original modifications."""
    # Construction of this matrix is the computationally limiting step of this construction (O(N^4)) in our usual use,
    # but we only need to perform it once per calculation since it's frequency-independent.
    if type(ri_1) == np.ndarray:
        ri_1_L = ri_1_R = ri_1
    else:
        (ri_1_L, ri_1_R) = ri_1

    if type(ri_2) == np.ndarray:
        ri_2_L = ri_2_R = ri_2
    else:
        (ri_2_L, ri_2_R) = ri_2


    U = np.dot(ri_1, ri_2.T)

    ri_L = np.concatenate([ri_1, einsum("p,np->np", D, ri_2) + np.dot(U.T, ri_1) / 2], axis=0)
    ri_R = np.concatenate([einsum("p,np->np", D, ri_1) + np.dot(U, ri_2) / 2, ri_2], axis=0)
    return ri_L, ri_R

def construct_inverse_RI(D, ri):
    if type(ri) == np.ndarray:
        ri_L = ri_R = ri
    else:
        (ri_L, ri_R) = ri

    naux = ri_R.shape[0]
    # This construction scales as O(N^4).
    U = einsum("np,p,mp->nm", ri_R, D ** (-1), ri_L)
    # This inversion and square root should only scale as O(N^3).
    U = np.linalg.inv(np.eye(naux) + U)
    Urt = scipy.linalg.sqrtm(U)
    # Evaluate the resulting RI
    if type(ri) == np.ndarray:
        return einsum("p,np,nm->mp", D ** (-1), ri_L, Urt)
    else:
        return einsum("p,np,nm->mp", D ** (-1), ri_L, Urt), einsum("p,np,nm->mp", D ** (-1), ri_R, Urt.T)