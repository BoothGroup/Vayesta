import numpy as np

from vayesta.core.util import *

import scipy.optimize


def get_grid(rik_MP_L, rik_MP_R, rik_PP_L, rik_PP_R, D, target_rot, npoints, ainit = 1.0):
    """Given all required information for calculation, generate optimal quadrature grid."""

    return gen_ClenCur_quad(ainit, npoints, even = True)
    pass





def gen_ClenCur_quad(a, npoints, even = False):
    symfac = 1.0 + even
    # If even we only want points up to t <= pi/2
    tvals = [(j/npoints) * (np.pi / symfac ) for j in range(1, npoints+1)]

    points = [a/np.tan(t) for t in tvals]
    weights = [a * np.pi * symfac / (2 * npoints * (np.sin(t)**2)) for t in tvals]
    if even: weights[-1] /= 2
    return points, weights


def get_grid_opt_integral_nodiff(rik_MP_L, rik_MP_R, D, target_rot, npoints, ainit = 1.0, ):
    """Generate optimal quadrature grid via minimising difference in a diagonal approximation to our matrices."""
    mat = D**2 + einsum("np,np->p",rik_MP_L, rik_MP_R)

    def get_val(a):
        res= sum(eval_diag_approx_val(a, npoints, mat) - mat**(0.5))
        #print(a,res)
        return res
    def get_deriv1(a):
        return sum(eval_diag_approx_deriv1(a, npoints, mat))
    def get_deriv2(a):
        return sum(eval_diag_approx_deriv2(a, npoints, mat))

    delta = 1e-6
    print(get_val(ainit))
    print("Testing derivs...")
    print(get_deriv1(ainit), (get_val(ainit+delta) - get_val(ainit - delta)) / (2*delta))
    print(get_deriv2(ainit), (get_deriv1(ainit + delta) - get_deriv1(ainit - delta)) / (2*delta))

    solve = scipy.optimize.newton(get_val, x0=ainit, fprime=get_deriv1, fprime2=get_deriv2)
    print("!",solve)
    return gen_ClenCur_quad(solve, npoints, True)

def get_grid_opt_integral_diff1(rik_MP_L, rik_MP_R, rik_PP_L, rik_PP_R, D, target_rot, npoints, ainit = 1.0, ):
    """Generate optimal quadrature grid via minimising difference in a diagonal approximation to our matrices."""
    mat = D**2 + einsum("np,np->p",rik_MP_L, rik_MP_R)
    mat2 = D**2 + einsum("np,np->p",rik_PP_L, rik_PP_R)

    def get_val(a):
        res = sum(eval_diag_approx_val(a, npoints, mat) - mat**(0.5) - eval_diag_approx_val(a, npoints, mat2) + mat2**(0.5))
        #print(a,res)
        return res
    def get_deriv1(a):
        return sum(eval_diag_approx_deriv1(a, npoints, mat) - eval_diag_approx_deriv1(a, npoints, mat2))
    def get_deriv2(a):
        return sum(eval_diag_approx_deriv2(a, npoints, mat) - eval_diag_approx_deriv2(a, npoints, mat2))


    delta = 1e-6
    print(get_val(ainit))
    print("Testing derivs...")
    print(get_deriv1(ainit), (get_val(ainit+delta) - get_val(ainit - delta)) / (2*delta))
    print(get_deriv2(ainit), (get_deriv1(ainit + delta) - get_deriv1(ainit - delta)) / (2*delta))
    #print("Mat1 deriv:")
    #print(eval_diag_approx_deriv1(ainit, npoints, mat))
    #print((eval_diag_approx_val(ainit+delta, npoints, mat) - eval_diag_approx_val(ainit, npoints, mat))/delta)
    #print("Mat2 deriv:")
    #print(eval_diag_approx_deriv1(ainit, npoints, mat2))
    #print((eval_diag_approx_val(ainit + delta, npoints, mat2) - eval_diag_approx_val(ainit, npoints, mat2)) / delta)
    #print("Mat1 deriv2")
    #print(eval_diag_approx_deriv2(ainit, npoints, mat))
    #print((eval_diag_approx_deriv1(ainit+delta, npoints, mat) - eval_diag_approx_deriv1(ainit, npoints, mat))/delta)
    #print("Mat2 deriv2")
    #print(eval_diag_approx_deriv2(ainit, npoints, mat2))
    #print((eval_diag_approx_deriv1(ainit+delta, npoints, mat2) - eval_diag_approx_deriv1(ainit, npoints, mat2))/delta)

    solve = scipy.optimize.newton(get_val, x0=ainit, fprime=get_deriv1, fprime2=get_deriv2)
    print("Optimised quadrature grid with a={:4.2e}".format(solve))
    return gen_ClenCur_quad(solve, npoints, True)

def get_grid_opt_integral_diff2(rik_MP_L, rik_MP_R, D, target_rot, npoints, ainit = 1.0, ):
    """Generate optimal quadrature grid via minimising difference in a diagonal approximation to our matrices."""
    mat = D**2 + einsum("np,np->p",rik_MP_L, rik_MP_R)
    # In this approach, we only treat the dominant diagonal contribution to P approximately.
    mat2 = D**2

    def get_val(a):
        res = sum(eval_diag_approx_val(a, npoints, mat) - mat**(0.5) - eval_diag_approx_val(a, npoints, mat2) + mat2**(0.5))
        #print(a,res)
        return res
    def get_deriv1(a):
        return sum(eval_diag_approx_deriv1(a, npoints, mat) - eval_diag_approx_deriv1(a, npoints, mat2))
    def get_deriv2(a):
        return sum(eval_diag_approx_deriv2(a, npoints, mat) - eval_diag_approx_deriv2(a, npoints, mat2))


    delta = 1e-6
    print(get_val(ainit))
    print("Testing derivs...")
    print(get_deriv1(ainit), (get_val(ainit+delta) - get_val(ainit - delta)) / (2*delta))
    print(get_deriv2(ainit), (get_deriv1(ainit + delta) - get_deriv1(ainit - delta)) / (2*delta))
    #print("Mat1 deriv:")
    #print(eval_diag_approx_deriv1(ainit, npoints, mat))
    #print((eval_diag_approx_val(ainit+delta, npoints, mat) - eval_diag_approx_val(ainit, npoints, mat))/delta)
    #print("Mat2 deriv:")
    #print(eval_diag_approx_deriv1(ainit, npoints, mat2))
    #print((eval_diag_approx_val(ainit + delta, npoints, mat2) - eval_diag_approx_val(ainit, npoints, mat2)) / delta)
    #print("Mat1 deriv2")
    #print(eval_diag_approx_deriv2(ainit, npoints, mat))
    #print((eval_diag_approx_deriv1(ainit+delta, npoints, mat) - eval_diag_approx_deriv1(ainit, npoints, mat))/delta)
    #print("Mat2 deriv2")
    #print(eval_diag_approx_deriv2(ainit, npoints, mat2))
    #print((eval_diag_approx_deriv1(ainit+delta, npoints, mat2) - eval_diag_approx_deriv1(ainit, npoints, mat2))/delta)

    solve = scipy.optimize.newton(get_val, x0=ainit, fprime=get_deriv1, fprime2=get_deriv2)
    print(solve)
    return gen_ClenCur_quad(solve, npoints, True)


def eval_diag_approx_derivs(a, npoints, D):
    points, weights = gen_ClenCur_quad(a, npoints, even=True)
    val = np.zeros_like(D)
    deriv1 = np.zeros_like(D)
    deriv2 = np.zeros_like(D)
    # Evaluate the value at all points, as well as the first and second derivatives for whatever optimisation we want to
    # do later.
    for p,w in zip(points, weights):
        M = (D + p**2)**(-1)
        contrib = np.full_like(D, fill_value=1.0) - (p ** 2) * M
        # This evaluates the derivative of our integrand w.r.t. the position of the point.
        contrib_deriv1 = 2 * ((p ** 3) * (M**2) - p * M)
        contrib_deriv2 = - 2 * M + 10 * (p ** 2) * M - 8 * (p**4) * (M**3)
        # We seek the derivatives as we change a, and since this is a multiplicative shift only the first derivative
        # is nonzero, and equal to the point position itself.
        val += (w / np.pi) * contrib
        deriv1 += (w / np.pi) * contrib_deriv1 * p
        deriv2 += (w / np.pi) * contrib_deriv2 * p**2
    return val, deriv1, deriv2

def eval_diag_approx_val(a, npoints, D):
    points, weights = gen_ClenCur_quad(a, npoints, even=True)
    val = np.zeros_like(D)
    # Evaluate the value at all points, as well as the first and second derivatives for whatever optimisation we want to
    # do later.
    for p,w in zip(points, weights):
        M = (D + p**2)**(-1)
        contrib = np.full_like(D, fill_value=1.0) - (p ** 2) * M
        val += (w / np.pi) * contrib
    return val

def eval_diag_approx_deriv1(a, npoints, D):
    points, weights = gen_ClenCur_quad(a, npoints, even=True)
    deriv1 = np.zeros_like(D)
    # Evaluate the value at all points, as well as the first and second derivatives for whatever optimisation we want to
    # do later.
    for p,w in zip(points, weights):
        M = (D + p**2)**(-1)
        # This evaluates the derivative of our integrand w.r.t. the position of the point.
        contrib = np.full_like(D, fill_value=1.0) - (p ** 2) * M
        contrib_deriv1 = 2 * ((p ** 3) * (M**2) - p * M)
        # We seek the derivatives as we change a, so need the change in each point with a.
        # Since a is a multiplicative shift only the first derivative is nonzero, and equal to the point position itself
        # divided by a.
        deriv1 += (w / np.pi) * contrib_deriv1 * (p/a) + (w / (np.pi * a)) * contrib
    return deriv1

def eval_diag_approx_deriv2(a, npoints, D):
    points, weights = gen_ClenCur_quad(a, npoints, even=True)
    deriv2 = np.zeros_like(D)
    # Evaluate the value at all points, as well as the first and second derivatives for whatever optimisation we want to
    # do later.
    for p,w in zip(points, weights):
        M = (D + p**2)**(-1)
        # This evaluates the derivative of our integrand w.r.t. the position of the point.
        contrib = np.full_like(D, fill_value=1.0) - (p ** 2) * M
        contrib_deriv1 = 2 * ((p ** 3) * (M**2) - p * M)
        contrib_deriv2 = - 2 * M + 10 * (p ** 2) * (M**2) - 8 * (p**4) * (M**3)
        # We seek the derivatives as we change a, so need the change in each point with a.
        # Since a is a multiplicative shift only the first derivative is nonzero, and equal to the point position itself
        # divided by a.
        deriv2 += (w / np.pi) * contrib_deriv2 * (p/a)**2 + 2 * (w / (np.pi * a)) * contrib_deriv1 * (p/a)
    return deriv2