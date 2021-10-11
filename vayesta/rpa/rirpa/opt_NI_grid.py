import numpy as np

def get_grid(rik_MP_L, rik_MP_R, rik_PP_L, rik_PP_R, D, target_rot, npoints, ainit = 1.0):
    """Given all required information for calculation, generate optimal quadrature grid."""

    return gen_ClenCur_quad(ainit, npoints, even = True)
    pass





def gen_ClenCur_quad(a, npoints, even = False):
    symfac = 1.0 + even
    points = [1.0/np.tan(((x+1) /npoints) * (np.pi / symfac)) for x in range(npoints)]
    weights = [a * np.pi * symfac / (2 * npoints * (np.sin(t)**2)) for t in points]
    if even: weights[-1] /= 2
    return points, weights