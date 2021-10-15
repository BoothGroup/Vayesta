import numpy as np

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