"""Utility functions for projection of wave functions."""

import numpy as np


def project_c1(c1, p):
    if c1 is None: return None
    if p is None: return c1
    return np.dot(p, c1)

def project_c2(c2, p):
    if c2 is None: return None
    if p is None: return c2
    return np.tensordot(p, c2, axes=1)

def project_uc1(c1, p):
    if c1 is None: return None
    if p is None: return c1
    return (project_c1(c1[0], p[0]),
            project_c1(c1[1], p[1]))

def project_uc2(c2, p):
    if c2 is None: return None
    if p is None: return c2
    c2ba = (c2[2] if len(c2) == 4 else c2[1].transpose(1,0,3,2))
    return (project_c2(c2[0], p[0]),
            project_c2(c2[1], p[0]),
            #einsum('xi,ij...->ix...', p[1], c2[1]),
            project_c2(c2ba, p[1]),
            project_c2(c2[-1], p[1]))

def symmetrize_c2(c2, inplace=True):
    if not inplace:
        c2 = c2.copy()
    c2 = (c2 + c2.transpose(1,0,3,2))/2
    return c2

def symmetrize_uc2(c2, inplace=True):
    if not inplace:
        c2 = tuple(x.copy() for x in c2)
    c2aa = symmetrize_c2(c2[0])
    c2bb = symmetrize_c2(c2[-1])
    # Mixed spin:
    c2ab = c2[1]
    if len(c2) == 4:
        c2ab = (c2ab + c2[2].transpose(1,0,3,2))/2
    return (c2aa, c2ab, c2bb)
