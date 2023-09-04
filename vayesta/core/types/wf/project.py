"""Utility functions for projection of wave functions."""

import numpy as np
from vayesta.core.util import einsum, dot


def project_c1(c1, p):
    if c1 is None:
        return None
    if p is None:
        return c1
    return np.dot(p, c1)


def project_c2(c2, p):
    if c2 is None:
        return None
    if p is None:
        return c2
    return np.tensordot(p, c2, axes=1)


def project_uc1(c1, p):
    if c1 is None:
        return None
    if p is None:
        return c1
    return (project_c1(c1[0], p[0]), project_c1(c1[1], p[1]))


def project_uc2(c2, p):
    if c2 is None:
        return None
    if p is None:
        return c2
    c2ba = c2[2] if len(c2) == 4 else c2[1].transpose(1, 0, 3, 2)
    return (
        project_c2(c2[0], p[0]),
        project_c2(c2[1], p[0]),
        # einsum('xi,ij...->ix...', p[1], c2[1]),
        project_c2(c2ba, p[1]),
        project_c2(c2[-1], p[1]),
    )


def symmetrize_c2(c2, inplace=True):
    if not inplace:
        c2 = c2.copy()
    c2 = (c2 + c2.transpose(1, 0, 3, 2)) / 2
    return c2


def symmetrize_uc2(c2, inplace=True):
    if not inplace:
        c2 = tuple(x.copy() for x in c2)
    c2aa = symmetrize_c2(c2[0])
    c2bb = symmetrize_c2(c2[-1])
    # Mixed spin:
    c2ab = c2[1]
    if len(c2) == 4:
        c2ab = (c2ab + c2[2].transpose(1, 0, 3, 2)) / 2
    return (c2aa, c2ab, c2bb)


def transform_c1(c1, to, tv):
    if c1 is None:
        return None
    return dot(to.T, c1, tv)


def transform_c2(c2, to, tv, to2=None, tv2=None):
    if c2 is None:
        return None
    if to2 is None:
        to2 = to
    if tv2 is None:
        tv2 = tv
    # Use einsum for now- tensordot would be faster but less readable.
    return einsum("ijab,iI,jJ,aA,bB->IJAB", c2, to, to2, tv, tv2)


def transform_uc1(c1, to, tv):
    if c1 is None:
        return None
    return (transform_c1(c1[0], to[0], tv[0]), transform_c1(c1[1], to[1], tv[1]))


def transform_uc2(c2, to, tv):
    if c2 is None:
        return None
    c2aa = transform_c2(c2[0], to[0], tv[0])
    c2ab = transform_c2(c2[1], to[0], tv[0], to[1], tv[1])
    c2bb = transform_c2(c2[-1], to[1], tv[1])
    if len(c2) == 4:
        c2ba = transform_c2(c2[2], to[1], tv[1], to[0], tv[0])
        return c2aa, c2ab, c2ba, c2bb
    else:
        return c2aa, c2ab, c2bb
