import numpy as np

import pyscf
from pyscf.lib.parameters import BOHR

def to_bohr(a, unit):
    if unit[0].upper() == 'A':
        return  a/BOHR
    if unit[0].upper() == 'B':
        return a
    raise ValueError("Unknown unit: %s" % unit)

def orbital_sign_convention(mo_coeff, inplace=True):
    if not inplace:
        mo_coeff = mo_coeff.copy()
    absmax = np.argmax(abs(mo_coeff), axis=0)
    nmo = mo_coeff.shape[-1]
    swap = mo_coeff[absmax,np.arange(nmo)] < 0
    mo_coeff[:,swap] *= -1
    signs = np.ones((nmo,), dtype=int)
    signs[swap] = -1
    return mo_coeff, signs
