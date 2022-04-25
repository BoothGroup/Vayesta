"""Density Matrix Embedding Theory (DMET) method
Author: Charles Scott
email:  cjcargillscott@gmail.com
"""

import pyscf
import pyscf.scf

try:
    import cvxpy
except ModuleNotFoundError as e:
    raise ModuleNotFoundError("CVXPY is required for DMET correlation potential fitting.")

from .dmet import RDMET
from .udmet import UDMET

def DMET(mf, *args, **kwargs):
    """Determine restricted or unrestricted by inspection of mean-field object"""
    if isinstance(mf, pyscf.scf.uhf.UHF):
        return UDMET(mf, *args, **kwargs)
    return RDMET(mf, *args, **kwargs)
