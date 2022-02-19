"""Extended Density Matrix Embedding Theory (EDMET) method
Author: Charles Scott
email:  cjcargillscott@gmail.com
"""

import pyscf
import pyscf.scf

from .edmet import REDMET
from .uedmet import UEDMET

def EDMET(mf, *args, **kwargs):
    """Determine restricted or unrestricted by inspection of mean-field object"""
    if isinstance(mf, pyscf.scf.uhf.UHF):
        return UEDMET(mf, *args, **kwargs)
    return REDMET(mf, *args, **kwargs)