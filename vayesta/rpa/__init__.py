"""Random Phase Approximation (RPA)
Author: Charles Scott
email:  cjcargillscott@gmail.com
"""

from .rirpa import ssRIRPA
from .rpa import RPA
from .ssrpa import ssRRPA
from .ssurpa import ssURPA

import pyscf.scf

def ssRPA(mf, *args, **kwargs):
    """Determine restricted or unrestricted by inspection of mean-field object"""
    if isinstance(mf, pyscf.scf.uhf.UHF):
        return ssURPA(mf, *args, **kwargs)
    return ssRRPA(mf, *args, **kwargs)
