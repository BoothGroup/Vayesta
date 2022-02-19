"""Embedded wave function (EWF) method
Author: Max Nusspickel
email:  max.nusspickel@gmail.com
"""

import pyscf
import pyscf.scf

#from .ewf import EWF as REWF
from .ewf import REWF
from .uewf import UEWF

def EWF(mf, *args, **kwargs):
    """Determine restricted or unrestricted by inspection of mean-field object"""
    if isinstance(mf, pyscf.scf.uhf.UHF):
        return UEWF(mf, *args, **kwargs)
    return REWF(mf, *args, **kwargs)
