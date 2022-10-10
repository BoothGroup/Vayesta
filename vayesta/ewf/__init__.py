"""Embedded wave function (EWF) method
Author: Max Nusspickel
email:  max.nusspickel@gmail.com
"""

import pyscf
import pyscf.scf
import logging 

#from .ewf import EWF as REWF
from .ewf import REWF
from .uewf import UEWF

log = logging.getLogger(__name__)

def EWF(mf, *args, **kwargs):
    """Determine restricted or unrestricted by inspection of mean-field object"""
    if isinstance(mf, pyscf.scf.uhf.UHF):
        return UEWF(mf, *args, **kwargs)
    elif isinstance(mf, pyscf.scf.rohf.ROHF):
        log.warning("Converting ROHF reference to UHF")
        return UEWF(mf.to_uhf(), *args, **kwargs)
    return REWF(mf, *args, **kwargs)
