"""Extended Density Matrix Embedding Theory (EDMET) method
Author: Charles Scott
email:  cjcargillscott@gmail.com
"""

import pyscf
import pyscf.scf
import logging

from vayesta.edmet.edmet import REDMET
from vayesta.edmet.uedmet import UEDMET

log = logging.getLogger(__name__)

def EDMET(mf, *args, **kwargs):
    """Determine restricted or unrestricted by inspection of mean-field object"""
    if isinstance(mf, pyscf.scf.uhf.UHF):
        return UEDMET(mf, *args, **kwargs)
    elif isinstance(mf, pyscf.scf.rohf.ROHF):
        log.warning("Converting ROHF reference to UHF")
        return UEMET(mf.to_uhf(), *args, **kwargs)
    return REDMET(mf, *args, **kwargs)
