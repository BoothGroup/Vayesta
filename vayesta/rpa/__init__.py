"""Random Phase Approximation (RPA)
Author: Charles Scott
email:  cjcargillscott@gmail.com
"""

from vayesta.rpa.rirpa import ssRIRPA
from vayesta.rpa.rpa import RPA
from vayesta.rpa.ssrpa import ssRRPA
from vayesta.rpa.ssurpa import ssURPA

import pyscf.scf


def ssRPA(mf, *args, **kwargs):
    """Determine restricted or unrestricted by inspection of mean-field object"""
    if isinstance(mf, pyscf.scf.uhf.UHF):
        return ssURPA(mf, *args, **kwargs)
    return ssRRPA(mf, *args, **kwargs)
