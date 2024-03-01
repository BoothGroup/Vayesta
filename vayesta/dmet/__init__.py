"""Density Matrix Embedding Theory (DMET) method
Author: Charles Scott
email:  cjcargillscott@gmail.com
"""

import pyscf
import pyscf.scf
import logging

try:
    import cvxpy
except ModuleNotFoundError as e:
    cvxpy = None
    raise ModuleNotFoundError("CVXPY is required for DMET correlation potential fitting.") from e

from vayesta.dmet.dmet import RDMET
from vayesta.dmet.udmet import UDMET

log = logging.getLogger(__name__)


def DMET(mf, *args, **kwargs):
    """Determine restricted or unrestricted by inspection of mean-field object"""
    if isinstance(mf, pyscf.scf.uhf.UHF) or isinstance(mf, pyscf.pbc.scf.kuhf.KUHF):
        return UDMET(mf, *args, **kwargs)
    elif isinstance(mf, pyscf.scf.rohf.ROHF) or isinstance(mf, pyscf.pbc.scf.krohf.KROHF):
        log.warning("Converting ROHF reference to UHF")
        return UDMET(mf.to_uhf(), *args, **kwargs)
    return RDMET(mf, *args, **kwargs)
