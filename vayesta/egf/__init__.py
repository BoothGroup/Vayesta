"""Embedded Green's function (EGF) method
Author: Basil Ibrahim
email:  basilibrahim95@gmail.com
"""

import pyscf
import pyscf.scf
import logging

from vayesta.egf.egf import REGF

log = logging.getLogger(__name__)


def EGF(mf, *args, **kwargs):
    """Determine restricted or unrestricted by inspection of mean-field object"""
    if isinstance(mf, (pyscf.scf.uhf.UHF, pyscf.pbc.scf.uhf.UHF, pyscf.pbc.scf.kuhf.KUHF)):
        raise NotImplementedError()
    elif isinstance(mf, (pyscf.scf.rohf.ROHF, pyscf.pbc.scf.rohf.ROHF, pyscf.pbc.scf.krohf.KROHF)):
        log.warning("Converting ROHF reference to UHF")
        raise NotImplementedError()
    return REGF(mf, *args, **kwargs)
