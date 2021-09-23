import pyscf
import pyscf.scf

from .sao import SAO_Fragmentation, SAO_Fragmentation_UHF
from .iao import IAO_Fragmentation
from .iaopao import IAOPAO_Fragmentation
from .site import Site_Fragmentation, Site_Fragmentation_UHF

def is_uhf(mf):
    return isinstance(mf, pyscf.scf.uhf.UHF)

def make_sao_fragmentation(mf, *args, **kwargs):
    if is_uhf(mf):
        return SAO_Fragmentation_UHF(mf, *args, **kwargs)
    return SAO_Fragmentation(mf, *args, **kwargs)

def make_site_fragmentation(mf, *args, **kwargs):
    if is_uhf(mf):
        return Site_Fragmentation_UHF(mf, *args, **kwargs)
    return Site_Fragmentation(mf, *args, **kwargs)

def make_iao_fragmentation(mf, *args, **kwargs):
    if is_uhf(mf):
        raise NotImplementedError()
    return IAO_Fragmentation(mf, *args, **kwargs)

def make_iaopao_fragmentation(mf, *args, **kwargs):
    if is_uhf(mf):
        raise NotImplementedError()
    return IAOPAO_Fragmentation(mf, *args, **kwargs)
