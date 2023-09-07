import pyscf
import pyscf.scf

from vayesta.core.fragmentation.sao import SAO_Fragmentation as SAO_Fragmentation_RHF
from vayesta.core.fragmentation.sao import SAO_Fragmentation_UHF
from vayesta.core.fragmentation.iao import IAO_Fragmentation as IAO_Fragmentation_RHF
from vayesta.core.fragmentation.iao import IAO_Fragmentation_UHF
from vayesta.core.fragmentation.iaopao import IAOPAO_Fragmentation as IAOPAO_Fragmentation_RHF
from vayesta.core.fragmentation.iaopao import IAOPAO_Fragmentation_UHF
from vayesta.core.fragmentation.site import Site_Fragmentation as Site_Fragmentation_RHF
from vayesta.core.fragmentation.site import Site_Fragmentation_UHF
from vayesta.core.fragmentation.cas import CAS_Fragmentation as CAS_Fragmentation_RHF
from vayesta.core.fragmentation.cas import CAS_Fragmentation_UHF


def SAO_Fragmentation(emb, *args, **kwargs):
    if emb.is_uhf:
        return SAO_Fragmentation_UHF(emb, *args, **kwargs)
    return SAO_Fragmentation_RHF(emb, *args, **kwargs)


def Site_Fragmentation(emb, *args, **kwargs):
    if emb.is_uhf:
        return Site_Fragmentation_UHF(emb, *args, **kwargs)
    return Site_Fragmentation_RHF(emb, *args, **kwargs)


def IAO_Fragmentation(emb, *args, **kwargs):
    if emb.is_uhf:
        return IAO_Fragmentation_UHF(emb, *args, **kwargs)
    return IAO_Fragmentation_RHF(emb, *args, **kwargs)


def IAOPAO_Fragmentation(emb, *args, **kwargs):
    if emb.is_uhf:
        return IAOPAO_Fragmentation_UHF(emb, *args, **kwargs)
    return IAOPAO_Fragmentation_RHF(emb, *args, **kwargs)


def CAS_Fragmentation(emb, *args, **kwargs):
    if emb.is_uhf:
        return CAS_Fragmentation_UHF(emb, *args, **kwargs)
    return CAS_Fragmentation_RHF(emb, *args, **kwargs)
