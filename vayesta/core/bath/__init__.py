from .dmet import DMET_Bath_RHF
from .dmet import DMET_Bath_UHF
from .full import Full_Bath_RHF
from .full import Full_Bath_UHF

from .ewdmet import EwDMET_Bath

from .bno import BNO_Threshold
from .bno import BNO_Bath
from .bno import MP2_BNO_Bath as MP2_Bath_RHF
from .bno import UMP2_BNO_Bath as MP2_Bath_UHF

from .r2bath import R2_Bath_RHF

def DMET_Bath(fragment, *args, **kwargs):
    if fragment.base.is_rhf:
        return DMET_Bath_RHF(fragment, *args, **kwargs)
    if fragment.base.is_uhf:
        return DMET_Bath_UHF(fragment, *args, **kwargs)

def MP2_Bath(fragment, *args, **kwargs):
    if fragment.base.is_rhf:
        return MP2_Bath_RHF(fragment, *args, **kwargs)
    if fragment.base.is_uhf:
        return MP2_Bath_UHF(fragment, *args, **kwargs)

def R2_Bath(fragment, *args, **kwargs):
    if fragment.base.is_rhf:
        return R2_Bath_RHF(fragment, *args, **kwargs)
    if fragment.base.is_uhf:
        raise NotImplementedError

def Full_Bath(fragment, *args, **kwargs):
    if fragment.base.is_rhf:
        return Full_Bath_RHF(fragment, *args, **kwargs)
    if fragment.base.is_uhf:
        return Full_Bath_UHF(fragment, *args, **kwargs)
