from vayesta.core.bath.dmet import DMET_Bath_RHF
from vayesta.core.bath.dmet import DMET_Bath_UHF
from vayesta.core.bath.full import Full_Bath_RHF
from vayesta.core.bath.full import Full_Bath_UHF

from vayesta.core.bath.ewdmet import EwDMET_Bath_RHF

from vayesta.core.bath.bno import BNO_Threshold
from vayesta.core.bath.bno import BNO_Bath
from vayesta.core.bath.bno import MP2_BNO_Bath as MP2_Bath_RHF
from vayesta.core.bath.bno import UMP2_BNO_Bath as MP2_Bath_UHF

from vayesta.core.bath.rpa import RPA_BNO_Bath

from vayesta.core.bath.r2bath import R2_Bath_RHF


def DMET_Bath(fragment, *args, **kwargs):
    if fragment.base.is_rhf:
        return DMET_Bath_RHF(fragment, *args, **kwargs)
    if fragment.base.is_uhf:
        return DMET_Bath_UHF(fragment, *args, **kwargs)


def EwDMET_Bath(fragment, *args, **kwargs):
    if fragment.base.is_rhf:
        return EwDMET_Bath_RHF(fragment, *args, **kwargs)
    if fragment.base.is_uhf:
        raise NotImplementedError


def MP2_Bath(fragment, *args, **kwargs):
    if fragment.base.is_rhf:
        return MP2_Bath_RHF(fragment, *args, **kwargs)
    if fragment.base.is_uhf:
        return MP2_Bath_UHF(fragment, *args, **kwargs)


def RPA_Bath(fragment, *args, **kwargs):
    if fragment.base.is_rhf:
        return RPA_BNO_Bath(fragment, *args, **kwargs)
    if fragment.base.is_uhf:
        raise NotImplementedError


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
