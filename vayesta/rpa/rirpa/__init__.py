from vayesta.rpa.rirpa.RIRPA import ssRIRRPA
from vayesta.rpa.rirpa.RIURPA import ssRIURPA
from vayesta.rpa.rirpa.RIdRRPA import ssRIdRRPA
from vayesta.core.mf import RHF_MeanField, UHF_MeanField, read_mf
import pyscf.scf


def ssRIRPA(mf, *args, **kwargs):
    mf = read_mf(mf)
    if isinstance(mf, UHF_MeanField):
        return ssRIURPA(mf, *args, **kwargs)
    if "rixc" in kwargs:
        if kwargs["rixc"] is not None:
            return ssRIRRPA(mf, *args, **kwargs)
    return ssRIdRRPA(mf, *args, **kwargs)
