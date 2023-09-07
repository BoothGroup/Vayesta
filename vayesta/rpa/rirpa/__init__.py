from vayesta.rpa.rirpa.RIRPA import ssRIRRPA
from vayesta.rpa.rirpa.RIURPA import ssRIURPA
from vayesta.rpa.rirpa.RIdRRPA import ssRIdRRPA
import pyscf.scf


def ssRIRPA(mf, *args, **kwargs):
    if isinstance(mf, pyscf.scf.uhf.UHF):
        return ssRIURPA(mf, *args, **kwargs)
    if "rixc" in kwargs:
        if kwargs["rixc"] is not None:
            return ssRIRRPA(mf, *args, **kwargs)
    return ssRIdRRPA(mf, *args, **kwargs)
