from vayesta.rpa.rirpa.RIRPA import ssRIRRPA
from vayesta.rpa.rirpa.RIURPA import ssRIURPA
import pyscf.scf


def ssRIRPA(mf, *args, **kwargs):
    if isinstance(mf, pyscf.scf.uhf.UHF):
        return ssRIURPA(mf, *args, **kwargs)
    return ssRIRRPA(mf, *args, **kwargs)
