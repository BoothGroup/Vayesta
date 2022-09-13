from .ebfci import REBFCI, UEBFCI
import numpy as np


def EBFCI(mf, *args, **kwargs):
    def is_uhf(mf):
        return (np.ndim(mf.mo_coeff[0]) == 2)
    uhf = is_uhf(mf)
    if uhf:
        return UEBFCI(mf, *args, **kwargs)
    return REBFCI(mf, *args, **kwargs)
