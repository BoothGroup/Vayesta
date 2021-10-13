from .scmf import PDMET_RHF, PDMET_UHF
from .scmf import Brueckner_RHF, Brueckner_UHF

def PDMET(emb, *args, **kwargs):
    if emb.is_rhf:
        return PDMET_RHF(emb, *args, **kwargs)
    return PDMET_UHF(emb, *args, **kwargs)

def Brueckner(emb, *args, **kwargs):
    if emb.is_rhf:
        return Brueckner_RHF(emb, *args, **kwargs)
    return Brueckner_UHF(emb, *args, **kwargs)
