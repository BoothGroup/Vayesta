from vayesta.core.scmf.pdmet import PDMET_RHF
from vayesta.core.scmf.pdmet import PDMET_UHF
from vayesta.core.scmf.brueckner import Brueckner_RHF
from vayesta.core.scmf.brueckner import Brueckner_UHF
from vayesta.core.scmf.qpewdmet import QPEWDMET_RHF


def PDMET(emb, *args, **kwargs):
    if emb.is_rhf:
        return PDMET_RHF(emb, *args, **kwargs)
    return PDMET_UHF(emb, *args, **kwargs)


def Brueckner(emb, *args, **kwargs):
    if emb.is_rhf:
        return Brueckner_RHF(emb, *args, **kwargs)
    return Brueckner_UHF(emb, *args, **kwargs)

def QPEWDMET(emb, *args, **kwargs):
    if emb.is_rhf:
        return QPEWDMET_RHF(emb, *args, **kwargs)
    raise NotImplementedError("QP-EWDMET for UHF not implemented")