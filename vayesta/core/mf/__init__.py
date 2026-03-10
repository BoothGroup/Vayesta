import pyscf.scf

from .mf import PySCF_RHF, PySCF_UHF, RHF_MeanField, UHF_MeanField
from .folded_mf import Folded_PySCF_RHF, Folded_PySCF_UHF, Folded_PySCF_MeanField

def read_mf(mf):

    if isinstance(mf, pyscf.pbc.scf.krhf.KRHF):
        return Folded_PySCF_RHF(mf)
    elif isinstance(mf, pyscf.pbc.scf.kuhf.KUHF):
        return Folded_PySCF_UHF(mf)
    elif isinstance(mf, (pyscf.scf.rhf.RHF, pyscf.pbc.scf.rhf.RHF)):
        return PySCF_RHF(mf)
    elif isinstance(mf, (pyscf.scf.uhf.UHF, pyscf.pbc.scf.uhf.UHF)):
        return PySCF_UHF(mf)
    else:
        raise TypeError(f"Unsupported mean-field type: {type(mf)}")