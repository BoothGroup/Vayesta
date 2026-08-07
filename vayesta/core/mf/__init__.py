import pyscf.scf

from .mf import PySCF_RHF, PySCF_UHF, RHF_MeanField, UHF_MeanField
from .folded_mf import Folded_PySCF_RHF, Folded_PySCF_UHF, Folded_PySCF_MeanField

types = [PySCF_RHF, PySCF_UHF, RHF_MeanField, UHF_MeanField, Folded_PySCF_RHF, Folded_PySCF_UHF, Folded_PySCF_MeanField]

__all__ = ["read_mf"] + types

def read_mf(mf):
    """
    Read mean-field object and convert to Vayesta's internal representation if necessary.
    Currently supported mean-field types are PySCF's RHF, UHF, KRHF, KUHF.

    All k-point sampled mean-field objects are converted to Folded_PySCF_MeanField, 
    which folds the MO orbitals and integralsto the Gamma point of supercell.   
     
    Parameters
    ----------
    mf : object
        Mean-field object to read.

    Returns
    -------
    mf : object
        Vayesta's internal mean-field object.
    """
    if isinstance(mf, tuple(types)):
        return mf
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