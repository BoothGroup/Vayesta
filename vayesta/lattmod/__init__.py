"""Lattice model module"""

from vayesta.lattmod.latt import Hubbard1D
from vayesta.lattmod.latt import Hubbard2D
from vayesta.lattmod.latt import LatticeRHF
from vayesta.lattmod.latt import LatticeUHF
from vayesta.lattmod.latt_pbc import LatticeKRHF


def LatticeMF(mol, *args, **kwargs):
    """Use RHF by default, unless spin is not zero."""
    if mol.spin == 0:
        return LatticeRHF(mol, *args, **kwargs)
    return LatticeUHF(mol, *args, **kwargs)

def LatticeKMF(mol, kpts, *args, **kwargs):
    """Use KRHF by default, unless spin is not zero."""
    if mol.spin == 0:
        return LatticeKRHF(mol, kpts, *args, **kwargs)
    raise NotImplementedError("KUHF not implemented yet.")
    return LatticeUHF(mol, *args, **kwargs)
