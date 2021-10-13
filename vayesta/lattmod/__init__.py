"""Lattice model module"""

from .latt import Hubbard1D
from .latt import Hubbard2D
from .latt import LatticeRHF
from .latt import LatticeUHF

def LatticeMF(mol, *args, **kwargs):
    """Use RHF by default, unless spin is not zero."""
    if mol.spin == 0:
        return LatticeRHF(mol, *args, **kwargs)
    return LatticeUHF(mol, *args, **kwargs)
