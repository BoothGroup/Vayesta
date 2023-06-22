"""Lattice model module"""

from vayesta.lattmod.latt import Hubbard1D
from vayesta.lattmod.latt import Hubbard2D
from vayesta.lattmod.latt import LatticeRHF
from vayesta.lattmod.latt import LatticeUHF

def LatticeMF(mol, *args, **kwargs):
    """Use RHF by default, unless spin is not zero."""
    if mol.spin == 0:
        return LatticeRHF(mol, *args, **kwargs)
    return LatticeUHF(mol, *args, **kwargs)
