import os.path
import numpy as np

import pyscf
import pyscf.lo

from vayesta.core.util import *
from .fragmentation import Fragmentation
from .ufragmentation import Fragmentation_UHF


class CAS_Fragmentation(Fragmentation):
    """Fragmentation into mean-field states."""

    name = "CAS"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Don't need to do much setup here..

    def add_orbital_fragment(self):

    @property
    def nocc(self):
        return sum(self.mo_occ>0)

    @property
    def nvir(self):
        return self.nmo - self.nocc

    def get_atom_indices_symbols(self, *args, **kwargs):
        raise NotImplementedError("Atomic fragmentation is not compatible with CAS fragmentation")

    def get_orbital_fragment_indices(self, orbitals, atom_filter=None, name=None):

        pass

class CAS_Fragmentation_UHF(Fragmentation_UHF):

