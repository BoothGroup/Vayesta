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

    @property
    def nocc(self):
        return sum(self.mo_occ>0)

    @property
    def nvir(self):
        return self.nmo - self.nocc

    def get_coeff(self):
        return self.mo_coeff.copy()

    def get_labels(self):
        return [str(x) for x in range(0, self.nmo)]

    def get_atom_indices_symbols(self, *args, **kwargs):
        raise NotImplementedError("Atomic fragmentation is not compatible with CAS fragmentation")

    def get_orbital_fragment_indices(self, orbitals, atom_filter=None, name=None):

        if atom_filter is not None:
            raise ValueError("CAS fragmentation incompatible with atom_filter option.")

        indices, orbital_labels = orbitals, [str(x) for x in orbitals]
        if name is None: name = '/'.join(orbital_labels)
        self.log.debugv("Orbital indices of fragment %s: %r", name, indices)
        self.log.debugv("Orbital labels of fragment %s: %r", name, orbital_labels)
        return name, indices

class CAS_Fragmentation_UHF(Fragmentation_UHF):
    pass
