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

    def get_coeff(self):
        return self.mo_coeff.copy()

    def get_labels(self):
        return [("", "HF", str(x), "") for x in range(0, self.nmo)]

    def get_atom_indices_symbols(self, *args, **kwargs):
        raise NotImplementedError("Atomic fragmentation is not compatible with CAS fragmentation")

    # Need to overload this function since only accept integer specification in this case.
    def get_orbital_indices_labels(self, orbitals):
        if isinstance(orbitals[0], (int, np.integer)):
            orbital_indices = orbitals
            orbital_labels = (np.asarray(self.labels, dtype=object)[orbitals]).tolist()
            orbital_labels = [('%s%3s %s%-s' % tuple(l)).strip() for l in orbital_labels]
            return orbital_indices, orbital_labels
        raise ValueError("A list of integers is required! orbitals= %r" % orbitals)

class CAS_Fragmentation_UHF(Fragmentation_UHF, CAS_Fragmentation):

    def get_labels(self):
        return [("", "HF", str(x), "") for x in range(0, self.nmo[0])]
