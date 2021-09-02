
import dataclasses

import numpy as np
from timeit import default_timer as timer
import copy

from vayesta.dmet import DMET


@dataclasses.dataclass
class EDMETResults:
    cluster_sizes: np.ndarray = None
    e_corr: float = None

VALID_SOLVERS = [None, "", "MP2", "CISD", "CCSD", 'TCCSD', "CCSD(T)", 'FCI', "FCI-spin0", "FCI-spin1"]

class EDMET(DMET):


    def kernel(self):
        # Just a single-shot application initially. Sadly will still need chemical potential...
