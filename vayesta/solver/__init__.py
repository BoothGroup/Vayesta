import numpy as np

from .mp2 import MP2_Solver
from .mp2 import UMP2_Solver
from .ccsd import CCSD_Solver
from .ccsd import UCCSD_Solver
from .cisd import CISD_Solver
from .cisd import UCISD_Solver
from .fci import FCI_Solver
from .fci import UFCI_Solver
from .ebfci import EBFCI_Solver


def is_uhf(mf):
    return (np.ndim(mf.mo_coeff[0]) == 2)

def get_solver_class(mf, solver):
    solver = solver.upper()
    uhf = is_uhf(mf)
    if solver == 'MP2':
        if uhf:
            return UMP2_Solver
        return MP2_Solver
    if solver in ('CCSD', 'CCSD(T)', 'TCCSD'):
        if uhf:
            return UCCSD_Solver
        return CCSD_Solver
    if solver == 'CISD':
        if uhf:
            return UCISD_Solver
        return CISD_Solver
    if solver == 'FCI':
        if uhf:
            return UFCI_Solver
        return FCI_Solver
    if solver == 'EBFCI':
        if uhf:
            raise NotImplementedError("EBFCI with spin-unrestricted orbitals not implemented!")
        return EBFCI_Solver
    raise ValueError("Unknown solver: %s" % solver)
