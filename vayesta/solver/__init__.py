import numpy as np

from .ccsd import CCSD_Solver
from .uccsd import UCCSD_Solver
from .fci import FCI_Solver
from .ebfci import EBFCI_Solver

def is_uhf(mf):
    return (np.ndim(mf.mo_coeff[0]) == 2)

def get_solver_class(mf, solver):
    solver = solver.upper()
    uhf = is_uhf(mf)
    if solver in ('CCSD', 'CCSD(T)', 'TCCSD'):
        if uhf:
            return UCCSD_Solver
        return CCSD_Solver
    if solver == 'FCI':
        if uhf:
            raise NotImplementedError("FCI with spin-unrestricted orbitals not implemented!")
        return FCI_Solver
    if solver == 'EBFCI':
        if uhf:
            raise NotImplementedError("EBFCI with spin-unrestricted orbitals not implemented!")
        return EBFCI_Solver
    raise ValueError("Unknown solver: %s" % solver)
