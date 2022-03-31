import numpy as np

from .ccsd import CCSD_Solver
from .fci import FCI_Solver
from .ebfci import EBFCI_Solver

# New solver interface
from .ccsd2 import CCSD_Solver as CCSD_Solver2
from .ccsd2 import UCCSD_Solver
from .fci2 import FCI_Solver as FCI_Solver2
from .fci2 import UFCI_Solver
from .cisd import CISD_Solver
from .cisd import UCISD_Solver
from .ebfci2 import EBFCI_Solver as EBFCI_Solver2
from .ebfci2 import UEBFCI_Solver
from .ebccsd import EBCCSD_Solver, UEBCCSD_Solver

def is_uhf(mf):
    return (np.ndim(mf.mo_coeff[0]) == 2)

def get_solver_class(mf, solver):
    solver = solver.upper()
    uhf = is_uhf(mf)
    if solver in ('CCSD', 'CCSD(T)', 'TCCSD'):
        if uhf:
            raise NotImplementedError("CCSD with spin-unrestricted orbitals not implemented!")
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

def get_solver_class2(mf, solver):
    solver = solver.upper()
    uhf = is_uhf(mf)
    if solver in ('CCSD', 'CCSD(T)', 'TCCSD'):
        if uhf:
            return UCCSD_Solver
        return CCSD_Solver2
    if solver in ('CISD'):
        if uhf:
            return UCISD_Solver
        return CISD_Solver
    if solver == 'FCI':
        if uhf:
            return UFCI_Solver
        return FCI_Solver2
    if solver.upper() == 'FCIQMC':
        from .fciqmc import FCIQMCSolver, UFCIQMCSolver
        if uhf:
            return UFCIQMCSolver
        return FCIQMCSolver
    if solver == 'EBFCI':
        if uhf:
            return UEBFCI_Solver
        return EBFCI_Solver2
    if solver.upper() == 'EBFCIQMC':
        from .ebfciqmc import EBFCIQMCSolver, UEBFCIQMCSolver
        if uhf:
            return UEBFCIQMCSolver
        return EBFCIQMCSolver
    if solver == 'EBCCSD':
        if uhf:
            return UEBCCSD_Solver
        return EBCCSD_Solver
    raise ValueError("Unknown solver: %s" % solver)
