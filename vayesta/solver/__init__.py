import numpy as np

from vayesta.solver.mp2 import MP2_Solver
from vayesta.solver.mp2 import UMP2_Solver
from vayesta.solver.ccsd import CCSD_Solver
from vayesta.solver.ccsd import UCCSD_Solver
from vayesta.solver.cisd import CISD_Solver
from vayesta.solver.cisd import UCISD_Solver
from vayesta.solver.fci import FCI_Solver
from vayesta.solver.fci import UFCI_Solver
from vayesta.solver.ebfci import EBFCI_Solver
from vayesta.solver.ebfci import UEBFCI_Solver
from vayesta.solver.ebccsd import EBCCSD_Solver
from vayesta.solver.ebccsd import UEBCCSD_Solver
from vayesta.solver.dump import DumpSolver


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
            return UEBFCI_Solver
        return EBFCI_Solver
    if solver == 'EBCCSD':
        if uhf:
            return UEBCCSD_Solver
        return EBCCSD_Solver
    if solver == 'DUMP':
        return DumpSolver
    raise ValueError("Unknown solver: %s" % solver)
