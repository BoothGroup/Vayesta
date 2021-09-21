from .solver_cc import CCSDSolver
from .solver_fci import FCISolver
from .solver_ebfci import EBFCISolver

def get_solver_class(solver):
    solver = solver.upper()
    if solver in ('CCSD', 'CCSD(T)', 'TCCSD'):
        return CCSDSolver
    if solver == 'FCI':
        return FCISolver
    if solver == 'EBFCI':
        return EBFCISolver
    raise ValueError("Unknown solver: %s" % solver)
