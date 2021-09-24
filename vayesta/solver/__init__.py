from .ccsd import CCSDSolver
from .fci import FCISolver
from .ebfci import EBFCISolver

def get_solver_class(mf, solver):
    solver = solver.upper()
    if solver in ('CCSD', 'CCSD(T)', 'TCCSD'):
        return CCSDSolver
    if solver == 'FCI':
        return FCISolver
    if solver == 'EBFCI':
        return EBFCISolver
    raise ValueError("Unknown solver: %s" % solver)
