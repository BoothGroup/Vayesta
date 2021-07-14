def get_solver_class(solver):
    raise NotImplementedError
    if solver.upper() in ('CCSD', 'CCSD(T)', 'TCCSD'):
        return CCSDSolver
    if solver.upper() == 'FCI':
        return FCISolver
    raise NotImplementedError("Unknown solver %s" % solver)
