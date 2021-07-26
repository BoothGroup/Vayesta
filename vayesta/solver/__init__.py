def get_solver_class(solver):
    if solver.upper() in ('CCSD', 'CCSD(T)', 'TCCSD'):
        from .solver_cc import CCSDSolver
        return CCSDSolver
    if solver.upper() == 'FCI':
        from .solver_fci import FCISolver
        return FCISolver
    raise NotImplementedError("Unknown solver %s" % solver)
