def get_solver_class(solver):
    if solver.upper() in ('CCSD', 'CCSD(T)', 'TCCSD'):
        from .solver_cc import CCSDSolver
        return CCSDSolver
    if solver.upper() == 'FCI':
        from .solver_fci import FCISolver
        return FCISolver
    if solver.upper() == 'EBFCI':
        from .solver_ebfci import EBFCISolver
        return EBFCISolver
    if solver.upper() == 'FCIQMC':
        from .solver_qmc import FCIQMCSolver
        return FCIQMCSolver
    if solver.upper() == 'EBFCIQMC':
        from .solver_ebqmc import EBFCIQMCSolver
        return EBFCIQMCSolver
    raise NotImplementedError("Unknown solver %s" % solver)
