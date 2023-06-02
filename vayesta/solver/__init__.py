from .ccsd import RCCSD_Solver, UCCSD_Solver
from .cisd import RCISD_Solver, UCISD_Solver
from .coupled_ccsd import coupledRCCSD_Solver
from .dump import DumpSolver
from .ebcc import REBCC_Solver, UEBCC_Solver, EB_REBCC_Solver, EB_UEBCC_Solver
from .ebfci import EB_EBFCI_Solver, EB_UEBFCI_Solver
from .ext_ccsd import extRCCSD_Solver, extUCCSD_Solver
from .fci import FCI_Solver, UFCI_Solver
from .hamiltonian import is_ham, is_uhf_ham, is_eb_ham, ClusterHamiltonian
from .mp2 import RMP2_Solver, UMP2_Solver
from .tccsd import TRCCSD_Solver


def get_solver_class(ham, solver):
    assert (is_ham(ham))
    uhf = is_uhf_ham(ham)
    eb = is_eb_ham(ham)
    return _get_solver_class(uhf, eb, solver, ham.log)


def check_solver_config(is_uhf, is_eb, solver, log):
    _get_solver_class(is_uhf, is_eb, solver, log)


def _get_solver_class(is_uhf, is_eb, solver, log):
    try:
        solver_cls = _get_solver_class_internal(is_uhf, is_eb, solver)
        return solver_cls
    except ValueError as e:
        spinmessage = "unrestricted" if is_uhf else "restricted"
        bosmessage = "coupled electron-boson" if is_eb else "purely electronic"

        fullmessage = f"Error; solver {solver} not available for {spinmessage} {bosmessage} systems"
        log.critical(fullmessage)
        raise ValueError(fullmessage)


def _get_solver_class_internal(is_uhf, is_eb, solver):
    solver = solver.upper()
    # First check if we have a CC approach as implemented in pyscf.
    if solver == "CCSD" and not is_eb:
        # Use pyscf solvers.
        if is_uhf:
            return UCCSD_Solver
        else:
            return RCCSD_Solver
    if solver == "TCCSD":
        if is_uhf or is_eb:
            raise ValueError("TCCSD is not implemented for unrestricted or electron-boson calculations!")
        return TRCCSD_Solver
    if solver == "EXTCCSD":
        if is_eb:
            raise ValueError("extCCSD is not implemented for electron-boson calculations!")
        if is_uhf:
            return extUCCSD_Solver
        return extRCCSD_Solver
    if solver == "COUPLEDCCSD":
        if is_eb:
            raise ValueError("coupledCCSD is not implemented for electron-boson calculations!")
        if is_uhf:
            raise ValueError("coupledCCSD is not implemented for unrestricted calculations!")
        return coupledRCCSD_Solver

    # Now consider general CC ansatzes; these are solved via EBCC.
    if "CC" in solver:
        if is_uhf:
            if is_eb:
                solverclass = EB_UEBCC_Solver
            else:
                solverclass = UEBCC_Solver
        else:
            if is_eb:
                solverclass = EB_REBCC_Solver
            else:
                solverclass = REBCC_Solver
        if solver == "EBCC":
            # Default to `opts.ansatz`.
            return solverclass
        if solver[:2] == "EB":
            solver = solver[2:]
        if solver == "CCSD" and is_eb:
            # Need to specify CC level for coupled electron-boson model; throw an error rather than assume.
            raise ValueError(
                "Please specify a coupled electron-boson CC ansatz as a solver, for example CCSD-S-1-1,"
                "rather than CCSD")

        def get_right_CC(*args, **kwargs):

            if kwargs.get("ansatz", None) is not None:
                raise ValueError(
                    "Desired CC ansatz specified differently in solver and solver_options.ansatz."
                    "Please use only specify via one approach, or ensure they agree.")
            kwargs["ansatz"] = solver
            return solverclass(*args, **kwargs)

        return get_right_CC
    if solver == "FCI":
        if is_uhf:
            if is_eb:
                return EB_UEBFCI_Solver
            else:
                return UFCI_Solver
        else:
            if is_eb:
                return EB_EBFCI_Solver
            else:
                return FCI_Solver
    if is_eb:
        raise ValueError("%s solver is not implemented for coupled electron-boson systems!", solver)
    if solver == "MP2":
        if is_uhf:
            return UMP2_Solver
        else:
            return RMP2_Solver
    if solver == "CISD":
        if is_uhf:
            return UCISD_Solver
        else:
            return RCISD_Solver
    if solver == "DUMP":
        return DumpSolver
    raise ValueError("Unknown solver: %s" % solver)
