from .ebcc import REBCC_Solver, UEBCC_Solver, EB_REBCC_Solver, EB_UEBCC_Solver
from .fci import FCI_Solver, UFCI_Solver
from .ebfci import EB_EBFCI_Solver, EB_UEBFCI_Solver
from .hamiltonian import is_ham, is_uhf_ham, is_eb_ham, ClusterHamiltonian
from .ccsd import RCCSD_Solver, UCCSD_Solver
from .mp2 import RMP2_Solver, UMP2_Solver
from .cisd import RCISD_Solver, UCISD_Solver
from .tccsd import TRCCSD_Solver

def get_solver_class(ham, solver):
    assert(is_ham(ham))
    uhf = is_uhf_ham(ham)
    eb = is_eb_ham(ham)
    return _get_solver_class(uhf, eb, solver)

def check_solver_config(is_uhf, is_eb, solver, log):
    try:
        solver = _get_solver_class(is_uhf, is_eb, solver)
    except ValueError as e:
        spinmessage = "unrestricted" if is_uhf else "restricted"
        bosmessage = "coupled electron-boson" if is_eb else "purely electronic"

        log.critical("Error; solver %s not available for %s %s systems", solver, spinmessage, bosmessage)
        raise e


def _get_solver_class(is_uhf, is_eb, solver):
    solver = solver.upper()

    if solver == "CCSD":
        if is_eb:
            # EB solvers only available via EBCC.
            solver = "EBCCSD"
        else:
            # Use pyscf solvers.
            if is_uhf:
                return UCCSD_Solver
            else:
                return RCCSD_Solver
    if solver[:4] == "EBCC":
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
        if len(solver) == 4:
            return solverclass
        else:
            specrank = solver[2:]

            def get_right_CC(*args, **kwargs):
                if "ansatz" in kwargs:
                    if specrank != kwargs["ansatz"]:
                        raise ValueError(
                            "Desired CC rank specified differently in solver specifier and solver_options."
                            "Please use only specify via one approach, or ensure they agree.")
                kwargs["ansatz"] = specrank
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
    if solver == "TCCSD":
        if is_uhf:
            raise ValueError("TCCSD is not implemented for unrestricted calculations!")
        return TRCCSD_Solver
    raise ValueError("Unknown solver: %s" % solver)
