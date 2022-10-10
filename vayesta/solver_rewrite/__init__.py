import numpy as np

from .ebcc import REBCC_Solver, UEBCC_Solver, EB_REBCC_Solver, EB_UEBCC_Solver
from .fci import FCI_Solver, UFCI_Solver
from .ebfci import EB_EBFCI_Solver, EB_UEBFCI_Solver
from vayesta.solver import get_solver_class as get_solver_class_other
from .hamiltonian import *
from .ccsd import RCCSD_Solver, UCCSD_Solver
from .mp2 import RMP2_Solver, UMP2_Solver

def is_uhf(ham):
    return issubclass(type(ham), UClusterHamiltonian)

def is_eb(ham):
    return issubclass(type(ham), EB_RClusterHamiltonian)

def get_solver_class(ham, solver):
    assert(issubclass(type(ham), RClusterHamiltonian))
    solver = solver.upper()
    uhf = is_uhf(ham)
    eb = is_eb(ham)

    if solver == "CCSD":
        if eb:
            # EB solvers only available via EBCC.
            solver = "EBCCSD"
        else:
            # Use pyscf solvers.
            if uhf:
                return UCCSD_Solver
            else:
                return RCCSD_Solver
    if solver[:4] == "EBCC":
        if uhf:
            if eb:
                solverclass = EB_UEBCC_Solver
            else:
                solverclass = UEBCC_Solver
        else:
            if eb:
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
                        raise RuntimeError(
                            "Desired CC rank specified differently in solver specifier and solver_options."
                            "Please use only specify via one approach, or ensure they agree.")
                kwargs["ansatz"] = specrank
                return solverclass(*args, **kwargs)

            return get_right_CC
    if solver == "FCI":
        if uhf:
            if eb:
                return EB_UEBFCI_Solver
            else:
                return UFCI_Solver
        else:
            if eb:
                return EB_EBFCI_Solver
            else:
                return FCI_Solver
    if solver == "MP2":
        if eb:
            raise ValueError("MP2 solver is not implemented for coupled electron-boson systems!")
        if uhf:
            return UMP2_Solver
        else:
            return RMP2_Solver
    raise ValueError("Unknown solver: %s" % solver)
