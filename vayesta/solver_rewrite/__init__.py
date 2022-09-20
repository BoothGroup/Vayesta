import numpy as np

from .ebcc import REBCC_Solver, UEBCC_Solver, EB_REBCC_Solver, EB_UEBCC_Solver
from .fci import FCI_Solver, UFCI_Solver
from .ebfci import EB_EBFCI_Solver, EB_UEBFCI_Solver
from vayesta.solver import get_solver_class as get_solver_class_other
from .hamiltonian import *


def is_uhf(ham):
    return issubclass(type(ham), UClusterHamiltonian)

def is_eb(ham):
    return issubclass(type(ham), EB_RClusterHamiltonian)

def get_solver_class(ham, solver):
    assert(issubclass(type(ham), RClusterHamiltonian))
    solver = solver.upper()
    uhf = is_uhf(ham)
    eb = is_eb(ham)

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
            specrank = solver[4:]

            def get_right_CC(*args, **kwargs):
                if "fermion_excitations" in kwargs:
                    if specrank != kwargs["fermion_excitations"]:
                        raise RuntimeError(
                            "Desired CC rank specified differently in solver specifier and solver_options."
                            "Please use only specify via one approach, or ensure they agree.")
                kwargs["fermion_excitations"] = specrank
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
    return get_solver_class_other(ham.mf, solver)
    # raise ValueError("Unknown solver: %s" % solver)
