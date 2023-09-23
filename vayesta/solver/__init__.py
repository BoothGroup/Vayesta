from __future__ import annotations
from typing import *

from vayesta.solver.ccsd import RCCSD_Solver, UCCSD_Solver
from vayesta.solver.cisd import RCISD_Solver, UCISD_Solver
from vayesta.solver.coupled_ccsd import coupledRCCSD_Solver
from vayesta.solver.dump import DumpSolver
from vayesta.solver.ebfci import EB_EBFCI_Solver, EB_UEBFCI_Solver
from vayesta.solver.ext_ccsd import extRCCSD_Solver, extUCCSD_Solver
from vayesta.solver.fci import FCI_Solver, UFCI_Solver
from vayesta.solver.hamiltonian import is_ham, is_uhf_ham, is_eb_ham, ClusterHamiltonian
from vayesta.solver.mp2 import RMP2_Solver, UMP2_Solver
from vayesta.solver.tccsd import TRCCSD_Solver

try:
    from vayesta.solver.ebcc import REBCC_Solver, UEBCC_Solver, EB_REBCC_Solver, EB_UEBCC_Solver
    _has_ebcc = True
except ImportError:
    REBCC_Solver = UEBCC_Solver = EB_REBCC_Solver = EB_UEBCC_Solver = None
    _has_ebcc = False

if TYPE_CHECKING:
    from logging import Logger


def get_solver_class(ham, solver):
    assert is_ham(ham)
    uhf = is_uhf_ham(ham)
    eb = is_eb_ham(ham)
    return _get_solver_class(solver, uhf, eb, ham.log)


def check_solver_config(solver, is_uhf, is_eb, log):
    _get_solver_class(solver, is_uhf, is_eb, log)


def _get_solver_class(solver: str, is_uhf: bool, is_eb: bool, log: Logger) -> Type:
    try:
        solver_cls = _get_solver_class_internal(solver, is_uhf, is_eb, log)
        return solver_cls
    except ValueError as e:
        spinmessage = "unrestricted" if is_uhf else "restricted"
        ebmessage = " with electron-boson coupling" if is_eb else ""
        fullmessage = f"solver '{solver}' not available for {spinmessage} systems{ebmessage}"
        log.critical(fullmessage)
        raise ValueError(fullmessage)


# (solver_string, is_uhf, is_eb) -> SolverClass
_solver_dict: Dict[Tuple[str, bool, bool], Type] = {
    ('MP2', False, False): RMP2_Solver,
    ('MP2', True, False): UMP2_Solver,
    ('CISD', False, False): RCISD_Solver,
    ('CISD', True, False): UCISD_Solver,
    ('CCSD', False, False): RCCSD_Solver,
    ('CCSD', True, False): UCCSD_Solver,
    ('TCCSD', False, False): TRCCSD_Solver,
    ('TCCSD', True, False): NotImplemented,
    ('extCCSD', False, False): extRCCSD_Solver,
    ('extCCSD', True, False): extUCCSD_Solver,
    ('coupledCCSD', False, False): coupledRCCSD_Solver,
    ('coupledCCSD', True, False): NotImplemented,
    ('FCI', False, False): FCI_Solver,
    ('FCI', True, False): UFCI_Solver,
    ('FCI', False, True): EB_EBFCI_Solver,
    ('FCI', True, True): EB_UEBFCI_Solver,
    ('Dump', False, False): DumpSolver,
    ('Dump', True, False): DumpSolver,
}


# (is_uhf, is_eb) -> SolverClass
_ebcc_solver_dict: Dict[Tuple[bool, bool], Type] = {
    (False, False): REBCC_Solver,
    (True, False): UEBCC_Solver,
    (False, True): EB_REBCC_Solver,
    (True, True): EB_UEBCC_Solver,
}


def _get_solver_class_internal(solver: str, is_uhf: bool, is_eb: bool, log: Logger) -> Type | Callable:
    solver_cls = _solver_dict.get((solver, is_uhf, is_eb), None)
    if solver_cls is NotImplemented:
        spinsym = 'unrestricted' if is_uhf else 'restricted'
        raise NotImplementedError(f"solver '{solver}' for {spinsym} spin-symmetry is not implemented")
    if solver_cls is not None:
        return solver_cls
    if 'CC' not in solver:
        raise ValueError(f"unknown solver '{solver}'")
    # Try EBCC next
    return _get_solver_class_ebcc(solver, is_uhf, is_eb, log)


def _get_solver_class_ebcc(solver: str, is_uhf: bool, is_eb: bool, log: Logger) -> Type | Callable:
    if not _has_ebcc:
        raise ImportError(f"{solver} solver is only accessible via ebcc. Please install ebcc.")
    solver_cls = _ebcc_solver_dict[is_uhf, is_eb]
    if solver == "EBCC":
        # Default to `opts.ansatz`.
        return solver_cls
    if solver[:2] == "EB":
        solver = solver[2:]
    if solver == "CCSD" and is_eb:
        log.warning("CCSD solver requested for coupled electron-boson system; defaulting to CCSD-SD-1-1.")
        solver = "CCSD-SD-1-1"

    # This is just a wrapper to allow us to use the solver option as the ansatz kwarg in this case.
    def get_right_cc(*args, **kwargs):
        setansatz = kwargs.get("ansatz", None)
        if setansatz != solver:
            raise ValueError(
                "Desired CC ansatz specified differently in solver and solver_options.ansatz."
                "Please use only specify via one approach, or ensure they agree."
            )
        kwargs["ansatz"] = solver
        return solver_cls(*args, **kwargs)

    return get_right_cc
