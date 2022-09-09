import numpy as np

from .ebcc import EBCC_Solver, UEBCC_Solver, EB_EBCC_Solver, UEB_EBCC_Solver


def is_uhf(mf):
    return (np.ndim(mf.mo_coeff[0]) == 2)

def get_solver_class(mf, solver):
    solver = solver.upper()
    uhf = is_uhf(mf)

    if solver[:4] == "EBCC":
        if uhf:
            solverclass = UEBCC_Solver
        else:
            solverclass = EBCC_Solver
        if len(solver) == 4:
            return solverclass
        else:
            specrank = solver[4:]
            def get_right_CC(*args, **kwargs):
                print("!!")
                print(kwargs)
                print(specrank)
                if "rank" in kwargs:
                    if specrank != kwargs["rank"]:
                        raise RuntimeError(
                            "Desired CC rank specified differently in solver specifier and solver_options."
                            "Please use only specify via one approach, or ensure they agree.")
                kwargs["rank"] = specrank
                return solverclass(*args, **kwargs)
            return get_right_CC
    raise ValueError("Unknown solver: %s" % solver)

def get_eb_solver_class(mf, solver):
    solver = solver.upper()
    uhf = is_uhf(mf)
    if solver == "EBCC":
        if uhf:
            solverclass = UEBCC_Solver
        else:
            solverclass = EBCC_Solver
        if len(solver) == 4:
            return solverclass
        else:
            specrank = solver[4:].split("-")
            if len(specrank) != 3:
                raise ValueError("Unknown electron-boson CC solver requested; please specify in the form EBCCSDn-m-l,"
                                 " or via solver options.")

            def get_right_CC(*args, **kwargs):

                def update_opts(inp, key, val):
                    if key in inp:
                        if val != inp[key]:
                            raise RuntimeError(
                                "Desired CC rank specified differently in solver specifier and solver_options."
                                "Please use only specify via one approach, or ensure they agree.")
                    inp[key] = val

                update_opts(kwargs, "fermion_excitations", specrank[0])
                update_opts(kwargs, "boson_excitations", "SD"[:int(specrank[1])+1])
                update_opts(kwargs, "fermion_coupling_rank", int(specrank[2]))
                update_opts(kwargs, "boson_coupling_rank", int(specrank[2]))
                return solverclass(*args, **kwargs)
            return get_right_CC
    raise ValueError("Unknown solver: %s" % solver)
