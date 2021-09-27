
import pyscf
import pyscf.cc

from .ccsd import CCSD_Solver

class UCCSD_Solver(CCSD_Solver):

    SOLVER_CLS = pyscf.cc.uccsd.UCCSD
    SOLVER_CLS_DF = SOLVER_CLS      # No DF-UCCSD in PySCF
