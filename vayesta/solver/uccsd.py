
import pyscf
import pyscf.cc

from vayesta.core.util import *

#from .ccsd import CCSD_Solver
from .ccsd2 import CCSD_Solver

class UCCSD_Solver(CCSD_Solver):

    SOLVER_CLS = pyscf.cc.uccsd.UCCSD
    SOLVER_CLS_DF = SOLVER_CLS      # No DF-UCCSD in PySCF

    def get_c2(self):
        """C2 in intermediate normalization."""
        ta, tb = self.t1
        taa, tab, tbb = self.t2
        caa = taa + einsum('ia,jb->ijab', ta, ta) - einsum('ib,ja->ijab', ta, ta)
        cbb = tbb + einsum('ia,jb->ijab', tb, tb) - einsum('ib,ja->ijab', tb, tb)
        cab = tab + einsum('ia,jb->ijab', ta, tb)
        return (caa, cab, cbb)

    def get_c2e(self):
        """C2 used for energy"""
        ta, tb = self.t1
        taa, tab, tbb = self.t2
        caa = taa + 2*einsum('ia,jb->ijab', ta, ta) - 2*einsum('ib,ja->ijab', ta, ta)
        cbb = tbb + 2*einsum('ia,jb->ijab', tb, tb) - 2*einsum('ib,ja->ijab', tb, tb)
        cab = tab +   einsum('ia,jb->ijab', ta, tb)
        return (caa, cab, cbb)

    def t_diagnostic(self):
        """T diagnostic not implemented for UCCSD in PySCF."""
        self.log.info("T diagnostic not implemented for UCCSD in PySCF.")
        return None
