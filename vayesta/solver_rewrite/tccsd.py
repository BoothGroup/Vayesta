from .ccsd import RCCSD_Solver
from .fci import FCI_Solver
import numpy as np
import dataclasses


class TRCCSD_Solver(RCCSD_Solver):
    @dataclasses.dataclass
    class Options(RCCSD_Solver.Options):
        tcc_fci_opts: dict = dataclasses.field(default_factory=dict)
        c_cas_occ: np.array = None
        c_cas_vir: np.array = None


    def kernel(self, t1=None, t2=None, l1=None, l2=None, coupled_fragments=None, t_diagnostic=True):




        # Now usual setup for full CCSD calculation.
        mf_clus, frozen = self.hamil.to_pyscf_mf(allow_dummy_orbs=True, allow_df=True)
        solver_cls = self.get_solver_class(mf_clus)
        self.log.debugv("PySCF solver class= %r" % solver_cls)
        mycc = solver_cls(mf_clus, frozen=frozen)

    def