import pyscf
import pyscf.ci

from vayesta.core.util import *
from .ccsd2 import CCSD_Solver


class CISD_Solver(CCSD_Solver):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Results
        self.civec = None
        self.c0 = None
        self.c1 = None      # In intermediate normalization!
        self.c2 = None      # In intermediate normalization!

    def kernel(self, eris=None):

        # Integral transformation
        if eris is None: eris = self.get_eris()

        # Add additional potential
        if self.opts.v_ext is not None:
            self.log.debugv("Adding self.opts.v_ext to eris.fock")
            # Make sure there are no side effects:
            eris = copy.copy(eris)
            # Replace fock instead of modifying it!
            eris.fock = (eris.fock + self.opts.v_ext)
        #self.log.debugv("sum(eris.mo_energy)= %.8e", sum(eris.mo_energy))
        #self.log.debugv("Tr(eris.fock)= %.8e", np.trace(eris.fock))

        # Tailored CC
        with log_time(self.log.timing, "Time for CISD: %s"):
            #self.log.info("Solving CISD-equations %s initial guess...", "with" if (t2 is not None) else "without")
            self.log.info("Solving CISD-equations")
            self.solver.kernel(eris=eris)
            if not self.solver.converged:
                self.log.error("%s not converged!", self.__class__.__name__)
            else:
                self.log.debugv("%s converged.", self.__class__.__name__)
            self.e_corr = self.solver.e_corr
            self.converged = self.solver.converged
            self.log.debug("Cluster: E(corr)= % 16.8f Ha", self.solver.e_corr)

        self.civec = self.solver.ci
        c0, c1, c2 = self.solver.cisdvec_to_amplitudes(self.civec)
        self.c0 = c0
        self.c1 = c1/c0
        self.c2 = c2/c0

    def get_solver_class(self):
        # No DF version for CISD
        return pyscf.ci.cisd.CISD

    def get_c1(self):
        return self.c1

    def get_c2(self):
        return self.c2

    def get_init_guess(self):
        return {'c1' : self.c1 , 't2' : self.c2}

    def make_rdm1(self, *args, **kwargs):
        raise NotImplementedError()

    def make_rdm2(self, *args, **kwargs):
        raise NotImplementedError()

class UCISD_Solver(CISD_Solver):

    def get_solver_class(self):
        # No DF version for UCISD
        return pyscf.ci.ucisd.UCISD
