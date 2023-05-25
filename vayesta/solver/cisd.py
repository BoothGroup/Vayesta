import copy
import pyscf
import pyscf.ci
from vayesta.core.util import deprecated, einsum, log_time
from vayesta.core.types import WaveFunction
import vayesta.solver.ccsd as ccsd
from vayesta.solver.solver import ClusterSolver


class CISD_Solver(ccsd.CCSD_Solver):

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
        if self.v_ext is not None:
            self.log.debugv("Adding self.v_ext to eris.fock")
            # Make sure there are no side effects:
            eris = copy.copy(eris)
            # Replace fock instead of modifying it!
            eris.fock = (eris.fock + self.v_ext)
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
        self.c0, self.c1, self.c2 = self.solver.cisdvec_to_amplitudes(self.civec)

        self.wf = WaveFunction.from_pyscf(self.solver)

    def get_solver_class(self):
        # No DF version for CISD
        return pyscf.ci.cisd.CISD

    @deprecated()
    def get_t1(self):
        return self.get_c1(intermed_norm=True)

    @deprecated()
    def get_t2(self):
        return (self.c2 - einsum('ia,jb->ijab', self.c1, self.c1))/self.c0

    @deprecated()
    def get_c1(self, intermed_norm=False):
        norm = 1/self.c0 if intermed_norm else 1
        return norm*self.c1

    @deprecated()
    def get_c2(self, intermed_norm=False):
        norm = 1/self.c0 if intermed_norm else 1
        return norm*self.c2

    @deprecated()
    def get_l1(self, **kwargs):
        return None

    @deprecated()
    def get_l2(self, **kwargs):
        return None

    def get_init_guess(self):
        return {'c0' : self.c0, 'c1' : self.c1 , 'c2' : self.c2}

    def make_rdm1(self, *args, **kwargs):
        raise NotImplementedError()

    def make_rdm2(self, *args, **kwargs):
        raise NotImplementedError()


class UCISD_Solver(CISD_Solver):

    def get_solver_class(self):
        # No DF version for UCISD
        return pyscf.ci.ucisd.UCISD

    @deprecated()
    def get_t2(self):
        ca, cb = self.get_c1(intermed_norm=True)
        caa, cab, cbb = self.get_c2(intermed_norm=True)
        taa = caa - einsum('ia,jb->ijab', ca, ca) + einsum('ib,ja->ijab', ca, ca)
        tbb = cbb - einsum('ia,jb->ijab', cb, cb) + einsum('ib,ja->ijab', cb, cb)
        tab = cab - einsum('ia,jb->ijab', ca, cb)
        return (taa, tab, tbb)

    @deprecated()
    def get_c1(self, intermed_norm=False):
        norm = 1/self.c0 if intermed_norm else 1
        return (norm*self.c1[0], norm*self.c1[1])

    @deprecated()
    def get_c2(self, intermed_norm=False):
        norm = 1/self.c0 if intermed_norm else 1
        return (norm*self.c2[0], norm*self.c2[1], norm*self.c2[2])
