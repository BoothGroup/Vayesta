import dataclasses
import numpy as np
from .solver import ClusterSolver, UClusterSolver, EBClusterSolver, UEBClusterSolver
from vayesta.core.types import Orbitals
from vayesta.core.types import CCSD_WaveFunction, WaveFunction


class EBCC_Solver(ClusterSolver):
    @dataclasses.dataclass
    class Options(ClusterSolver.Options):
        fermion_excitations: str = "SD"
        t_as_lambda: bool = False       # If true, use Lambda=T approximation


    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            import ebcc
        except ImportError as e:
            raise ImportError("Cannot import ebcc; required to use as a solver.")

    def kernel_solver(self, mf_clus, eris_energy=None):
        import ebcc
        mycc = ebcc.EBCC(mf_clus, log=self.log, fermion_excitations=self.opts.fermion_excitations)
        mycc.kernel()
        if not self.opts.t_as_lambda:
            mycc.solve_lambda()
        # Now just need to wrangle EBCC results into wavefunction format.
        mo = Orbitals(self.cluster.c_active, occ=self.cluster.nocc_active)
        self.construct_wavefunction(mycc, mo)

    def construct_wavefunction(self, mycc, mo):
        # Simply alias required quantities for now; this ensures functionality for arbitrary orders of CC.
        self.wf = WaveFunction(mo)
        self.wf.make_rdm1 = mycc.make_rdm1_f
        self.wf.make_rdm2 = mycc.make_rdm2_f


class UEBCC_Solver(UClusterSolver, EBCC_Solver):
    # This should automatically work other than ensuring spin components are in a tuple.
    def construct_wavefunction(self, mycc, mo):
        # Simply alias required quantities for now; this ensures functionality for arbitrary orders of CC.
        self.wf = WaveFunction(mo)

        def make_rdm1(*args, **kwargs):
            dm = mycc.make_rdm1_f(*args, **kwargs)
            return (dm.aa, dm.bb)

        self.wf.make_rdm1 = make_rdm1

        def make_rdm2(*args, **kwargs):
            dm = mycc.make_rdm2_f(*args, **kwargs)
            return (dm.aaaa, dm.aabb, dm.bbbb)

        self.wf.make_rdm2 = make_rdm2


class EB_EBCC_Solver(EBClusterSolver, EBCC_Solver):
    class Options(EBCC_Solver.Options):
        boson_excitations: str = "S"
        fermion_coupling_rank: int = 1
        boson_coupling_rank: int = 1

    def kernel_solver(self, mf_clus, freqs, couplings):
        import ebcc
        mycc = ebcc.EBCC(mf_clus, log=self.log, fermion_excitations=self.opts.fermion_excitations,
                         boson_excitations=self.opts.boson_excitations,
                         fermion_coupling_rank=self.opts.fermion_coupling_rank,
                         boson_coupling_rank=self.opts.boson_coupling_rank,
                         omega=freqs, g=couplings)
        mycc.kernel()
        mycc.solve_lambda()
        # Now just need to wrangle EBCC results into wavefunction format.
        mo = Orbitals(self.cluster.c_active, occ=self.cluster.nocc_active)
        # Currently no electron-boson implementation of wavefunction approaches, so need to use dummy object.

        self.wf = WaveFunction(mo)
        self.wf.make_rdm1 = mycc.make

        def make_rdm1(cc, t_as_lambda=False, with_mf=True, ao_basis=False):
            cc.make_rdm1_f()

        self.wf.make_rdm1 = make_rdm1


class UEB_EBCC_Solver(UEBClusterSolver, UEBCC_Solver):
    pass
