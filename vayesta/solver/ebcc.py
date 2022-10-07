import dataclasses

import numpy as np

from vayesta.core.types import WaveFunction, CCSD_WaveFunction
from vayesta.core.util import dot, einsum
from vayesta.solver.solver import ClusterSolver, UClusterSolver


class REBCC_Solver(ClusterSolver):
    @dataclasses.dataclass
    class Options(ClusterSolver.Options):
        ansatz: str = "CCSD"
        solve_lambda: bool = False  # If false, use Lambda=T approximation
        # Convergence
        max_cycle: int = 200  # Max number of iterations
        conv_tol: float = None  # Convergence energy tolerance
        conv_tol_normt: float = None  # Convergence amplitude tolerance

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            import ebcc
        except ImportError as e:
            raise ImportError("Cannot import ebcc; required to use as a solver.")

    @property
    def is_fCCSD(self):
        return self.opts.ansatz == "CCSD"

    def kernel(self):
        import ebcc
        # Use pyscf mean-field representation.
        mf_clus, frozen = self.hamil.to_pyscf_mf(allow_dummy_orbs=False)
        mycc = ebcc.EBCC(mf_clus, log=self.log, ansatz=self.opts.ansatz,
                         **self.get_nonnull_solver_opts())
        mycc.kernel()
        self.converged = mycc.converged
        if self.opts.solve_lambda:
            mycc.solve_lambda()
            self.converged = self.converged and mycc.converged_lambda
        # Now just need to wrangle EBCC results into wavefunction format.
        self.construct_wavefunction(mycc, self.hamil.mo)

    def get_nonnull_solver_opts(self):
        def add_nonull_opt(d, key, newkey):
            val = self.opts.get(key)
            if val is not None:
                d[newkey] = val

        opts = {}
        for key, newkey in zip(["max_cycle", "conv_tol", "conv_tol_normt"], ["max_cycle", "e_tol", "t_tol"]):
            add_nonull_opt(opts, key, newkey)
        return opts

    def construct_wavefunction(self, mycc, mo):
        if self.is_fCCSD:
            # Can use existing functionality
            try:
                self.wf = CCSD_WaveFunction(mo, mycc.t1, mycc.t2, l1=mycc.l1.T, l2=mycc.l2.transpose(2, 3, 0, 1))
            except TypeError:
                self.wf = CCSD_WaveFunction(mo, mycc.t1, mycc.t2)
        else:
            # Simply alias required quantities for now; this ensures functionality for arbitrary orders of CC via ebcc.
            self.wf = WaveFunction(mo)
            self.wf.make_rdm1 = mycc.make_rdm1_f
            self.wf.make_rdm2 = mycc.make_rdm2_f

    def _debug_exact_wf(self, wf):
        assert (self.is_fCCSD)
        mo = self.hamil.mo
        # Project onto cluster:
        ovlp = self.hamil._fragment.base.get_ovlp()
        ro = dot(wf.mo.coeff_occ.T, ovlp, mo.coeff_occ)
        rv = dot(wf.mo.coeff_vir.T, ovlp, mo.coeff_vir)
        t1 = dot(ro.T, wf.t1, rv)
        t2 = einsum('Ii,Jj,IJAB,Aa,Bb->ijab', ro, ro, wf.t2, rv, rv)
        if wf.l1 is not None:
            l1 = dot(ro.T, wf.l1, rv)
            l2 = einsum('Ii,Jj,IJAB,Aa,Bb->ijab', ro, ro, wf.l2, rv, rv)
        else:
            l1 = l2 = None
        self.wf = CCSD_WaveFunction(mo, t1, t2, l1=l1, l2=l2)
        self.converged = True

    def _debug_random_wf(self):
        mo = self.hamil.mo
        t1 = np.random.rand(mo.nocc, mo.nvir)
        l1 = np.random.rand(mo.nocc, mo.nvir)
        t2 = np.random.rand(mo.nocc, mo.nocc, mo.nvir, mo.nvir)
        l2 = np.random.rand(mo.nocc, mo.nocc, mo.nvir, mo.nvir)
        self.wf = CCSD_WaveFunction(mo, t1, t2, l1=l1, l2=l2)
        self.converged = True


class UEBCC_Solver(UClusterSolver, REBCC_Solver):
    @dataclasses.dataclass
    class Options(REBCC_Solver.Options, UClusterSolver.Options):
        pass

    # This should automatically work other than ensuring spin components are in a tuple.
    def construct_wavefunction(self, mycc, mo):
        if self.is_fCCSD:
            # Can use existing functionality
            def to_spin_tuple1(x):
                return x.aa, x.bb

            def to_spin_tuple2(x):
                return x.aaaa, x.abab, x.bbbb

            try:
                self.wf = CCSD_WaveFunction(mo,
                                            to_spin_tuple1(mycc.t1),
                                            to_spin_tuple2(mycc.t2),
                                            l1=tuple([x.T for x in to_spin_tuple1(mycc.l1)]),
                                            l2=tuple([x.transpose(2, 3, 0, 1) for x in to_spin_tuple2(mycc.l2)]),
                                            )
            except TypeError:
                self.wf = CCSD_WaveFunction(mo,
                                            to_spin_tuple1(mycc.t1),
                                            to_spin_tuple2(mycc.t2)
                                            )
        else:
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

    def _debug_exact_wf(self, wf):
        mo = self.cluster.mo
        # Project onto cluster:
        ovlp = self.hamil._fragment.base.get_ovlp()
        roa = dot(wf.mo.coeff_occ[0].T, ovlp, mo.coeff_occ[0])
        rob = dot(wf.mo.coeff_occ[1].T, ovlp, mo.coeff_occ[1])
        rva = dot(wf.mo.coeff_vir[0].T, ovlp, mo.coeff_vir[0])
        rvb = dot(wf.mo.coeff_vir[1].T, ovlp, mo.coeff_vir[1])
        t1a = dot(roa.T, wf.t1a, rva)
        t1b = dot(rob.T, wf.t1b, rvb)
        t2aa = einsum('Ii,Jj,IJAB,Aa,Bb->ijab', roa, roa, wf.t2aa, rva, rva)
        t2ab = einsum('Ii,Jj,IJAB,Aa,Bb->ijab', roa, rob, wf.t2ab, rva, rvb)
        t2bb = einsum('Ii,Jj,IJAB,Aa,Bb->ijab', rob, rob, wf.t2bb, rvb, rvb)
        t1 = (t1a, t1b)
        t2 = (t2aa, t2ab, t2bb)
        if wf.l1 is not None:
            l1a = dot(roa.T, wf.l1a, rva)
            l1b = dot(rob.T, wf.l1b, rvb)
            l2aa = einsum('Ii,Jj,IJAB,Aa,Bb->ijab', roa, roa, wf.l2aa, rva, rva)
            l2ab = einsum('Ii,Jj,IJAB,Aa,Bb->ijab', roa, rob, wf.l2ab, rva, rvb)
            l2bb = einsum('Ii,Jj,IJAB,Aa,Bb->ijab', rob, rob, wf.l2bb, rvb, rvb)
            l1 = (l1a, l1b)
            l2 = (l2aa, l2ab, l2bb)
        else:
            l1 = l2 = None
        self.wf = CCSD_WaveFunction(mo, t1, t2, l1=l1, l2=l2)
        self.converged = True

    def _debug_random_wf(self):
        raise NotImplementedError


class EB_REBCC_Solver(REBCC_Solver):
    @dataclasses.dataclass
    class Options(REBCC_Solver.Options):
        ansatz: str = "CCSD-S-1-1"
        fermion_wf: bool = False

    @property
    def is_fCCSD(self):
        if self.opts.fermion_wf:
            return super().is_fCCSD
        return False

    def get_nonnull_solver_opts(self):
        opts = super().get_nonnull_solver_opts()
        opts["omega"] = self.hamil.bos_freqs
        opts["g"] = self.get_couplings()
        return opts

    def get_couplings(self):
        # EBCC wants contribution  g_{xpq} p^\\dagger q b; need to transpose to get this contribution.
        return self.hamil.couplings.transpose(0, 2, 1)

    def construct_wavefunction(self, mycc, mo):
        super().construct_wavefunction(mycc, mo)

        def make_rdmeb(*args, **kwargs):
            dm = mycc.make_eb_coup_rdm()
            # We just want the bosonic excitation component.
            dm = dm[0].transpose(1, 2, 0)
            return (dm / 2, dm / 2)

        self.wf.make_rdmeb = make_rdmeb


class EB_UEBCC_Solver(EB_REBCC_Solver, UEBCC_Solver):
    @dataclasses.dataclass
    class Options(UEBCC_Solver.Options, EB_REBCC_Solver.Options):
        pass

    def get_couplings(self):
        # EBCC wants contribution  g_{xpq} p^\\dagger q b; need to transpose to get this contribution.
        return tuple([x.transpose(0, 2, 1) for x in self.hamil.couplings])

    def construct_wavefunction(self, mycc, mo):
        super().construct_wavefunction(mycc, mo)

        def make_rdmeb(*args, **kwargs):
            dm = mycc.make_eb_coup_rdm()
            # We just want the bosonic excitation component.
            dm = (dm.aa[0].transpose(1, 2, 0), dm.bb[0].transpose(1, 2, 0))
            return dm

        self.wf.make_rdmeb = make_rdmeb
