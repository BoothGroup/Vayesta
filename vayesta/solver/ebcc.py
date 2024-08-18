import dataclasses

import numpy as np

from vayesta.core.types import WaveFunction, CCSD_WaveFunction, EBCC_WaveFunction
from vayesta.core.util import dot, einsum
from vayesta.solver.solver import ClusterSolver, UClusterSolver
import ebcc


class REBCC_Solver(ClusterSolver):
    @dataclasses.dataclass
    class Options(ClusterSolver.Options):
        ansatz: str = "CCSD"
        solve_lambda: bool = False  # If false, use Lambda=T approximation
        # Convergence
        max_cycle: int = 200  # Max number of iterations
        conv_tol: float = None  # Convergence energy tolerance
        conv_tol_normt: float = None  # Convergence amplitude tolerance
        store_as_ccsd: bool = False  # Store results as CCSD_WaveFunction
        c_cas_occ: np.array = None  # Hacky place to put active space orbitals.
        c_cas_vir: np.array = None

    def kernel(self):
        # Use pyscf mean-field representation.
        mf_clus, frozen = self.hamil.to_pyscf_mf(allow_dummy_orbs=False)

        mf_clus.mo_coeff, space = self.get_space(self.hamil.cluster.c_active, mf_clus.mo_occ, frozen=frozen)

        mycc = ebcc.EBCC(
            mf_clus, log=self.log, ansatz=self.opts.ansatz, space=space, shift=False, **self.get_nonnull_solver_opts()
        )
        mycc.kernel()
        self.converged = mycc.converged
        if self.opts.solve_lambda:
            mycc.solve_lambda()
            self.converged = self.converged and mycc.converged_lambda
        # Now just need to wrangle EBCC results into wavefunction format.
        self.construct_wavefunction(mycc, self.hamil.get_mo(mf_clus.mo_coeff))

    def get_space(self, mo_coeff, mo_occ, frozen=None):
        s = self.hamil.orig_mf.get_ovlp()
        c_active_occ = self.opts.c_cas_occ
        c_active_vir = self.opts.c_cas_vir
        if c_active_occ is not None and c_active_vir is not None:
            c_active_occ = dot(mo_coeff.T, s, c_active_occ)
            c_active_vir = dot(mo_coeff.T, s, c_active_vir)
        elif c_active_occ is not None or c_active_vir is not None:
            raise ValueError(
                "Both or neither of the occupied and virtual components of an active space must be specified."
            )
        mo_coeff = dot(mo_coeff.T, s, mo_coeff)
        return gen_space(
            mo_coeff[:, mo_occ > 0], mo_coeff[:, mo_occ == 0], c_active_occ, c_active_vir, frozen_orbs=frozen
        )

    def get_nonnull_solver_opts(self):
        def add_nonull_opt(d, key, newkey):
            val = self.opts.get(key)
            if val is not None:
                d[newkey] = val

        opts = {}
        for key, newkey in zip(["max_cycle", "conv_tol", "conv_tol_normt"], ["max_iter", "e_tol", "t_tol"]):
            add_nonull_opt(opts, key, newkey)
        return opts

    def construct_wavefunction(self, mycc, mo, mbos=None):
        if self.opts.store_as_ccsd:
            # Can use existing functionality
            try:
                self.wf = CCSD_WaveFunction(mo, mycc.t1, mycc.t2, l1=mycc.l1.T, l2=mycc.l2.transpose(2, 3, 0, 1))
            except TypeError:
                self.wf = CCSD_WaveFunction(mo, mycc.t1, mycc.t2)
        else:
            self.wf = EBCC_WaveFunction(mo, mycc.ansatz, mycc.amplitudes, mycc.lambdas, mbos=mbos)

        # Need to rotate wavefunction back into original cluster active space.
        self.wf.rotate(t=mycc.mo_coeff.T, inplace=True)

    def _debug_exact_wf(self, wf):
        assert self.is_fCCSD
        mo = self.hamil.mo
        # Project onto cluster:
        ovlp = self.hamil._fragment.base.get_ovlp()
        ro = dot(wf.mo.coeff_occ.T, ovlp, mo.coeff_occ)
        rv = dot(wf.mo.coeff_vir.T, ovlp, mo.coeff_vir)
        t1 = dot(ro.T, wf.t1, rv)
        t2 = einsum("Ii,Jj,IJAB,Aa,Bb->ijab", ro, ro, wf.t2, rv, rv)
        if wf.l1 is not None:
            l1 = dot(ro.T, wf.l1, rv)
            l2 = einsum("Ii,Jj,IJAB,Aa,Bb->ijab", ro, ro, wf.l2, rv, rv)
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
        c_cas_occ: np.array = (None, None)  # Hacky place to put active space orbitals.
        c_cas_vir: np.array = (None, None)

    def get_space(self, mo_coeff, mo_occ, frozen=None):
        s = self.hamil.orig_mf.get_ovlp()

        def _get_space(c, occ, co_cas, cv_cas, fr):
            # Express active orbitals in terms of cluster orbitals.
            if co_cas is not None and cv_cas is not None:
                co_cas = dot(c.T, s, co_cas)
                cv_cas = dot(c.T, s, cv_cas)
            elif co_cas is not None or cv_cas is not None:
                raise ValueError("Both or neither of the occupied and virtual components of an active space.")
            c = dot(c.T, s, c)
            return gen_space(c[:, occ > 0], c[:, occ == 0], co_cas, cv_cas, frozen_orbs=fr)

        if frozen is None:
            frozen = [None, None]
        ca, spacea = _get_space(mo_coeff[0], mo_occ[0], self.opts.c_cas_occ[0], self.opts.c_cas_vir[0], frozen[0])
        cb, spaceb = _get_space(mo_coeff[1], mo_occ[1], self.opts.c_cas_occ[1], self.opts.c_cas_vir[1], frozen[1])
        return (ca, cb), (spacea, spaceb)

    # This should automatically work other than ensuring spin components are in a tuple.
    def construct_wavefunction(self, mycc, mo, mbos=None):
        if self.opts.store_as_ccsd:
            # Can use existing functionality
            def to_spin_tuple1(x):
                return x.aa, x.bb

            # NB EBCC doesn't antisymmetrise the T2s by default, and has an odd factor of two.

            def antisymmetrize_t2(t2):
                t2 = t2 - t2.transpose(1, 0, 2, 3)
                t2 = t2 - t2.transpose(0, 1, 3, 2)
                return t2 / 2.0

            def to_spin_tuple2(x):
                return antisymmetrize_t2(x.aaaa), x.abab, antisymmetrize_t2(x.bbbb)

            try:
                self.wf = CCSD_WaveFunction(
                    mo,
                    to_spin_tuple1(mycc.t1),
                    to_spin_tuple2(mycc.t2),
                    l1=tuple([x.T for x in to_spin_tuple1(mycc.l1)]),
                    l2=tuple([x.transpose(2, 3, 0, 1) for x in to_spin_tuple2(mycc.l2)]),
                )
            except TypeError:
                self.wf = CCSD_WaveFunction(mo, to_spin_tuple1(mycc.t1), to_spin_tuple2(mycc.t2))
        else:
            self.wf = EBCC_WaveFunction(mo, mycc.ansatz, mycc.amplitudes, mycc.lambdas, mbos=mbos)

        self.wf.rotate(t=[x.T for x in mycc.mo_coeff], inplace=True)

    def _debug_exact_wf(self, wf):
        mo = self.hamil.mo
        # Project onto cluster:
        ovlp = self.hamil._fragment.base.get_ovlp()
        roa = dot(wf.mo.coeff_occ[0].T, ovlp, mo.coeff_occ[0])
        rob = dot(wf.mo.coeff_occ[1].T, ovlp, mo.coeff_occ[1])
        rva = dot(wf.mo.coeff_vir[0].T, ovlp, mo.coeff_vir[0])
        rvb = dot(wf.mo.coeff_vir[1].T, ovlp, mo.coeff_vir[1])
        t1a = dot(roa.T, wf.t1a, rva)
        t1b = dot(rob.T, wf.t1b, rvb)
        t2aa = einsum("Ii,Jj,IJAB,Aa,Bb->ijab", roa, roa, wf.t2aa, rva, rva)
        t2ab = einsum("Ii,Jj,IJAB,Aa,Bb->ijab", roa, rob, wf.t2ab, rva, rvb)
        t2bb = einsum("Ii,Jj,IJAB,Aa,Bb->ijab", rob, rob, wf.t2bb, rvb, rvb)
        t1 = (t1a, t1b)
        t2 = (t2aa, t2ab, t2bb)
        if wf.l1 is not None:
            l1a = dot(roa.T, wf.l1a, rva)
            l1b = dot(rob.T, wf.l1b, rvb)
            l2aa = einsum("Ii,Jj,IJAB,Aa,Bb->ijab", roa, roa, wf.l2aa, rva, rva)
            l2ab = einsum("Ii,Jj,IJAB,Aa,Bb->ijab", roa, rob, wf.l2ab, rva, rvb)
            l2bb = einsum("Ii,Jj,IJAB,Aa,Bb->ijab", rob, rob, wf.l2bb, rvb, rvb)
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
        store_as_ccsd: bool = False  # Store results as fermionic CCSD_WaveFunction

    def get_nonnull_solver_opts(self):
        opts = super().get_nonnull_solver_opts()
        opts["omega"] = self.hamil.bos_freqs
        opts["g"] = self.get_couplings()
        opts["G"] = self.hamil.boson_nonconserving
        return opts

    def get_couplings(self):
        # EBCC wants contribution  g_{xpq} p^\\dagger q b; need to transpose to get this contribution.
        return self.hamil.couplings.transpose(0, 2, 1)

    def construct_wavefunction(self, mycc, mo, mbos=None):
        self.wf = EBCC_WaveFunction(
            mo, mycc.ansatz, mycc.amplitudes, mycc.lambdas, mbos=mbos, xi=self.hamil.polaritonic_shift
        )
        self.wf.rotate(t=mycc.mo_coeff.T, inplace=True)


class EB_UEBCC_Solver(EB_REBCC_Solver, UEBCC_Solver):
    @dataclasses.dataclass
    class Options(UEBCC_Solver.Options, EB_REBCC_Solver.Options):
        pass

    def get_couplings(self):
        # EBCC wants contribution  g_{xpq} p^\\dagger q b; need to transpose to get this contribution.
        #return tuple([x.transpose(0, 2, 1) for x in self.hamil.couplings])
        # EBCC now wants an array, with spin as the first index
        assert(np.allclose(self.hamil.couplings[0].shape, self.hamil.couplings[1].shape))
        sh = self.hamil.couplings[0].shape
        g = np.zeros((2, sh[0], sh[2], sh[1]), dtype=self.hamil.couplings[0].dtype)
        g[0,:,:,:] = self.hamil.couplings[0].transpose(0,2,1)
        g[1,:,:,:] = self.hamil.couplings[1].transpose(0,2,1)
        return g 

    def construct_wavefunction(self, mycc, mo, mbos=None):
        self.wf = EBCC_WaveFunction(
            mo, mycc.ansatz, mycc.amplitudes, mycc.lambdas, mbos=mbos, xi=self.hamil.polaritonic_shift
        )
        self.wf.rotate(t=[x.T for x in mycc.mo_coeff], inplace=True)


def gen_space(c_occ, c_vir, co_active=None, cv_active=None, frozen_orbs=None):
    """Given the occupied and virtual orbital coefficients in the local cluster basis, which orbitals are frozen, and
    any active space orbitals in this space generate appropriate coefficients and ebcc.Space inputs for a calculation.
    Inputs:
        c_occ: occupied orbital coefficients in local cluster basis.
        c_vir: virtual orbital coefficients in local cluster basis.
        co_active: occupied active space orbitals in local cluster basis.
        cv_active: virtual active space orbitals in local cluster basis.
        frozen_orbs: indices of orbitals to freeze in local cluster basis.
    Outputs:
        c: coefficients for the active space orbitals in the local cluster basis.
        space: ebcc.Space object defining the resulting active space behaviours.
    """
    no, nv = c_occ.shape[1], c_vir.shape[1]

    have_actspace = not (co_active is None and cv_active is None)

    if co_active is None or cv_active is None:
        if have_actspace:
            raise ValueError("Active space must specify both occupied and virtual orbitals.")
    # Default to just using original cluster orbitals.
    c = np.hstack([c_occ, c_vir])

    occupied = np.hstack([np.ones(no, dtype=bool), np.zeros(nv, dtype=bool)])
    frozen = np.zeros(no + nv, dtype=bool)
    if frozen_orbs is not None:
        frozen[frozen_orbs] = True

    def gen_actspace(c_full, c_act, c_frozen, tol=1e-8):
        """Generate orbital rotation to define our space, along with frozen and active identifiers."""
        # Check we don't have any overlap between frozen and active orbitals.
        if c_frozen.shape[1] > 0:
            af_ovlp = dot(c_act.T, c_frozen)
            if not np.allclose(np.linalg.svd(af_ovlp)[1], 0, atol=tol):
                raise ValueError("Specified frozen and active orbitals overlap!")
        if c_act.shape[1] > 0:
            # We could use the portion of the active space orbitals inside the cluster instead in future.
            full_active_ovlp = dot(c_full.T, c_act)
            if not np.allclose(np.linalg.svd(full_active_ovlp)[1], 1, atol=tol):
                raise ValueError("Specified active orbitals are not spanned by the full cluster space!")
        # Can now safely generate coefficients; get the space spanned by undetermined orbitals.
        d_rest = dot(c_full, c_full.T) - dot(c_act, c_act.T) - dot(c_frozen, c_frozen.T)
        e, c = np.linalg.eigh(d_rest)
        c_rest = c[:, e > tol]
        c = np.hstack([c_frozen, c_rest, c_act])
        # Check that we have the right number of orbitals.
        assert c.shape[1] == c_full.shape[1]
        # Generate information.
        frozen_orbs = np.zeros(c.shape[1], dtype=bool)
        active = np.zeros(c.shape[1], dtype=bool)
        frozen_orbs[: c_frozen.shape[1]] = True
        active[-c_act.shape[1] :] = True
        return c, frozen_orbs, active

    if have_actspace:
        c_occ, frozen_occ, active_occ = gen_actspace(c_occ, co_active, c_occ[:, frozen[:no]])
        c_vir, frozen_vir, active_vir = gen_actspace(c_vir, cv_active, c_vir[:, frozen[no:]])

        c = np.hstack([c_occ, c_vir[:, ::-1]])
        frozen = np.hstack([frozen_occ, frozen_vir[::-1]])
        active = np.hstack([active_occ, active_vir[::-1]])
    else:
        active = np.zeros_like(frozen, dtype=bool)

    return c, ebcc.Space(occupied=occupied, frozen=frozen, active=active)
