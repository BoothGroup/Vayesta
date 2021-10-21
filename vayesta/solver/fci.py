import dataclasses

import numpy as np

import pyscf
import pyscf.ao2mo
import pyscf.ci
import pyscf.mcscf
import pyscf.fci
import pyscf.fci.addons

from vayesta.core.util import *
from .solver import ClusterSolver


class FCI_Solver(ClusterSolver):

    @dataclasses.dataclass
    class Options(ClusterSolver.Options):
        threads: int = 1            # Number of threads for multi-threaded FCI
        lindep: float = None        # Linear dependency tolerance. If None, use PySCF default
        conv_tol: float = None      # Convergence tolerance. If None, use PySCF default
        solver_spin: bool = True    # Use direct_spin1 if True, or direct_spin0 otherwise
        fix_spin: float = 0.0       # If set to a number, the given S^2 value will be enforced

    @dataclasses.dataclass
    class Results(ClusterSolver.Results):
        # CI coefficients
        civec: np.array = None      # Vector of all CI-coefficients
        c0: float = None            # C0 coefficient
        c1: np.array = None         # C1 coefficients
        c2: np.array = None         # C2 coefficients

        def get_init_guess(self):
            return {'ci0' : self.civec}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if self.opts.solver_spin:
            solver = pyscf.fci.direct_spin1.FCISolver(self.mol)
        else:
            solver = pyscf.fci.direct_spin0.FCISolver(self.mol)
        self.log.debugv("type(solver)= %r", type(solver))
        # Set options
        if self.opts.threads is not None: solver.threads = self.opts.threads
        if self.opts.conv_tol is not None: solver.conv_tol = self.opts.conv_tol
        if self.opts.lindep is not None: solver.lindep = self.opts.lindep
        if self.opts.fix_spin not in (None, False):
            spin = self.opts.fix_spin
            self.log.debugv("Fixing spin of FCI solver to S^2= %f", spin)
            solver = pyscf.fci.addons.fix_spin_(solver, ss=spin)
        self.solver = solver

    def get_eris(self):
        with log_time(self.log.timing, "Time for AO->MO transformation: %s"):
            eris = self.base.get_eris_array(self.c_active)
        return eris

    def get_heff(self, eris, with_vext=True):
        nocc = self.nocc - self.nocc_frozen
        occ = np.s_[:nocc]
        f_act = np.linalg.multi_dot((self.c_active.T, self.base.get_fock(), self.c_active))
        v_act = 2*einsum('iipq->pq', eris[occ,occ]) - einsum('iqpi->pq', eris[occ,:,:,occ])
        h_eff = f_act - v_act
        # This should be equivalent to:
        #core = np.s_[:self.nocc_frozen]
        #dm_core = 2*np.dot(self.mo_coeff[:,core], self.mo_coeff[:,core].T)
        #v_core = self.mf.get_veff(dm=dm_core)
        #h_eff = np.linalg.multi_dot((self.c_active.T, self.base.get_hcore()+v_core, self.c_active))
        if with_vext and self.opts.v_ext is not None:
            h_eff += self.opts.v_ext
        return h_eff

    def kernel(self, ci0=None, eris=None):
        """Run FCI kernel."""

        if eris is None: eris = self.get_eris()
        heff = self.get_heff(eris)
        nelec = sum(self.mo_occ[self.get_active_slice()])
        assert np.isclose(nelec, round(nelec))
        nelec = int(round(nelec))

        t0 = timer()
        e_fci, civec = self.solver.kernel(heff, eris, self.nactive, nelec, ci0=ci0)
        if not self.solver.converged:
            self.log.error("FCI not converged!")
        else:
            self.log.debugv("FCI converged.")
        self.log.timing("Time for FCI: %s", time_string(timer()-t0))
        # TODO: This requires the E_core energy (and nuc-nuc repulsion)
        e_corr = np.nan
        s2, mult = self.solver.spin_square(civec, self.nactive, nelec)
        self.log.info("FCI: S^2= %.10f  multiplicity= %.10f", s2, mult)

        nocc = self.nocc - self.nocc_frozen
        cisdvec = pyscf.ci.cisd.from_fcivec(civec, self.nactive, nelec)
        c0, c1, c2 = pyscf.ci.cisd.cisdvec_to_amplitudes(cisdvec, self.nactive, nocc)

        results = self.Results(
                converged=self.solver.converged, e_corr=e_corr, c_occ=self.c_active_occ, c_vir=self.c_active_vir,
                civec=civec, c0=c0, c1=c1, c2=c2)

        if self.opts.make_rdm2:
            results.dm1, results.dm2 = self.solver.make_rdm12(civec, self.nactive, nelec)
        elif self.opts.make_rdm1:
            results.dm1 = self.solver.make_rdm1(civec, self.nactive, nelec)

        return results


    def kernel_casci(self, init_guess=None, eris=None):
        """Old kernel function, using an CASCI object."""
        nelec = sum(self.mo_occ[self.get_active_slice()])
        casci = pyscf.mcscf.CASCI(self.mf, self.nactive, nelec)
        casci.canonicalization = False
        if self.opts.threads is not None: casci.fcisolver.threads = self.opts.threads
        if self.opts.conv_tol is not None: casci.fcisolver.conv_tol = self.opts.conv_tol
        if self.opts.lindep is not None: casci.fcisolver.lindep = self.opts.lindep
        # FCI default values:
        #casci.fcisolver.conv_tol = 1e-10
        #casci.fcisolver.lindep = 1e-14

        self.log.debug("Running CASCI with (%d, %d) CAS", nelec, self.nactive)
        t0 = timer()
        e_tot, e_cas, wf, *_ = casci.kernel(mo_coeff=self.mo_coeff)
        self.log.debug("FCI done. converged: %r", casci.converged)
        self.log.timing("Time for FCI: %s", time_string(timer()-t0))
        e_corr = (e_tot-self.mf.e_tot)

        cisdvec = pyscf.ci.cisd.from_fcivec(wf, self.nactive, nelec)
        nocc = nelec // 2
        c0, c1, c2 = pyscf.ci.cisd.cisdvec_to_amplitudes(cisdvec, self.nactive, nocc)

        # Temporary workaround (eris needed for energy later)
        if self.mf._eri is not None:
            class ERIs:
                pass
            eris = ERIs()
            c_act = self.mo_coeff[:,self.get_active_slice()]
            eris.fock = np.linalg.multi_dot((c_act.T, self.base.get_fock(), c_act))
            g = pyscf.ao2mo.full(self.mf._eri, c_act)
            o = np.s_[:nocc]
            v = np.s_[nocc:]
            eris.ovvo = pyscf.ao2mo.restore(1, g, self.nactive)[o,v,v,o]
        else:
            # TODO
            pass

        results = self.Results(
                converged=casci.converged, e_corr=e_corr, c_occ=self.c_active_occ, c_vir=self.c_active_vir, eris=eris,
                c0=c0, c1=c1, c2=c2)

        if self.opts.make_rdm2:
            results.dm1, results.dm2 = casci.fcisolver.make_rdm12(wf, self.nactive, nelec)
        elif self.opts.make_rdm1:
            results.dm1 = casci.fcisolver.make_rdm1(wf, self.nactive, nelec)

        return results

    #kernel = kernel_casci
