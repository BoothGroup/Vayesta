import dataclasses
from timeit import default_timer as timer

import numpy as np
import scipy
import scipy.optimize

from vayesta.core.util import *


class ClusterSolver:
    """Base class for cluster solver"""

    @dataclasses.dataclass
    class Options(OptionsBase):
        make_rdm1: bool = False
        make_rdm2: bool = False
        cpt_frag: float = None          # Chemical potential on fragment space

    @dataclasses.dataclass
    class Results:
        converged: bool = False         # Indicates convergence of iterative solvers, such as CCSD or FCI
        e_corr: float = 0.0             # Cluster correlation energy
        c_occ: np.array = None          # Occupied active orbitals
        c_vir: np.array = None          # Virtual active orbitals
        dm1: np.array = None            # 1-particle reducied density matrix in active orbital representation
        dm2: np.array = None            # 2-particle reducied density matrix in active orbital representation

        def get_init_guess(self, *args, **kwargs):
            """Return initial guess to restart kernel.

            This should return a dictionary, such that it can be used as:
            >>> init_guess = solver.results.get_init_guess()
            >>> solver.kernel(**init_guess)
            """
            raise NotImplementedError()

    def __init__(self, fragment, mo_coeff, mo_occ, nocc_frozen, nvir_frozen,
            options=None, log=None, **kwargs):
        """

        Arguments
        ---------
        nocc_frozen : int
            Number of frozen occupied orbitals. Need to be at the start of mo_coeff.
        nvir_frozen : int
            Number of frozen virtual orbitals. Need to be at the end of mo_coeff.
        """
        self.log = log or fragment.log
        self.fragment = fragment
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self.nocc_frozen = nocc_frozen
        self.nvir_frozen = nvir_frozen

        if options is None:
            options = self.Options(**kwargs)
        else:
            options = options.replace(kwargs)
        self.opts = options
        self.log.debug("Solver options:")
        for key, val in self.opts.items():
            self.log.debug('  > %-24s %r', key + ':', val)

        # Check MO orthogonality
        err = abs(dot(mo_coeff.T, self.base.get_ovlp(), mo_coeff) - np.eye(mo_coeff.shape[-1])).max()
        if err > 1e-4:
            self.log.error("MOs are not orthonormal: %.3e !", err)
        elif err > 1e-8:
            self.log.warning("MOs are not orthonormal: %.3e", err)
        assert (err < 1e-6), ("MO not orthogonal: %.3e" % err)

        # Results
        self.results = None

    @property
    def base(self):
        return self.fragment.base

    @property
    def mf(self):
        return self.fragment.mf

    @property
    def mol(self):
        return self.fragment.mol

    @property
    def nmo(self):
        """Total number of MOs (not just active)."""
        return self.mo_coeff.shape[-1]

    @property
    def nocc(self):
        return np.count_nonzero(self.mo_occ > 0)

    @property
    def nactive(self):
        """Number of active MOs."""
        return self.nmo - self.nfrozen

    @property
    def nfrozen(self):
        return self.nocc_frozen + self.nvir_frozen

    def get_active_slice(self):
        slc = np.s_[self.nocc_frozen:self.nocc_frozen+self.nactive]
        return slc

    def get_frozen_indices(self):
        nmo = self.mo_coeff.shape[-1]
        idx = list(range(self.nocc_frozen)) + list(range(nmo-self.nvir_frozen, nmo))
        return idx

    @property
    def c_active_occ(self):
        """Active occupied orbital coefficients."""
        return self.mo_coeff[:,self.nocc_frozen:self.nocc]

    @property
    def c_active_vir(self):
        """Active virtual orbital coefficients."""
        return self.mo_coeff[:,self.nocc:self.nocc_frozen+self.nactive]

    @property
    def c_active(self):
        return self.mo_coeff[:,self.get_active_slice()]

    def get_eris(self):
        """Abstract method."""
        raise NotImplementedError()

    def get_v_frag(self, cpt=None, c_frag=None):
        """Potential due to fragment chemical potential in AO basis."""
        if cpt is None: cpt = self.opts.cpt_frag
        if c_frag is None: c_frag = self.fragment.c_proj
        cs = np.dot(c_proj.T, self.base.get_ovlp())
        p_frag = np.dot(cs.T, cs)
        return cpt * p_frag

    def get_nelec_frag(self, dm1, c_frag=None):
        if c_frag is None: c_frag = self.fragment.c_proj
        csc = dot(c_proj.T, self.base.get_ovlp(), self.c_active)
        ne_frag = einsum('xi,ij,xj->', csc, dm1, csc)
        return ne_frag

    def kernel_optimize_cpt(self, nelectron_target, cpt_guess=0, atol=1e-8, rtol=1e-8, cpt_radius=0.1, **kwargs):
        """Optimize a chemical potential within the fragment space to get the target number of electrons.

        Additional keyword arguments are passed to the solver kernel.

        Parameters
        ----------
        nelectron_target: int
            Target number of electrons.
        cpt_guess: float, optional
            Initial guess for fragment chemical potential. Default: 0.
        atol: float, optional
            Absolute electron number tolerance. Default: 1e-8.
        rtol: float, optional
            Relative electron number tolerance. Default: 1e-8
        cpt_radius: float, optional
            Search radius for chemical potential. Default: 0.1.

        Returns
        -------
        results:
            Solver results.
        """

        mf = self.base.mf
        # Save current hcore to restore later
        #hfunc0 = mf.get_hcore
        #hcache0 = self.base._hcore
        # Unmodified hcore
        h0 = self.base.get_hcore()
        # Fragment projector
        cs = np.dot(self.fragment.c_proj.T, self.base.get_ovlp())
        p_proj = np.dot(cs.T, cs)
        csc = np.dot(cs, self.c_active)

        class CptFound(RuntimeError):
            """Raise when electron error is below tolerance."""
            pass

        self.opts.make_rdm1 = True
        results = None
        err = None
        cpt_opt = None
        iterations = 0
        init_guess = {}

        #eris = kwargs.pop('eris', self.get_eris())
        eris = kwargs.pop('eris')

        def electron_err(cpt):
            nonlocal results, err, cpt_opt, iterations, init_guess
            if cpt:
                #mf.get_hcore = lambda *args : (h0 + cpt*p_proj)
                #self.base._hcore = None
                self.base.set_hcore(h0 + cpt*p_proj)
            kwargs.update(init_guess)
            self.log.debugv("kwargs keys for solver: %r", kwargs.keys())
            results = self.kernel(**kwargs)
            #results = self.kernel(eris=eris, **kwargs)
            init_guess = results.get_init_guess()
            # Restore get_hcore
            if cpt:
                self.base.set_hcore(h0)
                #mf.get_hcore = hfunc0
                #self.base._hcore = hcache0
            if not results.converged:
                raise ConvergenceError()
            ne_frag = einsum('xi,ij,xj->', csc, results.dm1, csc)
            err = (ne_frag - nelectron_target)
            self.log.debugv("Fragment chemical potential= %+12.8f Ha:  electrons in fragment= %.8f  error= %+.3e", cpt, ne_frag, err)
            iterations += 1
            if abs(err) < (atol + rtol*nelectron_target):
                cpt_opt = cpt
                raise CptFound()
            return err

        # First run with cpt_guess:
        try:
            err0 = electron_err(cpt_guess)
        except CptFound:
            self.log.debug("Chemical potential= %.6f leads to electron error= %.3e within tolerance (atol= %.1e, rtol= %.1e)", cpt_guess, err, atol, rtol)
            return results

        # Not enough electrons in fragment space -> lower fragment chemical potential:
        assert (cpt_radius > 0)
        if err0 < 0:
            bounds = np.asarray([cpt_guess-cpt_radius, cpt_guess])
        # Too many electrons in fragment space -> raise fragment chemical potential:
        else:
            bounds = np.asarray([cpt_guess, cpt_guess+cpt_radius])
        for ntry in range(3):
            try:
                cpt, res = scipy.optimize.brentq(electron_err, a=bounds[0], b=bounds[1], xtol=1e-12, full_output=True)
                if res.converged:
                    self.log.warning("Chemical potential converged to %+16.8f, but electron error is still %.3e", cpt, err)
                    cpt_opt = cpt
                    raise CptFound
            except CptFound:
                break
            # Could not find chemical potential in bracket:
            except ValueError:
                bounds *= 10
                self.log.warning("Interval for chemical potential search too small. New search interval: [%f %f]", *bounds)
                continue
            # Could not convergence in bracket:
            except ConvergenceError:
                bounds /= 2
                self.log.warning("Solver did not converge. New search interval: [%f %f]", *bounds)
                continue
            raise RuntimeError("Invalid state: electron error= %.3e" % err)
        else:
            errmsg = ("Could not find chemical potential within interval [%f %f]!" % (bounds[0], bounds[1]))
            self.log.critical(errmsg)
            raise RuntimeError(errmsg)

        self.log.info("Chemical potential optimized in %d iterations= %+16.8f Ha", iterations, cpt_opt)
        return results
