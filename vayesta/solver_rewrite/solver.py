import dataclasses
from timeit import default_timer as timer
import copy

import numpy as np
import scipy
import scipy.optimize

from pyscf import scf

from vayesta.core.util import *


class ClusterSolver:
    """Base class for cluster solver"""

    @dataclasses.dataclass
    class Options(OptionsBase):
        pass

    def __init__(self, hamil, log=None, **kwargs):
        """
        Arguments
        ---------
        """
        self.hamil = hamil
        self.log = (log or hamil.log)
        # --- Options:
        self.opts = self.Options()
        self.opts.update(**kwargs)
        self.log.info("Parameters of %s:" % self.__class__.__name__)
        self.log.info(break_into_lines(str(self.opts), newline='\n    '))

        # Additional external potential
        self.v_ext = None

        # --- Results
        self.converged = False
        self.e_corr = 0
        self.wf = None
        self.dm1 = None
        self.dm2 = None

    def kernel(self, *args, **kwargs):
        """Set up everything for a calculation on the CAS and pass this to the solver-specific kernel that runs on this
        information."""
        raise AbstractMethodError

    def optimize_cpt(self, nelectron, cpt_guess=0, atol=1e-6, rtol=1e-6, cpt_radius=0.5):
        """Enables chemical potential optimization to match a number of electrons in the fragment space.

        Parameters
        ----------
        nelectron: float
            Target number of electrons.
        c_frag: array
            Fragment orbitals.
        cpt_guess: float, optional
            Initial guess for fragment chemical potential. Default: 0.
        atol: float, optional
            Absolute electron number tolerance. Default: 1e-6.
        rtol: float, optional
            Relative electron number tolerance. Default: 1e-6
        cpt_radius: float, optional
            Search radius for chemical potential. Default: 0.5.

        Returns
        -------
        results:
            Solver results.
        """

        kernel_orig = self.kernel
        # Make projector into fragment space

        p_frag = self.hamil.target_space_projector

        class CptFound(RuntimeError):
            """Raise when electron error is below tolerance."""
            pass

        def kernel(self, *args, eris=None, **kwargs):
            result = None
            err = None
            cpt_opt = None
            iterations = 0
            init_guess = {}
            err0 = None

            # Avoid calculating the ERIs multiple times:
            if eris is None:
                eris = self.get_eris()

            def electron_err(cpt):
                nonlocal result, err, err0, cpt_opt, iterations, init_guess
                # Avoid recalculation of cpt=0.0 in SciPy:
                if (cpt == 0) and (err0 is not None):
                    self.log.debugv("Chemical potential %f already calculated - returning error= %.8f", cpt, err0)
                    return err0

                kwargs.update(init_guess)
                self.log.debugv("kwargs keys for solver: %r", kwargs.keys())

                replace = {}
                if cpt:
                    v_ext_0 = (self.v_ext if self.v_ext is not None else 0)
                    replace['v_ext'] = self.get_vext(v_ext_0, cpt)
                self.reset()
                with replace_attr(self, **replace):
                    results = kernel_orig(eris=eris, **kwargs)
                if not self.converged:
                    raise ConvergenceError()
                dm1 = self.wf.make_rdm1()
                if self.is_rhf:
                    ne_frag = einsum('xi,ij,xj->', p_frag, dm1, p_frag)
                else:
                    ne_frag = (einsum('xi,ij,xj->', p_frag[0], dm1[0], p_frag[0])
                               + einsum('xi,ij,xj->', p_frag[1], dm1[1], p_frag[1]))

                err = (ne_frag - nelectron)
                self.log.debug("Fragment chemical potential= %+12.8f Ha:  electrons= %.8f  error= %+.3e", cpt, ne_frag,
                               err)
                iterations += 1
                if abs(err) < (atol + rtol * nelectron):
                    cpt_opt = cpt
                    raise CptFound()
                # Initial guess for next chemical potential
                # init_guess = results.get_init_guess()
                init_guess = self.get_init_guess()
                return err

            # First run with cpt_guess:
            try:
                err0 = electron_err(cpt_guess)
            except CptFound:
                self.log.debug(
                    "Chemical potential= %.6f leads to electron error= %.3e within tolerance (atol= %.1e, rtol= %.1e)",
                    cpt_guess, err, atol, rtol)
                return result

            # Not enough electrons in fragment space -> raise fragment chemical potential:
            if err0 < 0:
                lower = cpt_guess
                upper = cpt_guess + cpt_radius
            # Too many electrons in fragment space -> lower fragment chemical potential:
            else:
                lower = cpt_guess - cpt_radius
                upper = cpt_guess

            self.log.debugv("Estimated bounds: %.3e %.3e", lower, upper)
            bounds = np.asarray([lower, upper], dtype=float)

            for ntry in range(5):
                try:
                    cpt, res = scipy.optimize.brentq(electron_err, a=bounds[0], b=bounds[1], xtol=1e-12,
                                                     full_output=True)
                    if res.converged:
                        raise RuntimeError(
                            "Chemical potential converged to %+16.8f, but electron error is still %.3e" % (cpt, err))
                except CptFound:
                    break
                # Could not find chemical potential in bracket:
                except ValueError:
                    bounds *= 2
                    self.log.warning("Interval for chemical potential search too small. New search interval: [%f %f]",
                                     *bounds)
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
            return result

        # Replace kernel:
        self.kernel = kernel.__get__(self)

    def get_vext(self, v_ext_0, cpt):
        return v_ext_0 - cpt * self.hamil.active_space_projector


class UClusterSolver(ClusterSolver):

    def get_vext(self, v_ext_0, cpt):
        pfrag = self.hamil.target_space_projector
        # Surely None would be a better default?
        if v_ext_0 == 0:
            v_ext_0 = (v_ext_0, v_ext_0)
        return (v_ext_0[0] - cpt * pfrag[0], v_ext_0[1] - cpt * pfrag[1])
