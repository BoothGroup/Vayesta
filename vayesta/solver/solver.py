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
        make_rdm1: bool = NotSet
        make_rdm2: bool = NotSet

    @dataclasses.dataclass
    class Results:
        converged: bool = False
        e_corr: float = 0.0
        c_occ: np.array = None
        c_vir: np.array = None
        # Density matrix in MO representation:
        dm1: np.array = None
        dm2: np.array = None
        eris: 'typing.Any' = None

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
        options = options.replace(self.base.opts, select=NotSet)
        self.opts = options


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

    def kernel_optimize_cpt(self, nelectron_target, *args, bounds=[-1.0, 1.0], tol=1e-8, **kwargs):

        mf = self.base.mf
        # Save current hcore to restore later
        hfunc0 = mf.get_hcore
        hcache0 = self.base._hcore
        # Unmodified hcore
        h0 = self.base.get_hcore()
        # Fragment projector
        cs = np.dot(self.fragment.c_frag.T, self.base.get_ovlp())
        p_frag = np.dot(cs.T, cs)
        csc = np.dot(cs, self.c_active)

        self.opts.make_rdm1 = True
        results = None
        err = None

        def electron_err(cpt):
            nonlocal results, err
            mf.get_hcore = lambda *args : (h0 - cpt*p_frag)
            # Important: delete cache of QuantumEmbeddingMethod:
            self.base._hcore = None
            results = self.kernel(*args, **kwargs)
            if not results.converged:
                raise ConvergenceError()
            ne_frag = einsum('xi,ij,xj->', csc, results.dm1, csc)
            err = (ne_frag - nelectron_target)
            self.log.debugv("Electron number in fragment= %.8f  target=  %.8f  error= %.8f  chem. pot.=  %16.8f Ha", ne_frag, nelectron_target, err, cpt)
            return err

        for ndouble in range(5):
            try:
            	cpt, res = scipy.optimize.brentq(electron_err, a=bounds[0], b=bounds[1], xtol=tol, full_output=True)
	    except ValueError:
                if err < 0:
                    bounds[1] *= 2
                else:
                    bounds[2] *= 2
                self.log.debug("Bounds for chemical potential search too small. New bounds: [%f %f]", *bounds)
                continue
            # Could not convergence in bracket:
            except ConvergenceError:
                bounds[0], bounds[1] = bounds[0]/2, bounds[1]/2
                self.log.debug("Solver did not converge. New bounds: [%f %f]", *bounds)

            if res.converged:
                break
            else:
                errmsg = "Correct chemical potential not found."
                self.log.critical(errmsg)
                raise RuntimeError(errmsg)
        else:
            errmsg = "Could not find chemical potential within [%f %f]" % (lower_bound, upper_bound)
            self.log.critical(errmsg)
            raise RuntimeError(errmsg)

        self.log.info("Optimized chemical potential= % 16.8f Ha", cpt)
        # Restore hcore
        mf.get_hcore = hfunc0
        self.base._hcore = hcache0
        return results
