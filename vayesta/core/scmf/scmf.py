"""Self-consistent mean-field decorators"""

import numpy as np
import pyscf
import pyscf.lib
from vayesta.core.util import *


class SCMF:

    name = "SCMF"

    def __init__(self, emb, etol=1e-8, dtol=1e-6, maxiter=100, damping=0.0, diis=True):
        self.emb = emb
        self.etol = (etol if etol is not None else np.inf)
        self.dtol = (dtol if dtol is not None else np.inf)
        self.maxiter = maxiter
        self.damping = damping
        self.diis = diis
        self.iteration = 0
        # Save original kernel
        self._kernel_orig = self.emb.kernel
        # Save original orbitals
        self._mo_orig = self.mf.mo_coeff
        # Output
        self.converged = False
        self.energies = []          # Total energy per iteration

    @property
    def log(self):
        return self.emb.log

    @property
    def mf(self):
        return self.emb.mf

    @property
    def e_tot(self):
        return self.energies[-1]

    @property
    def e_tot_oneshot(self):
        return self.energies[0]

    def get_diis(self):
        return pyscf.lib.diis.DIIS()

    @property
    def kernel_orig(self):
        """Original kernel of embedding method."""
        return self._kernel_orig

    def update_mo_coeff(self, mf, diis=None):
        """Get new set of MO coefficients.

        Must be implemented for any SCMF method."""
        raise AbstractMethodError()

    def check_convergence(self, e_tot, dm1, e_last=None, dm1_last=None, etol=None, dtol=None):
        if etol is None: etol = self.etol
        if dtol is None: dtol = self.dtol
        if e_last is not None:
            de = (e_tot - e_last)
            # RHF:
            if self.emb.is_rhf:
                ddm = abs(dm1-dm1_last).max() / 2
            else:
            # UHF:
                ddm = max(abs(dm1[0]-dm1_last[0]).max(),
                          abs(dm1[1]-dm1_last[1]).max())
        else:
            de = ddm = np.inf
        tighten = (1-self.damping)
        if (abs(de) < tighten*etol) and (ddm < tighten*dtol):
            return True, de, ddm
        return False, de, ddm

    def kernel(self, *args, **kwargs):
        diis = (self.get_diis() if self.diis else None)

        e_last = dm1_last = None
        for self.iteration in range(1, self.maxiter+1):

            self.log.info("%s iteration %3d", self.name, self.iteration)
            self.log.info("%s==============", len(self.name)*"=")

            if self.iteration > 1:
                self.emb.reset()

            # Run clusters, save results
            res = self.kernel_orig(*args, **kwargs)
            e_mf = self.mf.e_tot
            e_corr = self.emb.get_e_corr()
            e_tot = (e_mf + e_corr)
            self.energies.append(e_tot)

            # Update MF
            mo_coeff = self.update_mo_coeff(self.mf, diis=diis)
            self.emb.update_mf(mo_coeff)

            dm1 = self.mf.make_rdm1()
            # Check symmetry
            self.emb.check_fragment_symmetry(dm1)

            # Check convergence
            conv, de, ddm = self.check_convergence(e_tot, dm1, e_last, dm1_last)
            fmt = "%s iteration %3d (dE= %s  dDM= %9.3e): E(MF)= %s  E(corr)= %s  E(tot)= %s"
            estr = energy_string
            self.log.output(fmt, self.name, self.iteration, estr(de), ddm, estr(e_mf), estr(e_corr), estr(e_tot))
            if conv:
                self.log.info("%s converged in %d iterations", self.name, self.iteration)
                self.converged = True
                break
            e_last, dm1_last = e_tot, dm1

        else:
            self.log.warning("%s did not converge in %d iterations!", self.name, self.iteration)
        return res
