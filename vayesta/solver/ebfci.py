from .eb_fci import ebfci_slow

import dataclasses
from timeit import default_timer as timer

import numpy as np

import pyscf
import pyscf.ao2mo
import pyscf.ci
import pyscf.mcscf
import pyscf.fci
#import pyscf.fci.direct_spin0
#import pyscf.fci.direct_spin1

from vayesta.core.util import *
from vayesta.solver.fci import FCI_Solver


class EBFCI_Solver(FCI_Solver):

    @dataclasses.dataclass
    class Options(FCI_Solver.Options):
        max_boson_occ: int = NotSet
        make_rdm_eb: bool = True
        make_01_dd_mom: bool = False

    @dataclasses.dataclass
    class Results(FCI_Solver.Results):
        # CI coefficients
        c0: float = None
        c1: np.array = None
        c2: np.array = None
        rdm_eb: np.array = None
        dd_mom0: np.array = None
        dd_mom1: np.array = None

    def __init__(self, freqs, couplings, *args, **kwargs):
        # This sets some things we don't care about, but they shouldn't cause issues.
        super().__init__(*args, **kwargs)
        self.bos_freqs = freqs
        self.eb_coupling = couplings

    @property
    def nbos(self):
        return len(self.bos_freqs)

    def kernel(self, eris=None):
        """Run FCI kernel."""

        if eris is None: eris = self.get_eris()
        heff = self.get_heff(eris)
        nelec = sum(self.mo_occ[self.get_active_slice()])

        t0 = timer()

        nocc = self.nocc - self.nocc_frozen
        occ = np.s_[:nocc]
        vir = np.s_[nocc:]

        if self.opts.conv_tol is not None:
            conv_tol = self.opts.conv_tol
        else:
            conv_tol = 1e-12

        t0 = timer()

        e_fci, civec = ebfci_slow.kernel(heff, eris, self.eb_coupling, np.diag(self.bos_freqs), self.nactive, nelec,
                        self.nbos, self.opts.bos_occ_cutoff, tol=conv_tol)

        # For now assuming good convergence, to avoid interface difference between davidson and davidson1.
        self.log.debug("FCI done")#. converged: %r", fcisolver.converged)
        #if not fcisolver.converged:
        #    self.log.error("FCI not converged!")
        self.log.timing("Time for FCI: %s", time_string(timer()-t0))
        # Decomposition not yet set up for coupled electron-boson systems.
        #cisdvec = pyscf.ci.cisd.from_fcivec(wf, self.nactive, nelec)
        #c0, c1, c2 = pyscf.ci.cisd.cisdvec_to_amplitudes(cisdvec, self.nactive, nocc)

        results = self.Results(
                converged=True, e_corr=e_fci, c_occ=self.c_active_occ, c_vir=self.c_active_vir, civec=civec)
                #c0=c0, c1=c1, c2=c2)
        # Grab all required dms.
        if self.opts.make_01_dd_mom:
            results.dm1, results.dm2 = ebfci_slow.make_rdm12e(civec, self.nactive, nelec)
            # Calculating only the components of the dd response moments we needs cuts down on calculation time.
            frag_coeffs = np.linalg.multi_dot([self.fragment.c_active.T, self.base.get_ovlp(), self.fragment.c_frag])
            dd_moms = ebfci_slow.calc_dd_resp_mom(civec, e_fci, 1, self.nactive, nelec,
                                                  self.nbos, heff, eris, np.diag(self.bos_freqs), self.eb_coupling,
                                                  self.opts.bos_occ_cutoff, results.dm1, trace = False,
                                                  coeffs = frag_coeffs)
            results.dd_mom0 = dd_moms[0]
            results.dd_mom1 = dd_moms[1]
        elif self.opts.make_rdm2:
            results.dm1, results.dm2 = ebfci_slow.make_rdm12e(civec, self.nactive, nelec)
        elif self.opts.make_rdm1:
            results.dm1 = ebfci_slow.make_rdm1e(civec, self.nactive, nelec)

        if self.opts.make_rdm_eb:
            # For now, generate spin-integrated DM as this is what we'll get from FCIQMC.
            rdm_eb = ebfci_slow.make_eb_rdm(
                civec, self.nactive, nelec, self.nbos, self.opts.bos_occ_cutoff)
            results.rdm_eb = rdm_eb[::2,::2] + rdm_eb[1::2,1::2]
        return results
