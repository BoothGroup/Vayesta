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
from .solver import ClusterSolver


class EBFCISolver(ClusterSolver):
    """[TODO] rewrite this as a subclass of FCISolver? What benefits does this provide, given most functionality is
    in kernel method? Maybe could abstract """

    @dataclasses.dataclass
    class Options(ClusterSolver.Options):
        threads: int = 1
        lindep: float = None
        conv_tol: float = None
        make_rdm_ladder: bool = True

    @dataclasses.dataclass
    class Results(ClusterSolver.Results):
        # CI coefficients
        c0: float = None
        c1: np.array = None
        c2: np.array = None
        rdm_eb: np.array = None

    def __init__(self, freqs, couplings, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.bos_freqs = freqs
        self.eb_coupling = couplings

    @property
    def nbos(self):
        return len(self.bos_freqs)

    def kernel(self, bos_occ_cutoff=None, eris=None):
        """Run FCI kernel."""

        if bos_occ_cutoff is None:
            bos_occ_cutoff = self.fragment.opts.bos_occ_cutoff

        c_act = self.mo_coeff[:,self.get_active_slice()]

        if eris is None:
            # Temporary implementation
            import pyscf.ao2mo
            t0 = timer()
            eris = pyscf.ao2mo.full(self.mf._eri, c_act, compact=False).reshape(4*[self.nactive])
            self.log.timing("Time for AO->MO of (ij|kl):  %s", time_string(timer()-t0))

        nocc = self.nocc - self.nocc_frozen
        occ = np.s_[:nocc]
        vir = np.s_[nocc:]

        f_act = np.linalg.multi_dot((c_act.T, self.base.get_fock(), c_act))
        v_act = 2*einsum('iipq->pq', eris[occ,occ]) - einsum('iqpi->pq', eris[occ,:,:,occ])
        h_eff = f_act - v_act
        # This should be equivalent to:
        #core = np.s_[:self.nocc_frozen]
        #dm_core = 2*np.dot(self.mo_coeff[:,core], self.mo_coeff[:,core].T)
        #v_core = self.mf.get_veff(dm=dm_core)
        #h_eff = np.linalg.multi_dot((c_act.T, self.base.get_hcore()+v_core, c_act))

        nelec = sum(self.mo_occ[self.get_active_slice()])

        if self.opts.conv_tol is not None:
            conv_tol = self.opts.conv_tol
        else:
            conv_tol = 1e-12

        t0 = timer()

        self.log.info("Running FCI with boson occupation cutoff of %d", bos_occ_cutoff)

        e_fci, wf = ebfci_slow.kernel(h_eff, eris, self.eb_coupling, np.diag(self.bos_freqs), self.nactive, nelec,
                        self.nbos, bos_occ_cutoff, tol=conv_tol)

        # For now assuming good convergence, to avoid interface difference between davidson and davidson1.
        self.log.debug("FCI done")#. converged: %r", fcisolver.converged)
        #if not fcisolver.converged:
        #    self.log.error("FCI not converged!")
        self.log.timing("Time for FCI: %s", time_string(timer()-t0))
        # Decomposition not yet set up for coupled electron-boson systems.
        #cisdvec = pyscf.ci.cisd.from_fcivec(wf, self.nactive, nelec)
        #c0, c1, c2 = pyscf.ci.cisd.cisdvec_to_amplitudes(cisdvec, self.nactive, nocc)

        results = self.Results(
                converged=True, e_corr=e_fci, c_occ=self.c_active_occ, c_vir=self.c_active_vir)
                #c0=c0, c1=c1, c2=c2)
        # Grab all required dms.
        if self.opts.make_rdm2:
            results.dm1, results.dm2 = ebfci_slow.make_rdm12e(wf, self.nactive, nelec)
        elif self.opts.make_rdm1:
            results.dm1 = ebfci_slow.make_rdm1e(wf, self.nactive, nelec)

        if self.opts.make_rdm_ladder:
            # For now, generate spin-integrated DM as this is what we'll get from FCIQMC.
            results.rdm_eb = 2 * ebfci_slow.make_eb_rdm(wf, self.nactive, nelec, self.nbos, bos_occ_cutoff)[::2,::2]

        return results
