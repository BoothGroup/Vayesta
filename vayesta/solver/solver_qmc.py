import dataclasses
from timeit import default_timer as timer

import numpy as np

import pyscf
import pyscf.ao2mo
import pyscf.ci
import pyscf.mcscf
import pyscf.fci
from pyscf import tools, ao2mo

#import pyscf.fci.direct_spin0
#import pyscf.fci.direct_spin1

from vayesta.core.util import *
from .solver import ClusterSolver

from cisd_coeff import Hamiltonian, RestoredCisdCoeffs
import pickle as pkl


class Hamiltonian:
    '''
    Hamiltonian class for FCIQMC formatting
    '''
    h0 = None
    h1e = None
    eri = None
    nelec = None
    def from_arrays(self, h0, h1e, eri, nelec):
        self.h0, self.h1e, self.eri, self.nelec = h0, h1e, eri, nelec
    def to_pickle(self, fname):
        with open(fname, 'wb') as f: pkl.dump([self.h0, self.h1e, self.eri, self.nelec], f)
    def from_pickle(self, fname):
        with open(fname, 'rb') as f:
            self.h0, self.h1e, self.eri, self.nelec = pkl.load(f)
    def write_fcidump(self, fname='FCIDUMP'):
        '''
        writes the provided integrals to a file in a standard format for FCI programs
        '''
        nsite = self.h1e.shape[0]
        if len(self.eri.shape)!=1:
            # ERIs must have 8-fold symmetry restored
            eri = ao2mo.restore(8, self.eri, nsite)
        else: eri = self.eri
        tools.fcidump.from_integrals(fname, self.h1e, eri, nsite, self.nelec, self.h0, 0, [1,]*nsite)
    def get_fci_energy(self):
        nsite = self.h1e.shape[0]
        return fci.direct_spin1.kernel(self.h1e, self.eri, nsite, self.nelec, verbose=6, ecore=self.h0)[0]


class FCIQMCSolver(ClusterSolver):


    class Options(ClusterSolver.Options):
        threads: int = 1
        lindep: float = None
        conv_tol: float = None


    @dataclasses.dataclass
    class Results(ClusterSolver.Results):
        # CI coefficients
        c0: float = None
        c1: np.array = None
        c2: np.array = None


    def kernel(self, init_guess=None, eris=None):
        """Run FCI kernel."""

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

        fcisolver = pyscf.fci.direct_spin1.FCISolver(self.mol)
        if self.opts.threads is not None: fcisolver.threads = self.opts.threads
        if self.opts.conv_tol is not None: fcisolver.conv_tol = self.opts.conv_tol
        if self.opts.lindep is not None: fcisolver.lindep = self.opts.lindep

        nelec = sum(self.mo_occ[self.get_active_slice()])
        
        h0 = 0.0 # No 0-electron energy for lattice models
        
        # Once Hamiltonian's are setup, write Hamiltonians in FCIQMC readable format
        
        qmc_H = Hamiltonian()
        qmc_H.from_arrays(h0, h_eff, eris, nelec)
        e_fci_fciqmc = qmc_H.get_fci_energy()
        
    
        qmc_H.to_pickle('Hubbard_Hamiltonian_cluster%1d.pkl'%(self.fragment.id))
        qmc_H.write_fcidump( fname='FCIDUMP_cluster%1d'%(self.fragment.id))
        
    
        t0 = timer()
        e_fci, wf = fcisolver.kernel(h_eff, eris, self.nactive, nelec)
        self.log.debug("FCI done. converged: %r", fcisolver.converged)
        if not fcisolver.converged:
            self.log.error("FCI not converged!")
        self.log.timing("Time for FCI: %s", time_string(timer()-t0))
        # TODO: This requires the E_core energy (and nuc-nuc repulsion)
        e_corr = np.nan

        cisdvec = pyscf.ci.cisd.from_fcivec(wf, self.nactive, nelec)
        c0, c1, c2 = pyscf.ci.cisd.cisdvec_to_amplitudes(cisdvec, self.nactive, nocc)
        
        print(e_fci, e_fci_fciqmc)
        assert 0
        assert e_fci == e_fci_fciqmc
        
        
        results = self.Results(
                converged=fcisolver.converged, e_corr=e_corr, c_occ=self.c_active_occ, c_vir=self.c_active_vir, eris=eris,
                c0=c0, c1=c1, c2=c2)

        if self.opts.make_rdm2:
            results.dm1, results.dm2 = fcisolver.make_rdm12(wf, self.nactive, nelec)
        elif self.opts.make_rdm1:
            results.dm1 = fcisolver.make_rdm1(wf, self.nactive, nelec)
        
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
