import logging
import dataclasses
import gc

import numpy as np

import pyscf
import pyscf.lib
import pyscf.agf2

from vayesta.ewf.fragment import EWFFragment
from vayesta.core.util import Options, NotSet, get_used_memory, time_string
from vayesta.core import QEmbeddingFragment
from vayesta.agf2 import ragf2, ewdmet_bath, util

try:
    from mpi4py import MPI
    timer = MPI.Wtime
except ImportError:
    from timeit import default_timer as timer


@dataclasses.dataclass
class EAGF2FragmentOptions(Options):
    ''' Options for EAGF2 fragments
    '''

    # --- Bath settings
    ewdmet: bool = NotSet
    nmom_bath: int = NotSet
    bno_threshold: float = NotSet
    bno_threshold_factor: float = NotSet
    dmet_threshold: float = 1e-4
    ewdmet_threshold: float = 1e-4

    # --- Solver settings
    solver_options: dict = NotSet

    # --- Appease EWF inheritance
    plot_orbitals: bool = False
    

@dataclasses.dataclass
class EAGF2FragmentResults:
    ''' Results for EAGF2 fragments
    '''

    fid: int = None
    n_active: int = None
    converged: bool = None
    c_frozen: np.ndarray = None
    c_active: np.ndarray = None
    e_corr: float = None
    rdm1: np.ndarray = None
    t_occ: np.ndarray = None
    t_vir: np.ndarray = None


class EAGF2Fragment(QEmbeddingFragment):

    def __init__(self, base, fid, name, c_frag, c_env, fragment_type, sym_factor=1,
            atoms=None, log=None, options=None, **kwargs):

        super().__init__(
                base, fid, name, c_frag, c_env, fragment_type, 
                sym_factor=sym_factor, atoms=atoms, log=log,
        )

        self.opts = options
        if self.opts is None:
            self.opts = EAGF2FragmentOptions(**kwargs)
        self.opts = self.opts.replace(self.base.opts, select=NotSet)
        for key, val in self.opts.items():
            self.log.infov("  > %-24s %r", key + ':', val)
        self.solver = ragf2.RAGF2
        self.log.infov("  > %-24s %r", 'Solver:', self.solver)

        self.c_env_occ = None
        self.c_env_vir = None
        self.c_act_occ = None
        self.c_act_occ = None

        self.results = None


    @property
    def e_corr(self):
        idx = np.argmin(self.bno_threshold)
        return self.e_corrs[idx]


    make_dmet_bath = EWFFragment.make_dmet_bath
    make_bno_bath = EWFFragment.make_bno_bath
    truncate_bno = EWFFragment.truncate_bno


    def make_ewdmet_bath(self):
        ''' Make EwDMET bath orbitals
        '''

        t0 = timer()

        self.log.info("Making EwDMET bath")
        self.log.info("******************")
        self.log.changeIndentLevel(1)

        c_bath, c_env_occ, c_env_vir = ewdmet_bath.make_ewdmet_bath(
                self,
                self.c_env,
                nmom=self.opts.nmom_bath,
        )

        self.log.info("EwDMET bath character:")
        s = self.mf.get_ovlp()
        for i in range(c_bath.shape[-1]):
            s_bath = np.linalg.multi_dot((c_bath[:,i].T.conj(), s, c_bath[:,[i]]))
            arg = np.argsort(-s_bath)
            s_bath = s_bath[arg]
            n = np.amin((len(s_bath), 6))
            labels = np.asarray(self.mol.ao_labels())[arg][:n]
            lines = [("%s = %.5f" % (labels[i].strip(), s_bath[i])) for i in range(n)]
            self.log.info("  > %2d:  %s", i+1, '  '.join(lines))

        self.log.timing("Time for EwDMET bath:  %s", time_string(timer() - t0))
        self.log.changeIndentLevel(-1)

        self.c_env_occ, self.c_env_vir = c_env_occ, c_env_vir
        self.c_cluster_occ, self.c_cluster_vir = self.diagonalize_cluster_dm(
                self.c_frag,
                c_bath,
                tol=2*self.opts.ewdmet_threshold,
        )
        self.log.info(
                "Cluster orbitals:  n(occ) = %d  n(vir) = %d",
                self.c_cluster_occ.shape[-1], self.c_cluster_vir.shape[-1],
        )


    def make_bath(self):
        ''' Make bath orbitals
        '''

        if self.opts.ewdmet:
            return self.make_ewdmet_bath()
        else:
            #TODO make consistent with EWF API
            EWFFragment.make_bath(self)

            c_nbo_occ, c_env_occ = \
                    self.truncate_bno(self.c_no_occ, self.n_no_occ, self.opts.bno_threshold)
            c_nbo_vir, c_env_vir = \
                    self.truncate_bno(self.c_no_vir, self.n_no_vir, self.opts.bno_threshold)

            self.c_env_occ = c_env_occ
            self.c_env_vir = c_env_vir
            self.c_cluster_occ = self.canonicalize_mo(self.c_cluster_occ, c_nbo_occ)[0]
            self.c_cluster_vir = self.canonicalize_mo(self.c_cluster_vir, c_nbo_vir)[0]


    def project_to_fragment(self, cluster_solver, mo_coeff):
        ''' Project quantities back onto the fragment space
        '''

        rdm1 = cluster_solver.make_rdm1(with_frozen=False)
        se = cluster_solver.se
        ovlp = self.mf.get_ovlp()

        def democratic_part(frag, matrix, mo_coeff):
            #TODO move to external function
            c = pyscf.lib.einsum('pa,pq,qi->ai', mo_coeff.conj(), ovlp, frag.c_frag)
            p_frag = np.dot(c, c.T.conj())

            c_full = np.hstack((frag.c_frag, frag.c_env))
            c = pyscf.lib.einsum('pa,pq,qi->ai', mo_coeff.conj(), ovlp, c_full)
            p_full = np.dot(c, c.T.conj())

            m = pyscf.lib.einsum('...pq,pi,qj->...ij', matrix, p_frag, p_full)
            m = 0.5 * (m + m.swapaxes(m.ndim-1, m.ndim-2).conj())

            return m

        rdm1 = democratic_part(self, rdm1, mo_coeff)
        t_occ = democratic_part(self, se.get_occupied().moment([0, 1], squeeze=False), mo_coeff)
        t_vir = democratic_part(self, se.get_virtual().moment([0, 1], squeeze=False), mo_coeff)
        #TODO higher moments?

        return rdm1, t_occ, t_vir


    def kernel(self, eris=None):
        ''' Run solver for a single BNO threshold
        '''

        if self.c_cluster_occ is None:
            self.make_bath()

        c_active = np.hstack((self.c_cluster_occ, self.c_cluster_vir))
        c_frozen = np.hstack((self.c_env_occ, self.c_env_vir))
        c_occ = np.hstack((self.c_env_occ, self.c_cluster_occ))
        c_vir = np.hstack((self.c_cluster_vir, self.c_env_vir))
        nactive = c_active.shape[-1]
        nocc, nvir = c_occ.shape[-1], c_vir.shape[-1]
        nocc_frozen = self.c_env_occ.shape[-1]
        nvir_frozen = self.c_env_vir.shape[-1]
        mo_coeff = np.hstack((c_occ, c_vir))

        # Check occupations
        #FIXME for EwDMET too ye?
        n_occ = self.get_mo_occupation(c_occ)
        if not np.allclose(n_occ, 2, atol=2*self.opts.dmet_threshold):
            raise RuntimeError("Incorrect occupation of occupied orbitals:\n%r" % n_occ)
        n_vir = self.get_mo_occupation(c_vir)
        if not np.allclose(n_vir, 0, atol=2*self.opts.dmet_threshold):
            raise RuntimeError("Incorrect occupation of virtual orbitals:\n%r" % n_vir)
        mo_occ = np.asarray(nocc*[2] + nvir*[0])
        mo_energy = self.mf.canonicalize(mo_coeff, mo_occ)[0]

        # Get ERIs
        if eris is None:
            #TODO re-use for self-consistency
            eri = pyscf.ao2mo.incore.full(self.mf._eri, c_active, compact=False)
            eri = eri.reshape((c_active.shape[1],) * 4)

        ## Get Veff due to frozen density
        #rdm1 = np.dot(c_frozen_occ, c_frozen_occ.T.conj()) * 2
        #veff = self.mf.get_veff(dm=rdm1)
        #veff = np.linalg.multi_dot((mo_coeff.T.conj(), veff, mo_coeff))

        ## Get the MO energies
        #rdm1 = np.dot(c_occ, c_occ.T.conj()) * 2
        #assert np.allclose(self.mf.make_rdm1(), rdm1)  # should be the same
        #fock = self.mf.get_fock(dm=rdm1)
        #fock = np.linalg.multi_dot((mo_coeff.T.conj(), fock, mo_coeff))
        #mo_energy, r = np.linalg.eigh(fock)

        # Run solver
        cluster_solver = self.solver(
                self.mf,
                mo_energy=mo_energy,
                mo_coeff=mo_coeff,
                mo_occ=mo_occ,
                frozen=(nocc_frozen, nvir_frozen),
                log=self.base.quiet_log,
                eri=eri,
                #veff=veff,
                dump_chkfile=False,
                **self.opts.solver_options,
        )
        cluster_solver.kernel()
        #TODO: brief output of results on current cluster

        e_corr = cluster_solver.e_corr
        rdm1, t_occ, t_vir = self.project_to_fragment(cluster_solver, c_active)

        results = EAGF2FragmentResults(
                fid=self.id,
                n_active=nactive,
                converged=cluster_solver.converged,
                c_frozen=c_frozen,
                c_active=c_active,
                e_corr=e_corr,
                rdm1=rdm1,
                t_occ=t_occ,
                t_vir=t_vir,
        )

        self.results = results

        # Force GC to free memory
        m0 = get_used_memory()
        del cluster_solver
        ndel = gc.collect()
        self.log.debugv("GC deleted %d objects and freed %.3f MB of memory", ndel, (get_used_memory()-m0)/1e6)

        return results
