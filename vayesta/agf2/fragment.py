import logging
import dataclasses
import gc

import numpy as np

import pyscf
import pyscf.agf2

from vayesta.ewf.fragment import EWFFragment
from vayesta.core.util import Options, NotSet, get_used_memory, time_string
from vayesta.core import QEmbeddingFragment
from vayesta.agf2 import RAGF2, make_ewdmet_bath, util

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
    dmet_threshold: float = NotSet
    ewdmet_threshold: float = NotSet
    bath_type: str = NotSet
    nmom_bath: int = NotSet

    # --- Solver settings
    solver_options: dict = NotSet

    # --- Appease EWF inheritance
    plot_orbitals: bool = False
    energy_partitioning: str = 'first-occ'
    

@dataclasses.dataclass
class EAGF2FragmentResults:
    ''' Results for EAGF2 fragments
    '''

    fid: int = None
    bno_threshold: float = None
    n_active: int = None
    converged: bool = None
    c_frozen: np.ndarray = None
    c_active: np.ndarray = None
    e_corr: float = None
    fock: np.ndarray = None
    se: pyscf.agf2.SelfEnergy = None


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
        self.solver = RAGF2
        self.log.infov("  > %-24s %r", 'Solver:', self.solver)

        # DMET-cluster
        self.c_cluster_occ = None
        self.c_cluster_vir = None

        # EwDMET
        self.c_env_occ = None
        self.c_env_vir = None

        # BNO
        self.c_no_occ = None
        self.c_no_vir = None
        self.n_no_occ = None
        self.n_no_vir = None

        # Active
        self.c_active_occ = None
        self.c_active_vir = None

        self.results = None


    @property
    def e_corr(self):
        idx = np.argmin(self.bno_threshold)
        return self.e_corrs[idx]


    make_dmet_bath = EWFFragment.make_dmet_bath
    project_amplitude_to_fragment = EWFFragment.project_amplitude_to_fragment
    project_amplitudes_to_fragment = EWFFragment.project_amplitudes_to_fragment
    apply_bno_threshold = EWFFragment.apply_bno_threshold


    def make_ewdmet_bath(self):
        ''' Make EwDMET bath orbitals
        '''

        t0 = timer()

        self.log.info("Making EwDMET bath")
        self.log.info("******************")
        self.log.changeIndentLevel(1)

        c_ewdmet, c_env_occ, c_env_vir = make_ewdmet_bath(
                self,
                self.c_env,
                nmom=self.opts.nmom_bath,
        )

        self.log.info("EwDMET bath character:")
        s = self.mf.get_ovlp()
        for i in range(c_ewdmet.shape[-1]):
            s_bath = np.linalg.multi_dot((c_ewdmet[:,i].T.conj(), s, c_ewdmet[:,[i]]))
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
                c_ewdmet,
                tol=2*self.opts.ewdmet_threshold,
        )
        self.log.info(
                "Cluster orbitals:  n(occ) = %d  n(vir) = %d",
                self.c_cluster_occ.shape[-1], self.c_cluster_vir.shape[-1],
        )


    def make_bath(self):
        ''' Make bath orbitals
        '''

        if self.opts.bath_type.lower() == 'dmet+mp2':
            return EWFFragment.make_bath(self)
        elif self.opts.bath_type.lower() == 'ewdmet':
            return self.make_ewdmet_bath()


    #def project_to_fragment(self, cluster_solver, mo_coeff):
    #    ''' Project quantities back onto the fragment space
    #    '''

    #    coeff = np.linalg.multi_dot((self.c_frag.T.conj(), self.mf.get_ovlp(), mo_coeff))
    #    
    #    fock = cluster_solver.get_fock(with_frozen=False)
    #    fock = np.linalg.multi_dot((coeff, fock, coeff.T.conj()))

    #    energy = cluster_solver.se.energy
    #    coupling = np.dot(coeff, cluster_solver.se.coupling)
    #    se = pyscf.agf2.SelfEnergy(energy, coupling, chempot=cluster_solver.se.chempot)

    #    return fock, se


    def get_coeffs(self, bno_threshold=None):
        ''' Get the active and frozen space coefficients
        '''

        if self.opts.bath_type.lower() == 'dmet+mp2':
            self.log.info("Occupied BNOs:")
            c_nbo_occ, c_frozen_occ = self.apply_bno_threshold(self.c_no_occ, self.n_no_occ, bno_threshold)
            self.log.info("Virtual BNOs:")
            c_nbo_vir, c_frozen_vir = self.apply_bno_threshold(self.c_no_vir, self.n_no_vir, bno_threshold)

            c_active_occ = self.canonicalize_mo(self.c_cluster_occ, c_nbo_occ)[0]
            c_active_vir = self.canonicalize_mo(self.c_cluster_vir, c_nbo_vir)[0]

        elif self.opts.bath_type.lower() == 'ewdmet':
            c_active_occ = self.c_cluster_occ
            c_active_vir = self.c_cluster_vir
            c_frozen_occ = self.c_env_occ
            c_frozen_vir = self.c_env_vir

        return c_active_occ, c_frozen_occ, c_active_vir, c_frozen_vir


    def kernel(self, bno_threshold=None, eris=None):
        ''' Run solver for a single BNO threshold
        '''

        if self.c_cluster_occ is None:
            self.make_bath()

        c_active_occ, c_frozen_occ, c_active_vir, c_frozen_vir = self.get_coeffs(bno_threshold)
        c_active = np.hstack((c_active_occ, c_active_vir))
        c_frozen = np.hstack((c_frozen_occ, c_frozen_vir))
        c_occ = np.hstack((c_frozen_occ, c_active_occ))
        c_vir = np.hstack((c_active_vir, c_frozen_vir))
        nactive = c_active.shape[-1]
        nocc, nvir = c_occ.shape[-1], c_vir.shape[-1]
        nocc_frozen = c_frozen_occ.shape[-1]
        nvir_frozen = c_frozen_vir.shape[-1]
        mo_coeff = np.hstack((c_occ, c_vir))

        # Check occupations
        if self.opts.bath_type == 'dmet+mp2':
            n_occ = self.get_mo_occupation(c_occ)
            if not np.allclose(n_occ, 2, atol=2*self.opts.dmet_threshold):
                raise RuntimeError("Incorrect occupation of occupied orbitals:\n%r" % n_occ)
            n_vir = self.get_mo_occupation(c_vir)
            if not np.allclose(n_vir, 0, atol=2*self.opts.dmet_threshold):
                raise RuntimeError("Incorrect occupation of virtual orbitals:\n%r" % n_vir)
        mo_occ = np.asarray(nocc*[2] + nvir*[0])

        # Get ERIs
        if eris is None:
            #TODO re-use for self-consistency
            eri = pyscf.ao2mo.incore.full(self.mf._eri, c_active, compact=False)
            eri = eri.reshape((c_active.shape[1],) * 4)

        self.c_active_occ = c_active_occ
        self.c_active_vir = c_active_vir

        # Get Veff due to frozen density
        rdm1 = np.dot(c_frozen_occ, c_frozen_occ.T.conj()) * 2
        veff = self.mf.get_veff(dm=rdm1)
        veff = np.linalg.multi_dot((mo_coeff.T.conj(), veff, mo_coeff))

        # Get the MO energies
        rdm1 = np.dot(c_occ, c_occ.T.conj()) * 2
        assert np.allclose(self.mf.make_rdm1(), rdm1)  # should be the same
        fock = self.mf.get_fock(dm=rdm1)
        fock = np.linalg.multi_dot((mo_coeff.T.conj(), fock, mo_coeff))
        mo_energy, r = np.linalg.eigh(fock)

        # Run solver
        cluster_solver = RAGF2(
                self.mf,
                mo_energy=mo_energy,
                mo_coeff=mo_coeff,
                mo_occ=mo_occ,
                frozen=(nocc_frozen, nvir_frozen),
                log=self.log,
                eri=eri,
                veff=veff,
                dump_chkfile=False,
                **self.opts.solver_options,
        )
        cluster_solver.kernel()

        e_corr = cluster_solver.e_corr
        #fock, se = self.project_to_fragment(cluster_solver, c_active)
        fock = cluster_solver.get_fock(with_frozen=False)
        se = cluster_solver.se

        results = EAGF2FragmentResults(
                fid=self.id,
                bno_threshold=bno_threshold,
                n_active=nactive,
                converged=cluster_solver.converged,
                c_frozen=c_frozen,
                c_active=c_active,
                e_corr=e_corr,
                fock=fock,
                se=se,
        )

        self.results = results

        # Force GC to free memory
        m0 = get_used_memory()
        del cluster_solver
        ndel = gc.collect()
        self.log.debugv("GC deleted %d objects and freed %.3f MB of memory", ndel, (get_used_memory()-m0)/1e6)

        return results
