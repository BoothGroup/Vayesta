import dataclasses
import gc

import numpy as np

import pyscf
import pyscf.lib
import pyscf.agf2

from vayesta.ewf.fragment import EWFFragment
from vayesta.core.util import OptionsBase, NotSet, get_used_memory, time_string
from vayesta.core import QEmbeddingFragment
from vayesta.eagf2 import ragf2, ewdmet_bath

try:
    from mpi4py import MPI
    timer = MPI.Wtime
except ImportError:
    from timeit import default_timer as timer


@dataclasses.dataclass
class EAGF2FragmentOptions(OptionsBase):
    ''' Options for EAGF2 fragments
    '''

    # --- Bath settings
    bath_type: str = NotSet
    nmom_bath: int = NotSet
    bno_threshold: float = NotSet
    bno_threshold_factor: float = NotSet
    dmet_threshold: float = 1e-4
    ewdmet_threshold: float = 1e-4

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
    n_active: int = None
    converged: bool = None
    c_frozen: np.ndarray = None
    c_active: np.ndarray = None
    e_corr: float = None
    e_1b: float = None
    e_2b: float = None
    ip: float = None
    ea: float = None
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
    project_amplitude_to_fragment = EWFFragment.project_amplitude_to_fragment
    project_amplitudes_to_fragment = EWFFragment.project_amplitudes_to_fragment


    #TODO remove side effects?
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
            s_bath = np.linalg.multi_dot((c_bath[:, i].T.conj(), s, c_bath[:, [i]]))
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


    #FIXME: make API consistent with EWF
    def make_dmet_mp2_bath(self):
        ''' Make DMET + MP2 BNO bath orbitals
        '''

        t0 = timer()

        self.log.info("Making DMET+MP2 bath")
        self.log.info("*******************")
        self.log.changeIndentLevel(1)

        self.c_cluster_occ, self.c_cluster_vir, \
                c_no_occ, n_no_occ, c_no_vir, n_no_vir = EWFFragment.make_bath(self)

        self.log.info("Making occupied BNO bath")
        self.log.info("------------------------")
        c_nbo_occ, c_env_occ = \
                self.truncate_bno(c_no_occ, n_no_occ, self.opts.bno_threshold)

        self.log.info("Making virtual BNO bath")
        self.log.info("-----------------------")
        c_nbo_vir, c_env_vir = \
                self.truncate_bno(c_no_vir, n_no_vir, self.opts.bno_threshold)

        self.c_env_occ = c_env_occ
        self.c_env_vir = c_env_vir
        self.c_cluster_occ = self.canonicalize_mo(self.c_cluster_occ, c_nbo_occ)[0]
        self.c_cluster_vir = self.canonicalize_mo(self.c_cluster_vir, c_nbo_vir)[0]

        self.log.timing("Time for DMET+MP2 bath:  %s", time_string(timer() - t0))
        self.log.changeIndentLevel(-1)


    def make_bath(self):
        ''' Make bath orbitals
        '''

        if self.opts.bath_type.upper() in ['EWDMET', 'POWER']:
            return self.make_ewdmet_bath()
        else:
            return self.make_dmet_mp2_bath()


    def project_to_fragment(self, cluster_solver, mo_coeff):
        ''' Project quantities back onto the fragment space
        '''

        rdm1 = cluster_solver.make_rdm1(with_frozen=False)
        se = cluster_solver.se
        ovlp = self.mf.get_ovlp()

        #TODO move democratic partitioning to external function
        c = pyscf.lib.einsum('pa,pq,qi->ai', mo_coeff.conj(), ovlp, self.c_frag)
        p_frag = np.dot(c, c.T.conj())

        c_full = np.hstack((self.c_frag, self.c_env))
        c = pyscf.lib.einsum('pa,pq,qi->ai', mo_coeff.conj(), ovlp, c_full)
        p_full = np.dot(c, c.T.conj())

        def democratic_part(matrix):
            m = pyscf.lib.einsum('...pq,pi,qj->...ij', matrix, p_frag, p_full)
            m = 0.5 * (m + m.swapaxes(m.ndim-1, m.ndim-2).conj())
            return m

        rdm1 = democratic_part(rdm1)
        t_occ = democratic_part(se.get_occupied().moment([0, 1], squeeze=False))
        t_vir = democratic_part(se.get_virtual().moment([0, 1], squeeze=False))
        #TODO higher moments?

        return rdm1, t_occ, t_vir


    def kernel(self, eris=None):
        ''' Run solver for a single BNO threshold
        '''

        if self.c_cluster_occ is None:
            self.make_bath()

        c_occ = np.hstack((self.c_env_occ, self.c_cluster_occ))
        c_vir = np.hstack((self.c_cluster_vir, self.c_env_vir))
        nactive = self.c_cluster_occ.shape[-1] + self.c_cluster_vir.shape[-1]
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

        mo_energy, mo_coeff = self.mf.canonicalize(mo_coeff, mo_occ)
        c_active = mo_coeff[:, list(range(nocc_frozen, nocc+nvir-nvir_frozen))]
        c_frozen = mo_coeff[:, list(range(nocc_frozen)) + 
                               list(range(nocc+nvir-nvir_frozen, nocc+nvir))]

        # Get ERIs
        if eris is None:
            eri = pyscf.ao2mo.incore.full(self.mf._eri, c_active, compact=False)
            eri = eri.reshape((c_active.shape[1],) * 4)

        # Run solver
        cluster_solver = self.solver(
                self.mf,
                mo_energy=mo_energy,
                mo_coeff=mo_coeff,
                mo_occ=mo_occ,
                frozen=(nocc_frozen, nvir_frozen),
                log=self.base.quiet_log,
                eri=eri,
                dump_chkfile=False,
                options=self.opts.solver_options,
        )
        cluster_solver.kernel()

        #FIXME FIXME FIXME
        #NOTE: AGF2 solver on each cluster must end on a Fock loop in order
        #      that the final cluster Fock loop returns the AGF2 result.
        #      This makes the results on each cluster look slightly wrong
        #      in the instance of a complete bath, but the correct result
        #      is recovered after partitioning.
        #
        # For example, the following gives different IPs:
        #    gf2 = ragf2.RAGF2(mf).run()
        #    print(gf2.gf.get_occupied().energy.max())
        #    w, v = gf2.solve_dyson(se=gf2.se, gf=gf2.gf, fock=gf2.get_fock())
        #    gf2.gf = gf2.gf.__class__(w, v[:gf2.nmo])
        #    gf2.gf, gf2.se = gf2.fock_loop()
        #    print(gf2.gf.get_occupied().energy.max())
        #
        # But the following is correct:
        #    gf2 = ragf2.RAGF2(mf).run()
        #    print(gf2.gf.get_occupied().energy.max())
        #    gf2.gf, gf2.se = gf2.fock_loop()
        #    w, v = gf2.solve_dyson(se=gf2.se, gf=gf2.gf, fock=gf2.get_fock())
        #    gf2.gf = gf2.gf.__class__(w, v[:gf2.nmo])
        #    gf2.gf, gf2.se = gf2.fock_loop()
        #    print(gf2.gf.get_occupied().energy.max())
        cluster_solver.gf, cluster_solver.se = cluster_solver.fock_loop()

        e_corr = cluster_solver.e_corr
        rdm1, t_occ, t_vir = self.project_to_fragment(cluster_solver, c_active)

        results = EAGF2FragmentResults(
                fid=self.id,
                n_active=nactive,
                converged=cluster_solver.converged,
                c_frozen=c_frozen,
                c_active=c_active,
                e_corr=e_corr,
                e_1b=cluster_solver.e_1b,
                e_2b=cluster_solver.e_2b,
                ip=cluster_solver.e_ip,
                ea=cluster_solver.e_ea,
                rdm1=rdm1,
                t_occ=t_occ,
                t_vir=t_vir,
        )

        self.results = results

        # Force GC to free memory
        m0 = get_used_memory()
        del cluster_solver
        ndel = gc.collect()
        self.log.debugv("GC deleted %d objects and freed %.3f MB of memory",
                        ndel, (get_used_memory()-m0)/1e6)

        return results
