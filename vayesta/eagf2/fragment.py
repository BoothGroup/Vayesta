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
    ''' Options for EAGF2 fragments - see `EAGF2Fragment`.
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
    ''' Results for EAGF2 fragments.

    Attributes
    ----------
    fid : int
        Fragment ID.
    n_active : int
        Number of active orbitals.
    converged : bool
        Whether the cluster calculation converged successfully.
    c_frozen : np.ndarray
        Frozen cluster orbital coefficients.
    c_active : np.ndarray
        Active cluster orbital coefficients.
    e_corr : float
        Correlation energy.
    e_1b : float
        One-body part of total energy, including nuclear repulsion.
    e_2b : float
        Two-body part of total energy.
    ip : float
        Ionisation potential.
    ea : float
        Electron affinity.
    rdm1 : np.ndarray
        Reduced one-body density matrix, democratically partitioned and
        transformed into the MO basis.
    fock : np.ndarray
        Fock matrix, democratically partitioned and transformed into the
        MO basis.
    t_occ : np.ndarray
        Occupied self-energy moments, democratically partitioned and 
        transformed into the into the MO basis.
    t_vir : np.ndarray
        Virtual self-energy moments, democratically partitioned and 
        transformed into the into the MO basis.
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
    fock: np.ndarray = None
    t_occ: np.ndarray = None
    t_vir: np.ndarray = None


class EAGF2Fragment(QEmbeddingFragment):

    def __init__(self, base, fid, name, c_frag, c_env, fragment_type, sym_factor=1,
                 atoms=None, aos=None, log=None, options=None, **kwargs):
        ''' Embedded AGF2 fragment.

        Parameters
        ----------
        base : EAGF2
            Parent `EAGF2` method the fragment is part of.
        fid : int
            Fragment ID.
        name : str
            Name of fragment.
        log : logging.Logger
            Logger object. If None, the logger of the `base` object is
            used (default value is None).
        options : EAGF2FragmentOptions
            Options `dataclass`.
        c_frag : np.ndarray
            Fragment orbital coefficients.
        c_env : np.ndarray
            Environment orbital coefficients.
        fragment_type : {'IAO', 'Lowdin-AO'}
            Fragment orbital type.
        atoms : list or int
            Associated atoms (default value is None).
        aos : list or int
            Associated atomic orbitals (default value is None).
        sym_factor : float
            Symmetry factor (number of symmetry equivalent fragments)
            (default value is 1.0).
        **kwargs : dict, optional
            Additional arguments passed to `EAGF2FragmentOptions`.
        '''

        super().__init__(
                base, fid, name, c_frag, c_env, fragment_type,
                sym_factor=sym_factor, atoms=atoms, aos=aos, log=log,
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
        self.c_cluster_occ = None
        self.c_cluster_vir = None

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


    def make_ewdmet_bath(self):
        ''' Make EwDMET bath orbitals.

        Returns
        -------
        c_cluster_occ : np.ndarray
            Occupied cluster coefficients.
        c_cluster_vir : np.ndarray
            Virtual cluster coefficients.
        c_env_occ : np.ndarray
            Occupied environment coefficients.
        c_env_vir : np.ndarray
            Virtual environment coefficients.
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

        c_env_occ, c_env_vir = c_env_occ, c_env_vir
        c_cluster_occ, c_cluster_vir = self.diagonalize_cluster_dm(
                self.c_frag,
                c_bath,
                tol=2*self.opts.ewdmet_threshold,
        )
        self.log.info(
                "Cluster orbitals:  n(occ) = %d  n(vir) = %d",
                c_cluster_occ.shape[-1], c_cluster_vir.shape[-1],
        )

        return c_cluster_occ, c_cluster_vir, c_env_occ, c_env_vir


    def make_dmet_mp2_bath(self):
        ''' Make DMET + MP2 BNO bath orbitals.

        Returns
        -------
        c_cluster_occ : np.ndarray
            Occupied cluster coefficients.
        c_cluster_vir : np.ndarray
            Virtual cluster coefficients.
        c_env_occ : np.ndarray
            Occupied environment coefficients.
        c_env_vir : np.ndarray
            Virtual environment coefficients.
        '''

        t0 = timer()

        self.log.info("Making DMET+MP2 bath")
        self.log.info("*******************")
        self.log.changeIndentLevel(1)

        c_cluster_occ, c_cluster_vir, \
                c_no_occ, n_no_occ, c_no_vir, n_no_vir = EWFFragment.make_bath(self)

        self.log.info("Making occupied BNO bath")
        self.log.info("------------------------")
        c_nbo_occ, c_env_occ = \
                self.truncate_bno(c_no_occ, n_no_occ, self.opts.bno_threshold)

        self.log.info("Making virtual BNO bath")
        self.log.info("-----------------------")
        c_nbo_vir, c_env_vir = \
                self.truncate_bno(c_no_vir, n_no_vir, self.opts.bno_threshold)

        c_env_occ = c_env_occ
        c_env_vir = c_env_vir
        c_cluster_occ = self.canonicalize_mo(c_cluster_occ, c_nbo_occ)[0]
        c_cluster_vir = self.canonicalize_mo(c_cluster_vir, c_nbo_vir)[0]

        self.log.timing("Time for DMET+MP2 bath:  %s", time_string(timer() - t0))
        self.log.changeIndentLevel(-1)

        return c_cluster_occ, c_cluster_vir, c_env_occ, c_env_vir


    def make_bath(self):
        ''' Make bath orbitals.

        Returns
        -------
        c_cluster_occ : np.ndarray
            Occupied cluster coefficients.
        c_cluster_vir : np.ndarray
            Virtual cluster coefficients.
        c_env_occ : np.ndarray
            Occupied environment coefficients.
        c_env_vir : np.ndarray
            Virtual environment coefficients.
        '''

        if self.opts.bath_type.upper() in ['EWDMET', 'POWER']:
            return self.make_ewdmet_bath()
        else:
            return self.make_dmet_mp2_bath()


    def project_to_fragment(self, cluster_solver, mo_coeff):
        ''' Project quantities back onto the fragment space.

        Parameters
        ----------
        cluster_solver : vayesta.eagf2.ragf2.RAGF2
            RAGF2 cluster solver object.
        mo_coeff : np.ndarray
            MO coefficients corresponding to the active orbitals of the
            basis used in `cluster_solver`.

        Returns
        -------
        rdm1 : np.ndarray
            Reduced one-body density matrix, democratically partitioned and
            transformed into the MO basis.
        fock : np.ndarray
            Fock matrix, democratically partitioned and transformed into the
            MO basis.
        t_occ : np.ndarray
            Occupied self-energy moments, democratically partitioned and 
            transformed into the into the MO basis.
        t_vir : np.ndarray
            Virtual self-energy moments, democratically partitioned and 
            transformed into the into the MO basis.
        '''

        rdm1 = cluster_solver.make_rdm1(with_frozen=False)
        fock = cluster_solver.get_fock(with_frozen=False)
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
        fock = democratic_part(fock)
        t_occ = democratic_part(se.get_occupied().moment([0, 1], squeeze=False))
        t_vir = democratic_part(se.get_virtual().moment([0, 1], squeeze=False))
        #TODO higher moments?

        return rdm1, fock, t_occ, t_vir


    def kernel(self, eris=None):
        ''' Run solver.

        Parameters
        ----------
        eri : np.ndarray
            Four- or three-centre ERI array, if None then calculate
            inside cluster solver (default value is None).

        Returns
        -------
        results : EAGF2FragmentResults
            Object contained results of `EAGF2Fragment`, see 
            `EAGF2FragmentResults` for a list of attributes.
        '''

        if self.c_cluster_occ is None:
            self.c_cluster_occ, self.c_cluster_vir, self.c_env_occ, self.c_env_vir = \
                    self.make_bath()

        c_occ = np.hstack((self.c_env_occ, self.c_cluster_occ))
        c_vir = np.hstack((self.c_cluster_vir, self.c_env_vir))
        nactive = self.c_cluster_occ.shape[-1] + self.c_cluster_vir.shape[-1]
        nocc, nvir = c_occ.shape[-1], c_vir.shape[-1]
        nocc_frozen = self.c_env_occ.shape[-1]
        nvir_frozen = self.c_env_vir.shape[-1]
        mo_coeff = np.hstack((c_occ, c_vir))

        # Check occupations
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
            if getattr(self.mf, 'with_df', None) is None:
                eri = pyscf.ao2mo.incore.full(self.mf._eri, c_active, compact=False)
                eri = eri.reshape((c_active.shape[1],) * 4)
            else:
                if self.mf.with_df._cderi is None:
                    self.mf.with_df.build()
                eri = np.asarray(pyscf.lib.unpack_tril(self.mf.with_df._cderi, axis=-1))
                eri = ragf2._ao2mo_3c(eri, c_active, c_active)

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

        e_corr = cluster_solver.e_corr
        rdm1, fock, t_occ, t_vir = self.project_to_fragment(cluster_solver, c_active)

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
                fock=fock,
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


    def run(self):
        ''' Run self.kernel and return self.

        Returns
        -------
        frag : EAGF2Fragment
            `EAGF2Fragment` object containing calculation results.
        '''

        self.kernel()

        return self
