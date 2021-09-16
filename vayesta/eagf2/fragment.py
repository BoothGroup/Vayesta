import dataclasses

import numpy as np

import pyscf
import pyscf.lib
import pyscf.agf2
import pyscf.ao2mo

from vayesta.core.util import OptionsBase, NotSet, time_string
from vayesta.core import QEmbeddingFragment
from vayesta.core.helper import orbital_sign_convention
from vayesta.eagf2 import helper

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
    max_bath_order: int = NotSet
    bno_threshold: float = NotSet
    bno_threshold_factor: float = NotSet
    dmet_threshold: float = 1e-4

    # --- Appease EWF inheritance
    plot_orbitals: bool = False
    wf_partition: str = 'first-occ'
    sym_factor: float = 1.0


@dataclasses.dataclass
class EAGF2FragmentResults:
    ''' Results for EAGF2 fragments.

    Attributes
    ----------
    fid : int
        Fragment ID.
    n_active : int
        Number of active orbitals.
    c_active : np.ndarray
        Active cluster orbital coefficients.
    e_corr : float
        Correlation energy.
    e_1b : float
        One-body part of total energy, including nuclear repulsion.
    e_2b : float
        Two-body part of total energy.
    e_ip : float
        Ionisation potential.
    e_ea : float
        Electron affinity.
    moms : np.ndarray
        Occupied and virtual self-energy moments in the cluster basis.
    '''

    fid: int = None
    n_active: int = None
    c_active: np.ndarray = None
    moms: np.ndarray = None


class EAGF2Fragment(QEmbeddingFragment):

    Options = EAGF2FragmentOptions
    Results = EAGF2FragmentResults

    def __init__(self, *args, **kwargs):
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
        **kwargs : dict, optional
            Additional arguments passed to `EAGF2FragmentOptions`.
        '''

        super().__init__(*args, **kwargs)

        defaults = self.Options().replace(self.base.Options(), select=NotSet)
        for key, val in self.opts.items():
            if val != getattr(defaults, key):
                self.log.info('  > %-24s %3s %r', key + ':', '(*)', val)
            else:
                self.log.debugv('  > %-24s %3s %r', key + ':', '', val)

        self.c_env_occ = None
        self.c_env_vir = None
        self.c_cluster_occ = None
        self.c_cluster_vir = None

        # Initialise with no auxiliary space:
        self.se = pyscf.agf2.SelfEnergy([], [[],]*self.mf.mo_occ.size)
        self.fock = np.diag(self.mf.mo_energy)
        self.qmo_energy, self.qmo_coeff = np.linalg.eigh(self.fock)
        self.qmo_occ = self.mf.get_occ(self.qmo_energy, self.qmo_coeff)


    def make_power_bath(self, max_order=None, c_frag=None, c_env=None):
        ''' Make power bath orbitals.

        Arguments
        ---------
        max_order : int
            Maximum order of power orbital.

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

        self.log.info("Building power orbitals")
        self.log.info("***********************")
        self.log.changeIndentLevel(1)

        if max_order is None:
            max_order = self.opts.max_bath_order

        c_bath, c_env_occ, c_env_vir = helper.make_power_bath(
                self,
                max_order=max_order,
                c_frag=c_frag,
                c_env=c_env,
        )

        self.log.timing("Time for power orbital bath:  %s", time_string(timer() - t0))
        self.log.changeIndentLevel(-1)

        self.print_power_orbital_character(c_bath)

        c_cluster_occ, c_cluster_vir = self.diagonalize_cluster_dm(
                c_frag if c_frag is not None else self.c_frag,
                c_bath,
                tol=2*self.opts.dmet_threshold,
        )
        self.log.info(
                "Cluster orbitals:  n(occ) = %d  n(vir) = %d",
                c_cluster_occ.shape[-1], c_cluster_vir.shape[-1],
        )

        return c_cluster_occ, c_cluster_vir, c_env_occ, c_env_vir


    def make_dmet_bath(self, c_frag=None, c_env=None):
        return self.make_power_bath(max_order=0, c_frag=c_frag, c_env=c_env)


    def make_bno_bath(self, c_frag=None, c_env=None):
        raise NotImplementedError  #TODO


    def make_complete_bath(self, c_frag=None, c_env=None):
        ''' Build a complete bath.
        '''

        if c_frag is None:
            c_frag = self.c_frag
        if c_env is None:
            c_env = self.c_env

        nmo = c_frag.shape[0]

        c_cluster_occ, c_cluster_vir = self.diagonalize_cluster_dm(
                c_frag,
                c_env,
                tol=2*self.opts.dmet_threshold,
        )
        self.log.info(
                "Cluster orbitals:  n(occ) = %d  n(vir) = %d",
                c_cluster_occ.shape[-1], c_cluster_vir.shape[-1],
        )

        c_env_occ = np.zeros((nmo, 0))
        c_env_vir = np.zeros((nmo, 0))

        return c_cluster_occ, c_cluster_vir, c_env_occ, c_env_vir


    def make_bath(self, c_frag=None, c_env=None):
        if self.opts.bath_type.lower() == 'none':
            return self.make_dmet_bath(c_frag=c_frag, c_env=c_env)
        elif self.opts.bath_type.lower() == 'power':
            return self.make_power_bath(c_frag=c_frag, c_env=c_env)
        elif self.opts.bath_type.lower() == 'all':
            return self.make_complete_bath(c_frag=c_frag, c_env=c_env)
        elif self.opts.bath_type.lower() == 'mp2-bno':
            return self.make_bno_bath(self, c_frag=c_frag, c_env=c_env)


    def print_power_orbital_character(self, c_bath):
        ''' Print the character of the power orbitals including auxiliaries.
        '''

        self.log.info("Bath states")
        self.log.info("***********")
        self.log.changeIndentLevel(1)

        nmo = self.mf.mo_occ.size
        nocc = np.sum(self.mf.mo_occ > 0)
        nocc_aux = self.se.get_occupied().naux

        self.log.info("%4s  %7s    %7s    %7s    %7s", "Bath", "1h", "1p", "2h1p", "1h2p")
        parts = np.hsplit(c_bath, [nocc, nmo, nmo+nocc_aux])
        for i in range(c_bath.shape[-1]):
            self.log.debugv(
                    "%4d  %7.3f %%  %7.3f %%  %7.3f %%  %7.3f %%",
                    i, *(100*np.linalg.norm(v[i])**2 for v in parts),
            )
        self.log.info(
                "%4s  %7.3f %%  %7.3f %%  %7.3f %%  %7.3f %%",
                "Mean", *(100*np.linalg.norm(v)**2/c_bath.shape[-1] for v in parts),
        )

        self.log.changeIndentLevel(-1)


    def democratic_partition(self, m, c=None):
        ''' Democratically partition a matrix.
        '''

        if c is None:
            c = np.dot(self.results.c_active.T.conj(), self.c_frag)

        p_frag = np.dot(c, c.T.conj())

        m_demo = (
                + 0.5 * np.einsum('...pq,pi->...iq', m, p_frag)
                + 0.5 * np.einsum('...pq,qj->...pj', m, p_frag)
        )

        return m_demo


    def canonicalize_qmo(self, *qmo_coeff, eigvals=True, sign_convention=True):
        ''' Diagonalize Fock matrix within subspace, including auxiliaries.

        Parameters
        ----------
        *qmo_coeff : ndarrays
            Orbital coefficients.
        eigvals : bool
            Return energies of canonicalized QMOs.

        Returns
        -------
        c_canon : ndarray
            Canonicalized QMO coefficients.
        rot : ndarray
            Rotation matrix.
        e_canon : ndarray
            Canonicalized QMO energies, if `eigvals==True`.
        '''

        qmo_coeff = np.hstack(qmo_coeff)
        fock = np.linalg.multi_dot((qmo_coeff.T.conj(), self.se.get_array(self.fock), qmo_coeff))
        energy, rot = np.linalg.eigh(fock)
        canon = np.dot(qmo_coeff, rot)

        if sign_convention:
            canon, signs = orbital_sign_convention(canon)
            rot *= signs[None]

        if eigvals:
            return canon, rot, energy
        return canon, rot


    def diagonalize_cluster_dm(self, *qmo_coeff, tol=1e-4):
        '''
        Diagoanlize cluster (fragment_bath) DM to get fully occupied and
        virtual orbitals, including auxiliary space.

        Parameters
        ----------
        *qmo_coeff : ndarrays
            QMO coefficients.
        tol : float, optional
            If set, check that all eigenvalues of the cluster DM are close
            to 0 or 1, with the tolerance given by tol. Default= 1e-4.

        Returns
        -------
        c_occclt : ndarray
            Occupied cluster orbitals.
        c_virclt : ndarray
            Virtual cluster orbitals.
        '''

        c_cls = np.hstack(qmo_coeff)
        dm = self.mf.make_rdm1(self.qmo_coeff, self.qmo_occ)
        dm = np.linalg.multi_dot((c_cls.T.conj(), dm, c_cls)) / 2
        e, v = np.linalg.eigh(dm)

        if tol and not np.allclose(np.fmin(abs(e), abs(e-1)), 0, atol=tol, rtol=0):
            raise RuntimeError("Error while diagonalizing cluster DM: eigenvalues not "
                               "all close to 0 or 1:\n%s", e)

        e, v = e[::-1], v[:, ::-1]
        c_cls = np.dot(c_cls, v)
        nocc = sum(e >= 0.5)
        c_cls_occ, c_cls_vir = np.hsplit(c_cls, [nocc])

        return c_cls_occ, c_cls_vir


    def kernel(self, solver, se, fock, c_frag=None, c_env=None):
        ''' Run the solver for the fragment.
        '''

        self.se = se
        self.fock = fock
        self.qmo_energy, self.qmo_coeff = se.eig(fock)
        self.qmo_occ = np.array([2.0 * (x < se.chempot) for x in self.qmo_energy])

        c_cls_occ, c_cls_vir, c_env_occ, c_env_vir = self.make_bath(c_frag=c_frag, c_env=c_env)

        mo_coeff_act_occ, _, mo_energy_act_occ = self.canonicalize_qmo(c_cls_occ, eigvals=True)
        mo_coeff_act_vir, _, mo_energy_act_vir = self.canonicalize_qmo(c_cls_vir, eigvals=True)
        mo_coeff_act = np.hstack((mo_coeff_act_occ, mo_coeff_act_vir))

        c_occ = np.dot(self.mf.mo_coeff, mo_coeff_act_occ[:solver.nact])
        c_vir = np.dot(self.mf.mo_coeff, mo_coeff_act_vir[:solver.nact])

        with helper.QMOIntegrals(self, c_occ, c_vir, 'xija') as xija:
            t_occ = solver._build_moments(mo_energy_act_occ, mo_energy_act_vir, xija)
        with helper.QMOIntegrals(self, c_occ, c_vir, 'xabi') as xabi:
            t_vir = solver._build_moments(mo_energy_act_vir, mo_energy_act_occ, xabi)

        moms_frag = np.array([t_occ, t_vir])

        results = EAGF2FragmentResults(
                fid=self.id,
                n_active=mo_coeff_act.shape[-1],
                c_active=mo_coeff_act,
                moms=moms_frag,
        )

        self._results = results

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
