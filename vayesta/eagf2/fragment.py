import dataclasses

import numpy as np
import scipy.linalg

from pyscf import lib, agf2, ao2mo

from vayesta.core.util import OptionsBase, NotSet, time_string
from vayesta.core import QEmbeddingFragment
from vayesta.core.helper import orbital_sign_convention
from vayesta.eagf2 import helper

try:
    from mpi4py import MPI
    timer = MPI.Wtime
except ImportError:
    from timeit import default_timer as timer


def build_moments(frag, other):
    # Find the basis spanning the union of the occupied cluster spaces:
    c = np.hstack((frag.c_qmo_occ, other.c_qmo_occ))
    c_occ, sv, _ = np.linalg.svd(c, full_matrices=False)
    tol = np.finfo(c.dtype).eps * np.max(c.shape) * np.max(sv)
    c_occ = c_occ[:, np.abs(sv) > tol]
    e_occ, r_occ = np.linalg.eigh(np.dot(c_occ.T.conj() * frag.qmo_energy[None], c_occ))
    c_occ = np.dot(c_occ, r_occ)
    del r_occ

    # Find the basis spanning the union of the virtual cluster spaces:
    c = np.hstack((frag.c_qmo_vir, other.c_qmo_vir))
    c_vir, sv, _ = np.linalg.svd(c, full_matrices=False)
    tol = np.finfo(c.dtype).eps * np.max(c.shape) * np.max(sv)
    c_vir = c_vir[:, np.abs(sv) > tol]
    e_vir, r_vir = np.linalg.eigh(np.dot(c_vir.T.conj() * frag.qmo_energy[None], c_vir))
    c_vir = np.dot(c_vir, r_vir)
    del r_vir

    # Find rotations from AOs into the new space:
    c_occ_p = np.dot(frag.c_ao_qmo_proj_occ, c_occ)
    c_vir_p = np.dot(frag.c_ao_qmo_proj_vir, c_vir)
    c_occ_q = np.dot(other.c_ao_qmo_proj_occ, c_occ)
    c_vir_q = np.dot(other.c_ao_qmo_proj_vir, c_vir)
    c_p = frag.c_ao_cls
    c_q = other.c_ao_cls

    def _build_part(ei, ea, pija, qija):
        eija = lib.direct_sum('i+j-a->ija', ei, ei, ea).ravel()

        if not isinstance(pija, tuple):
            qija = 2.0 * qija - qija.swapaxes(1, 2)
            pija = pija.reshape(-1, eija.size)
            qija = qija.reshape(-1, eija.size)

            t0 = np.dot(pija, qija.T.conj())
            t1 = np.dot(pija * eija[None], qija.T.conj())

        else:
            raise NotImplementedError

        return np.array([t0, t1])

    with helper.QMOIntegrals(frag, c_occ_p, c_vir_p, c_p, which='xija', keep_3c=True) as pija:
        with helper.QMOIntegrals(frag, c_occ_q, c_vir_q, c_q, which='xija', keep_3c=True) as qija:
            t_occ = _build_part(e_occ, e_vir, pija, qija)

    with helper.QMOIntegrals(frag, c_occ_p, c_vir_p, c_p, which='xabi', keep_3c=True) as pabi:
        with helper.QMOIntegrals(frag, c_occ_q, c_vir_q, c_q, which='xabi', keep_3c=True) as qabi:
            t_vir = _build_part(e_vir, e_occ, pabi, qabi)

    return np.array([t_occ, t_vir])


@dataclasses.dataclass
class EAGF2FragmentOptions(OptionsBase):
    ''' Options for EAGF2 fragments - see `EAGF2Fragment`.
    '''

    # --- Bath settings
    bath_type: str = NotSet
    max_bath_order: int = NotSet
    bno_threshold: float = NotSet
    bno_threshold_factor: float = NotSet
    dmet_threshold: float = NotSet

    # --- Moment settings
    democratic: bool = NotSet

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
    n_active_other: int = None
    c_active_other: np.ndarray = None
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

        # Convert fragment and environment orbitals to MO basis:
        self.c_frag = np.linalg.multi_dot((self.mf.mo_coeff.T, self.base.get_ovlp(), self.c_frag))
        self.c_env = np.linalg.multi_dot((self.mf.mo_coeff.T, self.base.get_ovlp(), self.c_env))

        # Cluster and environment orbitals:
        self.c_env_occ = None
        self.c_env_vir = None
        self.c_cls_occ = None
        self.c_cls_vir = None

        # Initialise with no auxiliary space:
        self.se = agf2.SelfEnergy([], [[],]*self.mf.mo_occ.size)
        self.fock = np.diag(self.mf.mo_energy)
        self.qmo_energy, self.qmo_coeff = np.linalg.eigh(self.fock)
        self.qmo_occ = self.mf.get_occ(self.qmo_energy, self.qmo_coeff)

        # Rotations used for non-democratic partitioning:
        self.c_ao_cls = None
        self.c_qmo_occ = None
        self.c_qmo_vir = None
        self.c_ao_qmo_proj_occ = None
        self.c_ao_qmo_proj_vir = None


    #TODO other properties assuming AO c_frag?
    #FIXME wrong value is printed during initialization, but accurate afterward because ao->mo
    @property
    def nelectron(self):
        #TODO check this
        c = np.dot(self.base.get_ovlp(), self.mf.mo_coeff)
        rdm1 = np.linalg.multi_dot((c.T.conj(), self.mf.make_rdm1(), c))
        ne = np.einsum('ai,ab,bi->', self.c_frag.conj(), rdm1, self.c_frag)
        return ne


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

        if c_frag is None:
            c_frag = self.c_frag
        if c_env is None:
            c_env = self.c_env

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
                tol=self.opts.dmet_threshold,
        )

        self.log.timing("Time for power orbital bath:  %s", time_string(timer() - t0))
        self.log.changeIndentLevel(-1)

        self.print_power_orbital_character(c_bath)

        c_cluster_occ, c_cluster_vir = self.diagonalize_cluster_dm(
                c_frag,
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


    def democratic_partition(self, m, p1, p2):
        ''' Democratically partition a matrix.
        '''

        m_demo = (
                + 0.5 * np.einsum('...pq,pi,qj->...ij', m, p1, p2)
                + 0.5 * np.einsum('...pq,pi,qj->...ij', m, p2, p1)
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


    #TODO better name, no rotations here any more
    def make_rotations(self):
        '''
        Build the rotations and projections required for non-democratic
        partitioning for the current fragment.
        '''

        nmo = self.mol.nao
        mo_coeff = self.mf.mo_coeff

        c_cls_occ, _ = self.canonicalize_qmo(self.c_cls_occ, eigvals=False)
        c_cls_vir, _ = self.canonicalize_qmo(self.c_cls_vir, eigvals=False)

        c_cls = np.hstack((c_cls_occ, c_cls_vir))
        c_ao_cls = np.dot(mo_coeff, c_cls[:nmo])

        c_qmo_occ = np.dot(self.qmo_coeff.T.conj(), c_cls_occ)
        proj_qmo_occ = np.dot(c_qmo_occ, c_qmo_occ.T.conj())

        c_qmo_vir = np.dot(self.qmo_coeff.T.conj(), c_cls_vir)
        proj_qmo_vir = np.dot(c_qmo_vir, c_qmo_vir.T.conj())

        c_ao_qmo_proj_occ = np.linalg.multi_dot((mo_coeff, self.qmo_coeff[:nmo], proj_qmo_occ))
        c_ao_qmo_proj_vir = np.linalg.multi_dot((mo_coeff, self.qmo_coeff[:nmo], proj_qmo_vir))

        return c_ao_cls, c_qmo_occ, c_qmo_vir, c_ao_qmo_proj_occ, c_ao_qmo_proj_vir


    def kernel(self, solver, se=None, fock=None, other_frag=None):
        ''' Run the solver for the fragment.
        '''
        #TODO: make auxiliary and QMO spaces stored in parent instead of fragment?

        if not self.opts.democratic:
            assert other_frag is not None

        # Set auxiliary and QMO space if not set:
        if se is not None:
            self.se = se
        if fock is not None:
            self.fock = fock
        if se is not None or fock is not None:
            self.qmo_energy, self.qmo_coeff = se.eig(fock)
            self.qmo_occ = np.array([2.0 * (x < se.chempot) for x in self.qmo_energy])

        # Set bath if not set:
        if self.c_cls_occ is None:
            coeffs = self.make_bath()
            self.c_cls_occ, self.c_cls_vir, self.c_env_occ, self.c_env_vir = coeffs

        if not self.opts.democratic:
            # Set other auxiliary and QMO space:
            other_frag.se, other_frag.fock = self.se, self.fock
            other_frag.qmo_energy, other_frag.qmo_coeff, other_frag.qmo_occ = \
                    self.qmo_energy, self.qmo_coeff, self.qmo_occ

            # Set other bath space if not set:
            if other_frag.c_cls_occ is None:
                coeffs = other_frag.make_bath()
                other_frag.c_cls_occ, other_frag.c_cls_vir, \
                        other_frag.c_env_occ, other_frag.c_env_vir = coeffs

            # Set rotations if not set:
            if self.c_ao_cls is None:
                rotations = frag.make_rotations()
                self.c_ao_cls, self.c_qmo_occ, self.c_qmo_vir, \
                        self.c_ao_qmo_proj_occ, self.c_ao_qmo_proj_vir = rotations

            # Set other rotations if not set:
            if other_frag.c_ao_cls is None:
                rotations = other_frag.make_rotations()
                other_frag.c_ao_cls, other_frag.c_qmo_occ, other_frag.c_qmo_vir, \
                        other_frag.c_ao_qmo_proj_occ, other_frag.c_ao_qmo_proj_vir = rotations

        if self.opts.democratic:
            mo_coeff_occ, _, mo_energy_occ = self.canonicalize_qmo(self.c_cls_occ, eigvals=True)
            mo_coeff_vir, _, mo_energy_vir = self.canonicalize_qmo(self.c_cls_vir, eigvals=True)
            mo_coeff = np.hstack((mo_coeff_occ, mo_coeff_vir))

            mo_coeff_other = mo_coeff

            c_occ = np.dot(self.mf.mo_coeff, mo_coeff_occ[:solver.nact])
            c_vir = np.dot(self.mf.mo_coeff, mo_coeff_vir[:solver.nact])

            #TODO replace
            with helper.QMOIntegrals(self, c_occ, c_vir, 'xija') as xija:
                t_occ = solver._build_moments(mo_energy_occ, mo_energy_vir, xija)

            with helper.QMOIntegrals(self, c_occ, c_vir, 'xabi') as xabi:
                t_vir = solver._build_moments(mo_energy_vir, mo_energy_occ, xabi)

            moms = np.array([t_occ, t_vir])

        else:
            mo_coeff_occ, _ = self.canonicalize_qmo(self.c_cls_occ, eigvals=False)
            mo_coeff_vir, _ = self.canonicalize_qmo(self.c_cls_vir, eigvals=False)
            mo_coeff = np.hstack((mo_coeff_occ, mo_coeff_vir))

            mo_coeff_occ_other, _ = other_frag.canonicalize_qmo(other_frag.c_cls_occ, eigvals=False)
            mo_coeff_vir_other, _ = other_frag.canonicalize_qmo(other_frag.c_cls_vir, eigvals=False)
            mo_coeff_other = np.hstack((mo_coeff_occ_other, mo_coeff_vir_other))

            moms = build_moments(self, other_frag)

        results = EAGF2FragmentResults(
                fid=self.id,
                n_active=mo_coeff.shape[-1],
                c_active=mo_coeff,
                n_active_other=mo_coeff_other.shape[-1],
                c_active_other=mo_coeff_other,
                moms=moms,
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
