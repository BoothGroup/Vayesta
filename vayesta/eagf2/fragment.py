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


#def build_moments(
#        frag,
#        mo_coeff_occ, mo_coeff_vir,
#        mo_coeff_occ_other, mo_coeff_vir_other,
#        which='occupied',
#):
#    '''
#    Construct the first two moments of the occupied self-energy due to
#    a pair of clusters.
#    '''
#
#    ci_p = np.linalg.multi_dot((frag.mf.mo_coeff, frag.qmo_coeff[:frag.mol.nao], mo_coeff_occ))
#    ca_p = np.linalg.multi_dot((frag.mf.mo_coeff, frag.qmo_coeff[:frag.mol.nao], mo_coeff_vir))
#    c_p = np.hstack((ci_p, ca_p))
#
#    ci_q = np.linalg.multi_dot((frag.mf.mo_coeff, frag.qmo_coeff[:frag.mol.nao], mo_coeff_occ_other))
#    ca_q = np.linalg.multi_dot((frag.mf.mo_coeff, frag.qmo_coeff[:frag.mol.nao], mo_coeff_vir_other))
#    c_q = np.hstack((ci_q, ca_q))
#
#    nocc = np.sum(frag.qmo_occ > 0)
#    occ = slice(None, nocc)
#    vir = slice(nocc, None)
#    if which.lower().startswith('vir'):
#        mo_coeff_occ, mo_coeff_vir = mo_coeff_vir, mo_coeff_occ
#        mo_coeff_occ_other, mo_coeff_vir_other = mo_coeff_vir_other, mo_coeff_occ_other
#        ci_p, ca_p = ca_p, ci_p
#        ci_q, ca_q = ca_q, ci_q
#        occ, vir = vir, occ
#
#    ei = frag.qmo_energy[occ]
#    ea = frag.qmo_energy[vir]
#
#    cj = np.linalg.multi_dot((frag.mf.mo_coeff, frag.qmo_coeff[:frag.mol.nao, occ], mo_coeff_occ[occ], mo_coeff_occ[occ].T))
#    ca = np.linalg.multi_dot((frag.mf.mo_coeff, frag.qmo_coeff[:frag.mol.nao, vir], mo_coeff_vir[vir], mo_coeff_vir[vir].T))
#    pija = ao2mo.general(frag.mf._eri, (c_p, ci_p, cj, ca), compact=False)
#    pija = pija.reshape([c.shape[1] for c in (c_p, ci_p, cj, ca)])
#
#    cj = np.linalg.multi_dot((frag.mf.mo_coeff, frag.qmo_coeff[:frag.mol.nao, occ], mo_coeff_occ_other[occ], mo_coeff_occ_other[occ].T))
#    ca = np.linalg.multi_dot((frag.mf.mo_coeff, frag.qmo_coeff[:frag.mol.nao, vir], mo_coeff_vir_other[vir], mo_coeff_vir_other[vir].T))
#    qija = ao2mo.general(frag.mf._eri, (c_q, ci_q, cj, ca), compact=False)
#    qjia = ao2mo.general(frag.mf._eri, (c_q, cj, ci_q, ca), compact=False)
#    qija = qija.reshape([c.shape[1] for c in (c_q, ci_q, cj, ca)])
#    qjia = qjia.reshape([c.shape[1] for c in (c_q, cj, ci_q, ca)])
#
#    t0 = (
#        + 2.0 * lib.einsum('pkja,qlja,ik,il->pq', pija, qija, mo_coeff_occ[occ], mo_coeff_occ_other[occ])
#        - 1.0 * lib.einsum('pkja,qjla,ik,il->pq', pija, qjia, mo_coeff_occ[occ], mo_coeff_occ_other[occ])
#    )
#
#    delta = lib.direct_sum('i+j-a->ija', ei, ei, ea)
#    delta = lib.einsum('ija,ik,il->klja', delta, mo_coeff_occ[occ], mo_coeff_occ_other[occ])
#
#    t1 = (
#        + 2.0 * lib.einsum('pkja,qlja,klja->pq', pija, qija, delta)
#        - 1.0 * lib.einsum('pkja,qjla,klja->pq', pija, qjia, delta)
#    )
#
#    return np.array([t0, t1])

def build_moments(
        frag,
        mo_coeff_occ, mo_coeff_vir,
        mo_coeff_other_occ, mo_coeff_other_vir,
        which='occupied',
):
    c_p = np.linalg.multi_dot((frag.mf.mo_coeff, np.hstack((mo_coeff_occ, mo_coeff_vir))[:frag.mol.nao]))
    c_q = np.linalg.multi_dot((frag.mf.mo_coeff, np.hstack((mo_coeff_other_occ, mo_coeff_other_vir))[:frag.mol.nao]))
    occ = slice(None, np.sum(frag.qmo_occ > 0))
    vir = slice(np.sum(frag.qmo_occ > 0), None)

    # In the basis of MO+aux:
    proj_occ = np.dot(mo_coeff_occ, mo_coeff_occ.T.conj())
    proj_vir = np.dot(mo_coeff_vir, mo_coeff_vir.T.conj())
    proj_other_occ = np.dot(mo_coeff_other_occ, mo_coeff_other_occ.T.conj())
    proj_other_vir = np.dot(mo_coeff_other_vir, mo_coeff_other_vir.T.conj())

    # In the basis of QMOs:
    proj_occ = np.linalg.multi_dot((frag.qmo_coeff[:, occ].T.conj(), proj_occ, frag.qmo_coeff[:, occ]))
    proj_vir = np.linalg.multi_dot((frag.qmo_coeff[:, vir].T.conj(), proj_vir, frag.qmo_coeff[:, vir]))
    proj_other_occ = np.linalg.multi_dot((frag.qmo_coeff[:, occ].T.conj(), proj_other_occ, frag.qmo_coeff[:, occ]))
    proj_other_vir = np.linalg.multi_dot((frag.qmo_coeff[:, vir].T.conj(), proj_other_vir, frag.qmo_coeff[:, vir]))

    c_p_occ = np.linalg.multi_dot((frag.mf.mo_coeff, frag.qmo_coeff[:frag.mol.nao, occ], proj_occ))
    c_p_vir = np.linalg.multi_dot((frag.mf.mo_coeff, frag.qmo_coeff[:frag.mol.nao, vir], proj_vir))
    c_q_occ = np.linalg.multi_dot((frag.mf.mo_coeff, frag.qmo_coeff[:frag.mol.nao, occ], proj_other_occ))
    c_q_vir = np.linalg.multi_dot((frag.mf.mo_coeff, frag.qmo_coeff[:frag.mol.nao, vir], proj_other_vir))

    if which.lower().startswith('vir'):
        c_p_occ, c_p_vir = c_p_vir, c_p_occ
        c_q_occ, c_q_vir = c_q_vir, c_q_occ
        occ, vir = vir, occ

    pija = ao2mo.general(frag.mf._eri, (c_p, c_p_occ, c_p_occ, c_p_vir), compact=False)
    pija = pija.reshape([c.shape[1] for c in (c_p, c_p_occ, c_p_occ, c_p_vir)])

    qija = ao2mo.general(frag.mf._eri, (c_q, c_q_occ, c_q_occ, c_q_vir), compact=False)
    qija = qija.reshape([c.shape[1] for c in (c_q, c_q_occ, c_q_occ, c_q_vir)])

    eija = lib.direct_sum('i+j-a->ija', frag.qmo_energy[occ], frag.qmo_energy[occ], frag.qmo_energy[vir])
    qija = 2.0 * qija - qija.swapaxes(1, 2).copy()

    t0 = np.dot(
            np.reshape(pija, (-1, eija.size)),
            np.reshape(qija, (-1, eija.size)).T,
    )

    qija *= eija[None]

    t1 = np.dot(
            np.reshape(pija, (-1, eija.size)),
            np.reshape(qija, (-1, eija.size)).T,
    )

    #ei = frag.qmo_energy[occ]
    #ej = frag.qmo_energy[occ]
    #ea = frag.qmo_energy[vir]
    #t0, t1 = helper.build_moments(ei, ej, ea, pija, qija)

    return np.array([t0, t1])

def _build_moments(
        frag,
        mo_coeff_occ, mo_coeff_vir,
        mo_coeff_other_occ, mo_coeff_other_vir,
        which='occupied',
):
    c_p = np.linalg.multi_dot((frag.mf.mo_coeff, np.hstack((mo_coeff_occ, mo_coeff_vir))[:frag.mol.nao]))
    c_q = np.linalg.multi_dot((frag.mf.mo_coeff, np.hstack((mo_coeff_other_occ, mo_coeff_other_vir))[:frag.mol.nao]))

    # Projectors in the basis of MO+aux:
    proj_occ = np.dot(mo_coeff_occ, mo_coeff_occ.T.conj())
    proj_vir = np.dot(mo_coeff_vir, mo_coeff_vir.T.conj())
    proj_other_occ = np.dot(mo_coeff_other_occ, mo_coeff_other_occ.T.conj())
    proj_other_vir = np.dot(mo_coeff_other_vir, mo_coeff_other_vir.T.conj())

    # Find the basis spanning the union of the cluster spaces:
    c_occ = scipy.linalg.orth(proj_occ + proj_other_occ)
    c_vir = scipy.linalg.orth(proj_vir + proj_other_vir)
    #w, c_occ = np.linalg.eigh(proj_occ + proj_other_occ)
    #c_occ = c_occ[:, np.abs(w) > 1e-10]
    #w, c_vir = np.linalg.eigh(proj_vir + proj_other_vir)
    #c_vir = c_vir[:, np.abs(w) > 1e-10]
    c_occ, _, e_occ = frag.canonicalize_qmo(c_occ, eigvals=True)
    c_vir, _, e_vir = frag.canonicalize_qmo(c_vir, eigvals=True)

    # Transform orbitals into AO rotations:
    c_occ = np.dot(frag.mf.mo_coeff, c_occ[:frag.mol.nao])
    c_vir = np.dot(frag.mf.mo_coeff, c_vir[:frag.mol.nao])

    # If virtual required, swap:
    if which.lower().startswith('vir'):
        c_occ, c_vir = c_vir, c_occ
        e_occ, e_vir = e_vir, e_occ

    # Transform integrals:
    pija = ao2mo.general(frag.mf._eri, (c_p, c_occ, c_occ, c_vir), compact=False)
    pija = pija.reshape([c.shape[1] for c in (c_p, c_occ, c_occ, c_vir)])

    qija = ao2mo.general(frag.mf._eri, (c_q, c_occ, c_occ, c_vir), compact=False)
    qija = qija.reshape([c.shape[1] for c in (c_q, c_occ, c_occ, c_vir)])

    # Build moments:
    eija = lib.direct_sum('i+j-a->ija', e_occ, e_occ, e_vir)
    qija = 2.0 * qija - qija.swapaxes(1, 2).copy()

    t0 = np.dot(
            np.reshape(pija, (-1, eija.size)),
            np.reshape(qija, (-1, eija.size)).T,
    )

    qija *= eija[None]

    t1 = np.dot(
            np.reshape(pija, (-1, eija.size)),
            np.reshape(qija, (-1, eija.size)).T,
    )

    return np.array([t0, t1])


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


    def kernel(self, solver, se=None, fock=None, other_frag=None):
        ''' Run the solver for the fragment.
        '''

        if not self.opts.democratic:
            assert other_frag is not None

        if se is not None:
            self.se = se
        if fock is not None:
            self.fock = fock
        if se is not None or fock is not None:
            self.qmo_energy, self.qmo_coeff = se.eig(fock)
            self.qmo_occ = np.array([2.0 * (x < se.chempot) for x in self.qmo_energy])

        if other_frag is not None:
            for attr in ['se', 'fock', 'qmo_energy', 'qmo_coeff', 'qmo_occ']:
                setattr(other_frag, attr, getattr(self, attr))

        if self.c_cls_occ is None:
            coeffs = self.make_bath()
            self.c_cls_occ, self.c_cls_vir, self.c_env_occ, self.c_env_vir = coeffs

        mo_coeff_occ, _, mo_energy_occ = self.canonicalize_qmo(self.c_cls_occ, eigvals=True)
        mo_coeff_vir, _, mo_energy_vir = self.canonicalize_qmo(self.c_cls_vir, eigvals=True)
        mo_coeff = np.hstack((mo_coeff_occ, mo_coeff_vir))

        if self.opts.democratic:
            c_occ = np.dot(self.mf.mo_coeff, mo_coeff_occ[:solver.nact])
            c_vir = np.dot(self.mf.mo_coeff, mo_coeff_vir[:solver.nact])

            with helper.QMOIntegrals(self, c_occ, c_vir, 'xija') as xija:
                t_occ = solver._build_moments(mo_energy_occ, mo_energy_vir, xija)

            with helper.QMOIntegrals(self, c_occ, c_vir, 'xabi') as xabi:
                t_vir = solver._build_moments(mo_energy_vir, mo_energy_occ, xabi)

            mo_coeff_other = mo_coeff

        else:
            if other_frag.c_cls_occ is None:
                coeffs = other_frag.make_bath()
                other_frag.c_cls_occ, other_frag.c_cls_vir, \
                        other_frag.c_env_occ, other_frag.c_env_vir = coeffs

            mo_coeff_occ_other, _, mo_energy_occ_other = \
                    other_frag.canonicalize_qmo(other_frag.c_cls_occ, eigvals=True)
            mo_coeff_vir_other, _, mo_energy_vir_other = \
                    other_frag.canonicalize_qmo(other_frag.c_cls_vir, eigvals=True)
            mo_coeff_other = np.hstack((mo_coeff_occ_other, mo_coeff_vir_other))

            t_occ = build_moments(
                    self,
                    mo_coeff_occ, mo_coeff_vir,
                    mo_coeff_occ_other, mo_coeff_vir_other,
                    'occupied',
            )

            t_vir = build_moments(
                    self,
                    mo_coeff_occ, mo_coeff_vir,
                    mo_coeff_occ_other, mo_coeff_vir_other,
                    'virtual',
            )

        results = EAGF2FragmentResults(
                fid=self.id,
                n_active=mo_coeff.shape[-1],
                c_active=mo_coeff,
                n_active_other=mo_coeff_other.shape[-1],
                c_active_other=mo_coeff_other,
                moms=np.array([t_occ, t_vir]),
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
