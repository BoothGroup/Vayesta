import dataclasses
from timeit import default_timer as timer
import copy

import numpy as np
import scipy
import scipy.optimize

from pyscf import scf

from vayesta.core.util import break_into_lines, OptionsBase, AbstractMethodError, einsum, log_time, dot


class ClusterSolver:
    """Base class for cluster solver"""

    @dataclasses.dataclass
    class Options(OptionsBase):
        pass

    def __init__(self, mf, fragment, cluster, log=None, **kwargs):
        """
        Arguments
        ---------
        """
        self.mf = mf
        self.fragment = fragment
        self.cluster = cluster
        self.log = (log or fragment.log)
        # --- Options:
        self.opts = self.Options()
        self.opts.update(**kwargs)
        self.log.info("Parameters of %s:" % self.__class__.__name__)
        self.log.info(break_into_lines(str(self.opts), newline='\n    '))

        # Additional external potential
        self.v_ext = None

        # --- Results
        self.converged = False
        self.e_corr = 0
        self.wf = None
        self.dm1 = None
        self.dm2 = None

    @property
    def base(self):
        """TODO: Remove fragment/embedding dependence...?"""
        return self.fragment.base

    def get_fock(self):
        c_active = self.cluster.c_active
        return dot(c_active.T, self.base.get_fock(), c_active)

    def kernel(self, *args, **kwargs):
        """Set up everything for a calculation on the CAS and pass this to the solver-specific kernel that runs on this
        information."""
        mf = self.make_clus_mf()
        return self.kernel_solver(mf)

    def kernel_solver(self, mf_clus, eris_energy=None):
        raise AbstractMethodError()

    @property
    def _scf_class(self):
        return scf.RHF

    def make_clus_mf(self, screening=None):
        bare_eris = self.get_eris()
        heff, mo_energy = self.get_heff(bare_eris)
        clusmol, mo_coeff, mo_occ = self.get_clus_info()

        clusmf = self._scf_class(clusmol)
        clusmf.get_hcore = lambda *args: heff
        clusmf.get_ovlp = lambda *args: np.eye(clusmol.nao)
        fock = self.get_fock()
        clusmf.get_fock = lambda *args: fock
        clusmf.mo_coeff = mo_coeff
        clusmf.mo_occ = mo_occ
        #clusmf.mo_energy = mo_energy

        if screening is None:
            clusmf._eri = bare_eris
        else:
            pass
        return clusmf

    def get_clus_info(self):
        clusmol = self.mf.mol.__class__()
        clusmol.nelec = (self.cluster.nocc_active, self.cluster.nocc_active)
        clusmol.nao = self.cluster.norb_active
        clusmol.build()
        mo_coeff = np.eye(clusmol.nao)

        mo_occ = np.zeros((clusmol.nao,))
        mo_occ[:clusmol.nelec[0]] = 2.0

        return clusmol, mo_coeff, mo_occ

    def get_heff(self, eris, fock=None, with_vext=True):
        if fock is None:
            fock = self.get_fock()
        mo_energy = np.diag(fock)

        occ = np.s_[:self.cluster.nocc_active]
        v_act = 2 * einsum('iipq->pq', eris[occ, occ]) - einsum('iqpi->pq', eris[occ, :, :, occ])
        h_eff = fock - v_act
        # This should be equivalent to:
        # core = np.s_[:self.nocc_frozen]
        # dm_core = 2*np.dot(self.mo_coeff[:,core], self.mo_coeff[:,core].T)
        # v_core = self.mf.get_veff(dm=dm_core)
        # h_eff = np.linalg.multi_dot((self.c_active.T, self.base.get_hcore()+v_core, self.c_active))
        if with_vext and self.v_ext is not None:
            h_eff += self.v_ext
        return h_eff, mo_energy

    def get_eris(self, *args, **kwargs):
        with log_time(self.log.timing, "Time for AO->MO of ERIs:  %s"):
            coeff = self.cluster.c_active
            eris = self.base.get_eris_array(coeff)
        return eris


class UClusterSolver(ClusterSolver):

    def get_fock(self):
        c_active = self.cluster.c_active
        fock = self.base.get_fock()
        return (dot(c_active[0].T, fock[0], c_active[0]),
                dot(c_active[1].T, fock[1], c_active[1]))
        raise RuntimeError

    @property
    def _scf_class(self):
        return scf.UHF

    def get_clus_info(self):
        clusmol = self.mf.mol.__class__()
        clusmol.nelec = self.cluster.nocc_active
        na, nb = self.cluster.norb_active
        # NB if the number of alpha and beta active orbitals is different this approach can't work.
        assert (na == nb)
        clusmol.nao = na
        mo_coeff = np.zeros((2, clusmol.nao, clusmol.nao))
        mo_coeff[0] = np.eye(clusmol.nao)
        mo_coeff[1] = np.eye(clusmol.nao)

        mo_occ = np.zeros((2, clusmol.nao))
        mo_occ[0, :clusmol.nelec[0]] = 1
        mo_occ[1, :clusmol.nelec[1]] = 1

        return clusmol, mo_coeff, mo_occ

    def get_heff(self, eris, fock=None, with_vext=True):
        if fock is None:
            fock = self.get_fock()
        mo_energy = np.diagonal(fock, 1, 2)

        oa = np.s_[:self.cluster.nocc_active[0]]
        ob = np.s_[:self.cluster.nocc_active[1]]
        gaa, gab, gbb = eris
        va = (einsum('iipq->pq', gaa[oa, oa]) + einsum('pqii->pq', gab[:, :, ob, ob])  # Coulomb
              - einsum('ipqi->pq', gaa[oa, :, :, oa]))  # Exchange
        vb = (einsum('iipq->pq', gbb[ob, ob]) + einsum('iipq->pq', gab[oa, oa])  # Coulomb
              - einsum('ipqi->pq', gbb[ob, :, :, ob]))  # Exchange
        h_eff = (fock[0] - va, fock[1] - vb)
        if with_vext and self.v_ext is not None:
            h_eff = ((h_eff[0] + self.v_ext[0]),
                     (h_eff[1] + self.v_ext[1]))
        return h_eff, mo_energy

    def get_eris(self, *args, **kwargs):
        with log_time(self.log.timing, "Time for AO->MO of ERIs:  %s"):
            coeff = self.cluster.c_active
            eris = (self.base.get_eris_array(coeff[0]),
                    self.base.get_eris_array((coeff[0], coeff[0], coeff[1], coeff[1])),
                    self.base.get_eris_array(coeff[1]))
        return eris


class EBClusterSolver(ClusterSolver):
    @dataclasses.dataclass
    class Options(ClusterSolver.Options):
        polaritonic_shift: bool = True

    def kernel(self, *args, **kwargs):
        """Set up everything for a calculation on the CAS and pass this to the solver-specific kernel that runs on this
        information."""
        self.set_polaritonic_shift(self.fragment.bos_freqs, self.fragment.couplings)
        mf = self.make_clus_mf()
        couplings = self.get_polaritonic_shifted_couplings(self.fragment.bos_freqs, self.fragment.couplings)

        return self.kernel_solver(mf, self.fragment.bos_freqs, couplings)

    def kernel_solver(self, mf_clus, freqs, couplings):
        raise AbstractMethodError()

    @property
    def polaritonic_shift(self):
        try:
            return self._polaritonic_shift
        except AttributeError as e:
            self.log.critical("Polaritonic shift not yet set.")
            raise e

    def get_heff(self, eris, fock=None, with_vext=True):
        heff = super().get_heff(eris, fock, with_vext)

        if self.opts.polaritonic_shift:
            fock_shift = self.get_polaritonic_fock_shift(self.fragment.couplings)
            if not np.allclose(fock_shift[0], fock_shift[1]):
                self.log.critical("Polaritonic shift breaks cluster spin symmetry; please either use an unrestricted"
                                  "formalism or bosons without polaritonic shift.")
            heff = heff + fock_shift[0]
        return heff

    def set_polaritonic_shift(self, freqs, couplings):
        no = self.cluster.nocc_active
        if isinstance(no, int):
            noa = nob = no
        else:
            noa, nob = no
        self._polaritonic_shift = np.multiply(freqs ** (-1), einsum("npp->n", couplings[0][:, :noa, :noa]) +
                                              einsum("npp->n", couplings[1][:, :nob, :nob]))
        self.log.info("Applying Polaritonic shift gives energy change of %e",
                      -sum(np.multiply(self._polaritonic_shift ** 2, freqs)))

    def get_polaritonic_fock_shift(self, couplings):
        return tuple([- einsum("npq,n->pq", x + x.transpose(0, 2, 1), self.polaritonic_shift) for x in couplings])

    def get_polaritonic_shifted_couplings(self, freqs, couplings):
        temp = np.multiply(self.polaritonic_shift, freqs) / (2 * self.cluster.nocc_active)
        return tuple([x - einsum("pq,n->npq", np.eye(x.shape[1]), temp) for x in couplings])

    def get_eb_dm_polaritonic_shift(self):
        shift = self.polaritonic_shift
        if isinstance(self.dm1, tuple):
            # UHF calculation
            return tuple([-einsum("n,pq->pqn", shift, x) for x in self.dm1])
        else:
            return (-einsum("n,pq->pqn", shift, self.dm1 / 2),) * 2


# If we wanted an abstract base class for unrestricted electron-boson this would be all that was required; in practice
# can do this in actual solvers.
class UEBClusterSolver(EBClusterSolver, UClusterSolver):

    def get_heff(self, eris, fock=None, with_vext=True):
        heff = super().get_heff(eris, fock, with_vext)

        if self.opts.polaritonic_shift:
            fock_shift = self.get_polaritonic_fock_shift(self.fragment.couplings)
            heff = tuple([x + y for x, y in zip(heff, fock_shift)])
        return heff

    def get_polaritonic_shifted_couplings(self, freqs, couplings):
        temp = np.multiply(self.polaritonic_shift, freqs) / sum(self.cluster.nocc_active)
        return tuple([x - einsum("pq,n->npq", np.eye(x.shape[1]), temp) for x in couplings])
