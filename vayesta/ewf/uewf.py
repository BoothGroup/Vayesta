import numpy as np

from vayesta.core.qemb import UEmbedding
from vayesta.core.util import cache, deprecated, dot, einsum, log_method

from vayesta.ewf import REWF
from vayesta.ewf.ufragment import Fragment
from vayesta.misc import corrfunc
from vayesta.mpi import mpi

# Amplitudes
from vayesta.ewf.amplitudes import get_global_t1_uhf
from vayesta.ewf.amplitudes import get_global_t2_uhf
# Density-matrices
from vayesta.ewf.urdm import make_rdm1_ccsd
from vayesta.ewf.urdm import make_rdm1_ccsd_global_wf
from vayesta.ewf.urdm import make_rdm2_ccsd_global_wf
from vayesta.ewf.urdm import make_rdm2_ccsd_proj_lambda
from vayesta.ewf.icmp2 import get_intercluster_mp2_energy_uhf


class UEWF(REWF, UEmbedding):

    Fragment = Fragment

    # --- CC Amplitudes
    # -----------------

    # T-amplitudes
    get_global_t1 = get_global_t1_uhf
    get_global_t2 = get_global_t2_uhf

    def t1_diagnostic(self, warn_tol=0.02):
        # Per cluster
        for f in self.get_fragments(active=True, mpi_rank=mpi.rank):
            t1 = f.results.wf.t1
            if t1 is None:
                self.log.error("No T1 amplitudes found for %s.", f)
                continue
            nelec = t1[0].shape[0] + t1[1].shape[0]
            t1diag = (np.linalg.norm(t1[0]) / np.sqrt(nelec),
                      np.linalg.norm(t1[1]) / np.sqrt(nelec))
            if max(t1diag) > warn_tol:
                self.log.warning("T1 diagnostic for %-20s alpha= %.5f beta= %.5f", str(f)+':', *t1diag)
            else:
                self.log.info("T1 diagnostic for %-20s alpha= %.5f beta= %.5f", str(f)+':', *t1diag)
        # Global
        t1 = self.get_global_t1(mpi_target=0)
        if mpi.is_master:
            nelec = t1[0].shape[0] + t1[1].shape[0]
            t1diag = (np.linalg.norm(t1[0]) / np.sqrt(nelec),
                      np.linalg.norm(t1[1]) / np.sqrt(nelec))
            if max(t1diag) > warn_tol:
                self.log.warning("Global T1 diagnostic: alpha= %.5f beta= %.5f", *t1diag)
            else:
                self.log.info("Global T1 diagnostic: alpha= %.5f beta= %.5f", *t1diag)

    def d1_diagnostic(self):
        """Global wave function diagnostic."""
        t1a, t1b = self.get_global_t1()
        f = lambda x: np.sqrt(np.sort(np.abs(x[0])))[-1]
        d1ao = f(np.linalg.eigh(np.dot(t1a, t1a.T)))
        d1av = f(np.linalg.eigh(np.dot(t1a.T, t1a)))
        d1bo = f(np.linalg.eigh(np.dot(t1b, t1b.T)))
        d1bv = f(np.linalg.eigh(np.dot(t1b.T, t1b)))
        d1norm = max((d1ao, d1av, d1bo, d1bv))
        return d1norm

    # --- Density-matrices
    # --------------------

    # DM1

    @log_method()
    def _make_rdm1_mp2(self, *args, **kwargs):
        return make_rdm1_ccsd(self, *args, mp2=True, **kwargs)

    @log_method()
    def _make_rdm1_ccsd(self, *args, **kwargs):
        return make_rdm1_ccsd(self, *args, mp2=False, **kwargs)

    @log_method()
    def _make_rdm1_ccsd_global_wf(self, *args, ao_basis=False, with_mf=True, **kwargs):
        dm1a, dm1b = self._make_rdm1_ccsd_global_wf_cached(*args, **kwargs)
        if with_mf:
            dm1a[np.diag_indices(self.nocc[0])] += 1
            dm1b[np.diag_indices(self.nocc[1])] += 1
        if ao_basis:
            dm1a = dot(self.mo_coeff[0], dm1a, self.mo_coeff[0].T)
            dm1b = dot(self.mo_coeff[1], dm1b, self.mo_coeff[1].T)
        return (dm1a, dm1b)

    @cache(copy=True)
    def _make_rdm1_ccsd_global_wf_cached(self, *args, **kwargs):
        return make_rdm1_ccsd_global_wf(self, *args, **kwargs)

    @log_method()
    def _make_rdm1_ccsd_proj_lambda(self, *args, **kwargs):
        raise NotImplementedError()

    # DM2

    @log_method()
    def _make_rdm2_ccsd_global_wf(self, *args, **kwargs):
        return make_rdm2_ccsd_global_wf(self, *args, **kwargs)

    @log_method()
    def _make_rdm2_ccsd_proj_lambda(self, *args, **kwargs):
        return make_rdm2_ccsd_proj_lambda(self, *args, **kwargs)

    @log_method()
    def get_intercluster_mp2_energy(self, *args, **kwargs):
        return get_intercluster_mp2_energy_uhf(self, *args, **kwargs)

    def _get_dm_corr_energy_old(self, global_dm1=True, global_dm2=False, t_as_lambda=None):
        """Calculate correlation energy from reduced density-matrices.

        Parameters
        ----------
        global_dm1 : bool
            Use 1DM calculated from global amplitutes if True, otherwise use in cluster approximation. Default: True.
        global_dm2 : bool
            Use 2DM calculated from global amplitutes if True, otherwise use in cluster approximation. Default: False.

        Returns
        -------
        e_corr : float
            Correlation energy.
        """
        if t_as_lambda is None:
            t_as_lambda = self.opts.t_as_lambda
        if global_dm1:
            dm1a, dm1b = self._make_rdm1_ccsd_global_wf(t_as_lambda=t_as_lambda, with_mf=False)
        else:
            dm1a, dm1b = self._make_rdm1_ccsd(t_as_lambda=t_as_lambda, with_mf=False)

        # --- Core Hamiltonian + Non-cumulant 2DM contribution
        fa, fb = self.get_fock_for_energy(with_exxdiv=False)
        e1 = (einsum('pi,pq,qj,ij->', self.mo_coeff[0], fa, self.mo_coeff[0], dm1a)
            + einsum('pi,pq,qj,ij->', self.mo_coeff[1], fb, self.mo_coeff[1], dm1b))/self.ncells

        # --- Cumulant 2-DM contribution
        # Use global 2-DM
        if global_dm2:
            dm2aa, dm2ab, dm2bb = self._make_rdm2_ccsd_global_wf(t_as_lambda=t_as_lambda, with_dm1=False)
            eriaa = self.get_eris_array(self.mo_coeff[0])
            e2 = einsum('pqrs,pqrs', eriaa, dm2aa) / 2
            eriab = self.get_eris_array(2*[self.mo_coeff[0]] + 2*[self.mo_coeff[1]])
            e2 += einsum('pqrs,pqrs', eriab, dm2ab)
            eribb = self.get_eris_array(self.mo_coeff[1])
            e2 += einsum('pqrs,pqrs', eribb, dm2bb) / 2
        # Use fragment-local 2-DM
        else:
            e2 = self.get_dm_corr_energy_e2(t_as_lambda=t_as_lambda)
        e_corr = (e1 + e2)
        return e_corr
