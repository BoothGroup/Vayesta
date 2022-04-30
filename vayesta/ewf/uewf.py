import numpy as np

from vayesta.core import UEmbedding
from vayesta.core.util import *

from vayesta.ewf import REWF
from vayesta.ewf.ufragment import UEWFFragment as Fragment
from vayesta.core.mpi import mpi

# Amplitudes
from .amplitudes import get_global_t1_uhf
from .amplitudes import get_global_t2_uhf
# Density-matrices
from .urdm import make_rdm1_ccsd
from .urdm import make_rdm2_ccsd_proj_lambda
from .icmp2 import get_intercluster_mp2_energy_uhf


class UEWF(REWF, UEmbedding):

    Fragment = Fragment

    def get_init_mo_coeff(self, mo_coeff=None):
        """Orthogonalize insufficiently orthogonal MOs.

        (For example as a result of k2gamma conversion with low cell.precision)
        """
        if mo_coeff is None: mo_coeff = self.mo_coeff
        c = mo_coeff.copy()
        ovlp = self.get_ovlp()
        assert np.all(c.imag == 0), "max|Im(C)|= %.2e" % abs(c.imag).max()

        for s, spin in enumerate(('alpha', 'beta')):
            err = abs(dot(c[s].T, ovlp, c[s]) - np.eye(c[s].shape[-1])).max()
            if err > 1e-5:
                self.log.error("Orthogonality error of %s-MOs= %.2e !!!", spin, err)
            else:
                self.log.debug("Orthogonality error of %s-MOs= %.2e", spin, err)
        if self.opts.orthogonal_mo_tol and err > self.opts.orthogonal_mo_tol:
            raise NotImplementedError()
            #t0 = timer()
            #self.log.info("Orthogonalizing orbitals...")
            #c_orth = helper.orthogonalize_mo(c, ovlp)
            #change = abs(einsum('ai,ab,bi->i', c_orth, ovlp, c)-1)
            #self.log.info("Max. orbital change= %.2e%s", change.max(), " (!!!)" if change.max() > 1e-4 else "")
            #self.log.timing("Time for orbital orthogonalization: %s", time_string(timer()-t0))
            #c = c_orth
        return c

    def check_fragment_nelectron(self):
        nelec_frags = (sum([f.sym_factor*f.nelectron[0] for f in self.loop()]),
                       sum([f.sym_factor*f.nelectron[1] for f in self.loop()]))
        self.log.info("Total number of mean-field electrons over all fragments= %.8f , %.8f", *nelec_frags)
        if abs(nelec_frags[0] - np.rint(nelec_frags[0])) > 1e-4 or abs(nelec_frags[1] - np.rint(nelec_frags[1])) > 1e-4:
            self.log.warning("Number of electrons not integer!")
        return nelec_frags

    # --- CC Amplitudes
    # -----------------

    # T-amplitudes
    get_global_t1 = get_global_t1_uhf
    get_global_t2 = get_global_t2_uhf

    def t1_diagnostic(self, warn_tol=0.02):
        # Per cluster
        for f in self.get_fragments(mpi_rank=mpi.rank):
            t1 = f.results.t1
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


    # --- Density-matrices
    # --------------------

    def _make_rdm1_mp2(self, *args, **kwargs):
        return make_rdm1_ccsd(self, *args, mp2=True, **kwargs)

    def _make_rdm1_ccsd(self, *args, **kwargs):
        return make_rdm1_ccsd(self, *args, mp2=False, **kwargs)

    # TODO
    def _make_rdm2_ccsd(self, *args, **kwargs):
        raise NotImplementedError()

    def _make_rdm2_ccsd_proj_lambda(self, *args, **kwargs):
        return make_rdm2_ccsd_proj_lambda(self, *args, **kwargs)

    get_intercluster_mp2_energy = get_intercluster_mp2_energy_uhf
