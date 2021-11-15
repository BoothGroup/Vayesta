import numpy as np

from vayesta.core import UEmbedding
from vayesta.core.util import *

from vayesta.ewf import REWF
from vayesta.ewf.ufragment import UEWFFragment as Fragment

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
