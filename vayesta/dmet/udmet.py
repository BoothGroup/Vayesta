import numpy as np

from vayesta.core import UEmbedding
from vayesta.core.util import *

from vayesta.dmet import RDMET
from vayesta.dmet.ufragment import UDMETFragment as Fragment

from .sdp_sc import perform_SDP_fit


class UDMET(UEmbedding, RDMET):
    Fragment = Fragment

    def update_vcorr(self, fock):
        impurity_coeffs = self.get_impurity_coeffs()
        curr_rdms, delta_rdms = self.updater.update(self.hl_rdms)
        self.log.info("Change in high-level RDMs: {:6.4e}".format(delta_rdms))
        # Now for the DMET self-consistency!
        self.log.info("Now running DMET correlation potential fitting")
        # Note that we want the total number of electrons, not just in fragments, and that this uses spatial orbitals.
        vcorr_new = np.array((perform_SDP_fit(self.mol.nelec[0], fock[0], impurity_coeffs[0], [x[0] for x in curr_rdms],
                                             self.get_ovlp(), self.log),
                             perform_SDP_fit(self.mol.nelec[1], fock[1], impurity_coeffs[1], [x[1] for x in curr_rdms],
                                             self.get_ovlp(), self.log)))

        return vcorr_new

    def get_impurity_coeffs(self):
        sym_parents = self.get_symmetry_parent_fragments()
        sym_children = self.get_symmetry_child_fragments()
        return [
            [
                [parent.c_frag[x]] + [c.c_frag[x] for c in children] for (parent, children) in zip(sym_parents,
                                                                                                   sym_children)]
            for x in [0, 1]
        ]
