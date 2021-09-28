import numpy as np

from vayesta.core.util import *
from .fragmentation import Fragmentation

class SAO_Fragmentation(Fragmentation):

    name = "SAO"

    def get_coeff(self):
        ovlp = self.get_ovlp()
        idt = np.eye(self.nao)
        if np.allclose(ovlp, idt):
            return idt
        x, e_min = self.get_lowdin_orth_x(idt, ovlp)
        self.log.debug("Lowdin orthogonalization of AOs: n(in)= %3d -> n(out)= %3d , e(min)= %.3e",
                x.shape[0], x.shape[1], e_min)
        if e_min < 1e-12:
            self.log.warning("Small eigenvalue in Lowdin-orthogonalization: %.3e !", e_min)
        return x

    def get_labels(self):
        return self.mol.ao_labels(None)
