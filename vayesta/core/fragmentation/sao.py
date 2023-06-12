import numpy as np

from vayesta.core.fragmentation.fragmentation import Fragmentation
from vayesta.core.fragmentation.ufragmentation import Fragmentation_UHF

class SAO_Fragmentation(Fragmentation):

    name = "SAO"

    def get_coeff(self):
        ovlp = self.get_ovlp()
        idt = np.eye(self.nao)
        if np.allclose(ovlp, idt):
            return idt
        x, e_min = self.symmetric_orth(idt, ovlp)
        self.log.debugv("Lowdin orthogonalization of AOs: n(in)= %3d -> n(out)= %3d , e(min)= %.3e",
                x.shape[0], x.shape[1], e_min)
        if e_min < 1e-10:
            self.log.warning("Small eigenvalue in Lowdin orthogonalization: %.3e !", e_min)
        self.check_orthonormal(x)
        return x

    def get_labels(self):
        return self.mol.ao_labels(None)

    def search_labels(self, labels):
        return self.mol.search_ao_label(labels)

class SAO_Fragmentation_UHF(Fragmentation_UHF, SAO_Fragmentation):

    def get_coeff(self):
        x = super().get_coeff()
        return (x, x)
