import numpy as np

import pyscf
import pyscf.lo

from vayesta.core.util import *
from .iao import IAO_Fragmentation

class IAOPAO_Fragmentation(IAO_Fragmentation):

    name = "IAO/PAOs"

    def __init__(self, qemb, minao='minao'):
        super().__init__(qemb, minao=minao)

    def get_coeff(self):
        """Make projected atomic orbitals (PAOs)."""
        iao_coeff = super().get_coeff(add_virtuals=False)

        core, valence, rydberg = pyscf.lo.nao._core_val_ryd_list(self.mol)
        # In case a minimal basis set is used:
        if not rydberg:
            return np.zeros((self.nao, 0))

        # "Representation of Rydberg-AOs in terms of AOs"
        pao_coeff = np.eye(self.nao)[:,rydberg]
        # Project AOs onto non-IAO space:
        # (S^-1 - C.CT) . S = (1 - C.CT.S)
        ovlp = self.get_ovlp()
        p_pao = np.eye(self.nao) - dot(iao_coeff, iao_coeff.T, ovlp)
        pao_coeff = np.dot(p_pao, pao_coeff)

        # Orthogonalize PAOs:
        x, e_min = self.get_lowdin_orth_x(pao_coeff, ovlp)
        self.log.debug("Lowdin orthogonalization of PAOs: n(in)= %3d -> n(out)= %3d , e(min)= %.3e",
                x.shape[0], x.shape[1], e_min)
        if e_min < 1e-12:
            self.log.warning("Small eigenvalue in Lowdin-orthogonalization: %.3e !", e_min)
        pao_coeff = np.dot(pao_coeff, x)

        coeff = np.hstack((iao_coeff, pao_coeff))
        assert (coeff.shape[-1] == self.mf.mo_coeff.shape[-1])
        # Test orthogonality of IAO+PAO
        self.check_orth(coeff, "IAO+PAO")

        return coeff

    def get_labels(self):
        iao_labels = super().get_labels()
        core, valence, rydberg = pyscf.lo.nao._core_val_ryd_list(self.mol)
        pao_labels = (np.asarray(self.mol.ao_labels(None), dtype=object)[rydberg]).tolist()
        labels = iao_labels + pao_labels
        return labels
