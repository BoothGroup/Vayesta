import numpy as np
import pyscf.lo

from vayesta.core.util import dot
from vayesta.core import spinalg
from vayesta.core.fragmentation.iao import IAO_Fragmentation
from vayesta.core.fragmentation.iao import IAO_Fragmentation_UHF


class IAOPAO_Fragmentation(IAO_Fragmentation):
    name = "IAO/PAO"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Order according to AOs:
        self.order = None
        iaopao_labels = self.get_labels()
        ao_labels = self.mol.ao_labels(None)
        order = []
        for l in ao_labels:
            idx = iaopao_labels.index(l)
            order.append(idx)
        assert np.all([tuple(l) for l in (np.asarray(iaopao_labels, dtype=object)[order])] == ao_labels)
        self.order = order

    def get_pao_coeff(self, iao_coeff):
        core, valence, rydberg = pyscf.lo.nao._core_val_ryd_list(self.mol)
        niao = iao_coeff.shape[-1]
        npao = len(rydberg)
        if niao + npao != self.nao:
            self.log.fatal("Incorrect number of PAOs!")
            self.log.fatal("n(IAO)= %d  n(PAO)= %d  n(AO)= %d", niao, npao, self.nao)
            labels = np.asarray(self.mol.ao_labels())
            self.log.fatal("%d core AOs:\n%r", len(core), labels[core].tolist())
            self.log.fatal("%d valence AOs:\n%r", len(valence), labels[valence].tolist())
            self.log.fatal("%d Rydberg AOs:\n%r", len(rydberg), labels[rydberg].tolist())
            raise RuntimeError("Incorrect number of PAOs!")

        # In case a minimal basis set is used:
        if not rydberg:
            return np.zeros((self.nao, 0))
        # "Representation of Rydberg-AOs in terms of AOs"
        pao_coeff = np.eye(self.nao)[:, rydberg]
        # Project AOs onto non-IAO space:
        # (S^-1 - C.CT) . S = (1 - C.CT.S)
        ovlp = self.get_ovlp()
        p_pao = np.eye(self.nao) - dot(iao_coeff, iao_coeff.T, ovlp)
        pao_coeff = np.dot(p_pao, pao_coeff)

        # Orthogonalize PAOs:
        x, e_min = self.symmetric_orth(pao_coeff, ovlp)
        self.log.debugv(
            "Lowdin orthogonalization of PAOs: n(in)= %3d -> n(out)= %3d , e(min)= %.3e", x.shape[0], x.shape[1], e_min
        )
        if e_min < 1e-10:
            self.log.warning("Small eigenvalue in Lowdin orthogonalization: %.3e !", e_min)
        pao_coeff = np.dot(pao_coeff, x)
        return pao_coeff

    def get_coeff(self, order=None):
        """Make projected atomic orbitals (PAOs)."""
        if order is None:
            order = self.order
        iao_coeff = IAO_Fragmentation.get_coeff(self, add_virtuals=False)
        pao_coeff = self.get_pao_coeff(iao_coeff)
        coeff = spinalg.hstack_matrices(iao_coeff, pao_coeff)
        assert coeff.shape[-1] == self.mf.mo_coeff.shape[-1]
        # Test orthogonality of IAO+PAO
        self.check_orthonormal(coeff)
        if order is not None:
            return coeff[:, order]
        return coeff

    def get_labels(self, order=None):
        if order is None:
            order = self.order
        iao_labels = super().get_labels()
        core, valence, rydberg = pyscf.lo.nao._core_val_ryd_list(self.mol)
        pao_labels = [tuple(x) for x in np.asarray(self.mol.ao_labels(None), dtype=tuple)[rydberg]]
        labels = iao_labels + pao_labels
        if order is not None:
            return [tuple(l) for l in np.asarray(labels, dtype=object)[order]]
        return labels

    def search_labels(self, labels):
        return self.mol.search_ao_label(labels)


class IAOPAO_Fragmentation_UHF(IAOPAO_Fragmentation, IAO_Fragmentation_UHF):
    def get_coeff(self, order=None):
        """Make projected atomic orbitals (PAOs)."""
        if order is None:
            order = self.order
        iao_coeff = IAO_Fragmentation_UHF.get_coeff(self, add_virtuals=False)

        pao_coeff_a = IAOPAO_Fragmentation.get_pao_coeff(self, iao_coeff[0])
        pao_coeff_b = IAOPAO_Fragmentation.get_pao_coeff(self, iao_coeff[1])
        pao_coeff = (pao_coeff_a, pao_coeff_b)

        coeff = spinalg.hstack_matrices(iao_coeff, pao_coeff)
        assert coeff[0].shape[-1] == self.mf.mo_coeff[0].shape[-1]
        assert coeff[1].shape[-1] == self.mf.mo_coeff[1].shape[-1]
        # Test orthogonality of IAO+PAO
        self.check_orthonormal(coeff)

        if order is not None:
            return (coeff[0][:, order], coeff[1][:, order])
        return coeff


if __name__ == "__main__":
    import logging

    log = logging.getLogger(__name__)
    import pyscf.gto
    import pyscf.scf

    mol = pyscf.gto.Mole()
    mol.atom = "O 0 0 -1.2 ; C 0 0 0 ; O 0 0 1.2"
    mol.basis = "cc-pVDZ"
    mol.build()

    mf = pyscf.scf.RHF(mol)
    mf.kernel()

    iaopao = IAOPAO_Fragmentation(mf, log)

    ao_labels = mol.ao_labels(None)
    print("Atomic order")
    for i, l in enumerate(iaopao.get_labels()):
        print("%30r   vs  %30r" % (l, ao_labels[i]))
