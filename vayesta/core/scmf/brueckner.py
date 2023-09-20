import numpy as np
import scipy
import scipy.linalg

# from vayesta.misc import PCDIIS
from vayesta.core.util import dot, fix_orbital_sign
from vayesta.core.scmf.scmf import SCMF


class Brueckner_RHF(SCMF):
    name = "Brueckner"

    def __init__(self, *args, diis_obj="dm1", **kwargs):
        super().__init__(*args, **kwargs)
        self.diis_obj = diis_obj.lower()

    # def get_diis(self):
    #    """PC-DIIS"""
    #    nocc = np.count_nonzero(self.mf.mo_occ > 0)
    #    diis = PCDIIS(self._mo_orig[:,:nocc].copy())
    #    return diis

    def get_t1(self):
        """Get global T1 amplitudes from quantum embedding calculation."""
        return self.emb.get_global_t1()

    def update_mo_coeff(self, mo_coeff, mo_occ, diis=None, t1=None, mo_orig=None):
        """Get new MO coefficients."""
        if t1 is None:
            t1 = self.get_t1()
        if mo_orig is None:
            mo_orig = self._mo_orig
        self.log.debug("Norm of T1: L(2)= %.3e  L(inf)= %.3e", np.linalg.norm(t1), abs(t1).max())
        nocc, nvir = t1.shape
        nmo = nocc + nvir
        occ, vir = np.s_[:nocc], np.s_[nocc:]
        ovlp = self.emb.get_ovlp()
        # Perform DIIS in original MO basis, then transform back:
        if diis is not None and self.diis_obj == "t1":
            ro = dot(mo_coeff[:, occ].T, ovlp, mo_orig)
            rv = dot(mo_coeff[:, vir].T, ovlp, mo_orig)
            t1 = dot(ro.T, t1, rv)
            t1 = diis.update(t1, xerr=t1)
            ## Transform back
            t1 = dot(ro, t1, rv.T)

        mo_change = (1 - self.damping) * np.dot(mo_coeff[:, vir], t1.T)
        self.log.debug("Change of occupied Brueckner orbitals= %.3e", np.linalg.norm(mo_change))
        bmo_occ = mo_coeff[:, occ] + mo_change

        # Orthogonalize occupied orbitals
        # If there was no AO-overlap matrix:  bmo_occ = np.linalg.qr(bmo_occ)[0]
        dm_occ = np.dot(bmo_occ, bmo_occ.T)
        e, v = scipy.linalg.eigh(dm_occ, b=ovlp, type=2)
        bmo_occ = v[:, -nocc:]
        # DIIS of occupied density
        if diis and self.diis_obj == "dm1":
            dm_occ = np.dot(bmo_occ, bmo_occ.T)
            r = np.dot(ovlp, mo_orig)
            dm_occ = dot(r.T, dm_occ, r)
            dm_occ = diis.update(dm_occ)
            e, v = np.linalg.eigh(dm_occ)
            bmo_occ = np.dot(mo_orig, v)[:, -nocc:]

        # Virtual space
        r = dot(mf.mo_coeff.T, ovlp, bmo_occ)
        e, v = np.linalg.eigh(np.dot(r, r.T))
        assert np.allclose(e[:nvir], 0)
        assert np.allclose(e[nvir:], 1)
        bmo_vir = np.dot(mf.mo_coeff, v)[:, :nvir]

        assert (bmo_occ.shape[-1] == nocc) and (bmo_vir.shape[-1] == nvir)
        mo_coeff_new = np.hstack((bmo_occ, bmo_vir))
        mo_coeff_new = fix_orbital_sign(mo_coeff_new)[0]
        return mo_coeff_new


class Brueckner_UHF(Brueckner_RHF):

    def get_diis(self):
        """Two separate DIIS objects for alpha and beta orbitals."""
        return super().get_diis(), super().get_diis()

    def update_mo_coeff(self, mo_coeff, mo_occ, diis=None, t1=None, mo_orig=None):
        if t1 is None:
            t1a, t1b = self.get_t1()
        else:
            t1a, t1b = t1
        if diis is not None:
            diisa, diisb = diis
        else:
            diisa = diisb = None
        if mo_orig is None:
            mo_orig = self._mo_orig
        self.log.debug("Updating alpha MOs")
        mo_coeff_new_a = super().update_mo_coeff(mo_coeff[0], mo_occ[0], diis=diisa, t1=t1a, mo_orig=mo_orig[0])
        self.log.debug("Updating beta MOs")
        mo_coeff_new_b = super().update_mo_coeff(mo_coeff[1], mo_occ[1], diis=diisb, t1=t1b, mo_orig=mo_orig[1])
        mo_coeff_new = (mo_coeff_new_a, mo_coeff_new_b)
        return mo_coeff_new
