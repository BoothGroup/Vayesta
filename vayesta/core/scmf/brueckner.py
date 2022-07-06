import numpy as np
import scipy
import scipy.linalg
#from vayesta.misc import PCDIIS
from vayesta.core.util import *
from vayesta.core.scmf.scmf import SCMF


class Brueckner_RHF(SCMF):

    name = "Brueckner"

    def __init__(self, *args, diis_obj='dm1', **kwargs):
        super().__init__(*args, **kwargs)
        self.diis_obj = diis_obj.lower()

    #def get_diis(self):
    #    """PC-DIIS"""
    #    nocc = np.count_nonzero(self.mf.mo_occ > 0)
    #    diis = PCDIIS(self._mo_orig[:,:nocc].copy())
    #    return diis

    def get_t1(self):
        """Get global T1 amplitudes from quantum embedding calculation."""
        return self.emb.get_global_t1()

    def update_mo_coeff(self, mf, diis=None):
        """Get new MO coefficients."""
        t1 = self.get_t1()
        self.log.debug("Norm of T1: L(2)= %.3e  L(inf)= %.3e", np.linalg.norm(t1), abs(t1).max())
        nocc, nvir = t1.shape
        nmo = (nocc + nvir)
        occ, vir = np.s_[:nocc], np.s_[nocc:]
        ovlp = self.emb.get_ovlp()
        # Perform DIIS in original MO basis, then transform back:
        if diis and self.diis_obj == 't1':
            ro = dot(mf.mo_coeff[:,occ].T, ovlp, self._mo_orig)
            rv = dot(mf.mo_coeff[:,vir].T, ovlp, self._mo_orig)
            t1 = dot(ro.T, t1, rv)
            t1 = diis.update(t1, xerr=t1)
            ## Transform back
            t1 = dot(ro, t1, rv.T)

        mo_change = (1-self.damping)*np.dot(mf.mo_coeff[:,vir], t1.T)
        self.log.debug("Change of occupied Brueckner orbitals= %.3e", np.linalg.norm(mo_change))
        bmo_occ = (mf.mo_coeff[:,occ] + mo_change)

        # Orthogonalize occupied orbitals
        # If there was no AO-overlap matrix:  bmo_occ = np.linalg.qr(bmo_occ)[0]
        dm_occ = np.dot(bmo_occ, bmo_occ.T)
        e, v = scipy.linalg.eigh(dm_occ, b=ovlp, type=2)
        bmo_occ = v[:,-nocc:]
        # DIIS of occupied density
        if diis and self.diis_obj == 'dm1':
            dm_occ = np.dot(bmo_occ, bmo_occ.T)
            r = np.dot(ovlp, self._mo_orig)
            dm_occ = dot(r.T, dm_occ, r)
            dm_occ = diis.update(dm_occ)
            e, v = np.linalg.eigh(dm_occ)
            bmo_occ = np.dot(self._mo_orig, v)[:,-nocc:]

        # Virtual space
        dm_vir = (np.linalg.inv(ovlp) - np.dot(bmo_occ, bmo_occ.T))
        e, v = scipy.linalg.eigh(dm_vir, b=ovlp, type=2)
        bmo_vir = v[:,-nvir:]

        assert (bmo_occ.shape[-1] == nocc) and (bmo_vir.shape[-1] == nvir)
        mo_coeff = np.hstack((bmo_occ, bmo_vir))
        mo_coeff = fix_orbital_sign(mo_coeff)[0]
        return mo_coeff

class Brueckner_UHF(Brueckner_RHF):

    def update_mo_coeff(self, mf, diis=None):
        t1a, t1b = self.get_t1()
        self.log.debug("Norm of alpha/beta-T1 L(2)= %.3e %.3e L(inf)= %.3e %.3e",
                np.linalg.norm(t1a), np.linalg.norm(t1b), abs(t1a).max(), abs(t1b).max())
        nocca, nvira = t1a.shape
        noccb, nvirb = t1b.shape
        nmoa, nmob = nocca + nvira, noccb + nvirb
        occa, vira = np.s_[:nocca], np.s_[nocca:]
        occb, virb = np.s_[:noccb], np.s_[noccb:]
        ovlp = self.emb.get_ovlp()

        # Perform DIIS in original MO basis, then transform back:
        if diis and self.diis_obj == 't1':
            roa = dot(mf.mo_coeff[0][:,occa].T, ovlp, self._mo_orig[0])
            rob = dot(mf.mo_coeff[1][:,occb].T, ovlp, self._mo_orig[1])
            rva = dot(mf.mo_coeff[0][:,vira].T, ovlp, self._mo_orig[0])
            rvb = dot(mf.mo_coeff[1][:,virb].T, ovlp, self._mo_orig[1])

            t1a = dot(roa.T, t1a, rva)
            t1b = dot(rob.T, t1b, rvb)
            t1a, t1b = diis.update(np.asarry((t1a,t1b)), xerr=np.asarray((t1a,t1b)))
            #t1b = diis.update(t1b, xerr=t1b)
            ## Transform back
            t1a = dot(roa, t1a, rva.T)
            t1b = dot(rob, t1b, rvb.T)

        mo_change_a = (1-self.damping)*np.dot(mf.mo_coeff[0][:,vira], t1a.T)
        mo_change_b = (1-self.damping)*np.dot(mf.mo_coeff[1][:,virb], t1b.T)
        self.log.debug("Change of alpha/beta occupied Brueckner orbitals= %.3e %.3e",
                np.linalg.norm(mo_change_a), np.linalg.norm(mo_change_b))
        bmo_occ_a = (mf.mo_coeff[0][:,occa] + mo_change_a)
        bmo_occ_b = (mf.mo_coeff[1][:,occb] + mo_change_b)

        # Orthogonalize occupied orbitals
        # If there was no AO-overlap matrix:  bmo_occ = np.linalg.qr(bmo_occ)[0]
        dm_occ_a = np.dot(bmo_occ_a, bmo_occ_a.T)
        dm_occ_b = np.dot(bmo_occ_b, bmo_occ_b.T)
        ea, va = scipy.linalg.eigh(dm_occ_a, b=ovlp, type=2)
        eb, vb = scipy.linalg.eigh(dm_occ_b, b=ovlp, type=2)
        bmo_occ_a = va[:,-nocca:]
        bmo_occ_b = vb[:,-noccb:]

        # DIIS of occupied density
        if diis and self.diis_obj == 'dm1':
            dm_occ_a = np.dot(bmo_occ_a, bmo_occ_a.T)
            dm_occ_b = np.dot(bmo_occ_b, bmo_occ_b.T)
            ra = np.dot(ovlp, self._mo_orig[0])
            rb = np.dot(ovlp, self._mo_orig[1])
            dm_occ_a = dot(ra.T, dm_occ_a, ra)
            dm_occ_b = dot(rb.T, dm_occ_b, rb)
            #dm_occ_a = diis.update(dm_occ_a)
            #dm_occ_b = diis.update(dm_occ_b)
            dm_occ_a, dm_occ_b = diis.update(np.asarray((dm_occ_a, dm_occ_b)))
            ea, va = np.linalg.eigh(dm_occ_a)
            eb, vb = np.linalg.eigh(dm_occ_b)
            bmo_occ_a = np.dot(self._mo_orig[0], va)[:,-nocca:]
            bmo_occ_b = np.dot(self._mo_orig[1], vb)[:,-noccb:]

        # Virtual space
        dm_vir_a = (np.linalg.inv(ovlp) - np.dot(bmo_occ_a, bmo_occ_a.T))
        dm_vir_b = (np.linalg.inv(ovlp) - np.dot(bmo_occ_b, bmo_occ_b.T))
        ea, va = scipy.linalg.eigh(dm_vir_a, b=ovlp, type=2)
        eb, vb = scipy.linalg.eigh(dm_vir_b, b=ovlp, type=2)
        bmo_vir_a = va[:,-nvira:]
        bmo_vir_b = vb[:,-nvirb:]

        assert (bmo_occ_a.shape[-1] == nocca) and (bmo_vir_a.shape[-1] == nvira)
        assert (bmo_occ_b.shape[-1] == noccb) and (bmo_vir_b.shape[-1] == nvirb)
        mo_coeff_a = np.hstack((bmo_occ_a, bmo_vir_a))
        mo_coeff_b = np.hstack((bmo_occ_b, bmo_vir_b))
        return (mo_coeff_a, mo_coeff_b)
