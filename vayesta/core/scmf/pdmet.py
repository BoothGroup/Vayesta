import numpy as np
from vayesta.core.util import dot, fix_orbital_sign
from vayesta.core.scmf.scmf import SCMF


class PDMET_RHF(SCMF):
    name = "p-DMET"

    def __init__(self, *args, dm_type="default", **kwargs):
        super().__init__(*args, **kwargs)
        self.dm_type = dm_type.lower()

    def get_rdm1(self):
        """DM1 in MO basis."""
        dm_type = self.dm_type
        if dm_type.startswith("default"):
            try:
                dm1 = self.emb.make_rdm1()
            except NotImplementedError:
                dm1 = self.emb.make_rdm1_demo()
        elif dm_type.startswith("demo"):
            dm1 = self.emb.make_rdm1_demo()
        else:
            raise NotImplementedError("dm_type= %r" % dm_type)
        # Check electron number
        nelec_err = abs(np.trace(dm1) - self.emb.mol.nelectron)
        if nelec_err > 1e-5:
            self.log.warning("Large electron error in 1DM= %.3e", nelec_err)
        return dm1

    def update_mo_coeff(self, mo_coeff, mo_occ, diis=None, dm1=None, mo_orig=None):
        if dm1 is None:
            dm1 = self.get_rdm1()
        if mo_orig is None:
            mo_orig = self._mo_orig
        # Transform to original MO basis
        r = dot(mo_coeff.T, self.emb.get_ovlp(), mo_orig)
        dm1 = dot(r.T, dm1, r)
        if diis is not None:
            dm1 = diis.update(dm1)
        mo_occ_new, rot = np.linalg.eigh(dm1)
        mo_occ_new, rot = mo_occ_new[::-1], rot[:, ::-1]
        nocc = np.count_nonzero(mo_occ > 0)
        if abs(mo_occ_new[nocc - 1] - mo_occ_new[nocc]) < 1e-8:
            self.log.critical("p-DMET MO occupation numbers (occupied):\n%s", mo_occ_new[:nocc])
            self.log.critical("p-DMET MO occupation numbers (virtual):\n%s", mo_occ_new[nocc:])
            raise RuntimeError("Degeneracy in MO occupation!")
        else:
            self.log.debug("p-DMET MO occupation numbers (occupied):\n%s", mo_occ_new[:nocc])
            self.log.debug("p-DMET MO occupation numbers (virtual):\n%s", mo_occ_new[nocc:])
        mo_coeff = np.dot(mo_orig, rot)
        mo_coeff = fix_orbital_sign(mo_coeff)[0]
        return mo_coeff


class PDMET_UHF(PDMET_RHF):

    def get_diis(self):
        """Two separate DIIS objects for alpha and beta orbitals."""
        return super().get_diis(), super().get_diis()

    def get_rdm1(self):
        """DM1 in MO basis."""
        dm_type = self.dm_type
        if dm_type.startswith("default"):
            dm1 = self.emb.make_rdm1()
        elif dm_type.startswith("demo"):
            dm1 = self.emb.make_rdm1_demo()
        else:
            raise NotImplementedError("dm_type= %r" % dm_type)
        # Check electron number
        nelec_err = abs(np.trace(dm1[0] + dm1[1]) - self.emb.mol.nelectron)
        if nelec_err > 1e-5:
            self.log.warning("Large electron error in 1DM= %.3e", nelec_err)
        # Check spin
        spin_err = abs(np.trace(dm1[0] - dm1[1]) - self.emb.mol.spin)
        if spin_err > 1e-5:
            self.log.warning("Large spin error in 1DM= %.3e", spin_err)
        return dm1

    def update_mo_coeff(self, mo_coeff, mo_occ, diis=None, dm1=None, mo_orig=None):
        if dm1 is None:
            dm1a, dm1b = self.get_rdm1()
        else:
            dm1a, dm1b = dm1
        if diis is not None:
            diisa, diisb = diis
        else:
            diisa = diisb = None
        if mo_orig is None:
            mo_orig = self._mo_orig
        self.log.debug("Updating alpha MOs")
        mo_coeff_new_a = super().update_mo_coeff(mo_coeff[0], mo_occ[0], diis=diisa, dm1=dm1a, mo_orig=mo_orig[0])
        self.log.debug("Updating beta MOs")
        mo_coeff_new_b = super().update_mo_coeff(mo_coeff[1], mo_occ[1], diis=diisb, dm1=dm1b, mo_orig=mo_orig[1])
        mo_coeff_new = (mo_coeff_new_a, mo_coeff_new_b)
        return mo_coeff_new
