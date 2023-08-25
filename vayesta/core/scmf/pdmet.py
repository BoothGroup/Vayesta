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
            dm1 = self.emb.make_rdm1()
        elif dm_type.startswith("demo"):
            dm1 = self.emb.make_rdm1_demo()
        else:
            raise NotImplementedError("dm_type= %r" % dm_type)
        # Check electron number
        nelec_err = abs(np.trace(dm1) - self.emb.mol.nelectron)
        if nelec_err > 1e-5:
            self.log.warning("Large electron error in 1DM= %.3e", nelec_err)
        return dm1

    def update_mo_coeff(self, mf, diis=None):
        dm1 = self.get_rdm1()
        # Transform to original MO basis
        r = dot(self.emb.mo_coeff.T, self.emb.get_ovlp(), self._mo_orig)
        dm1 = dot(r.T, dm1, r)
        if diis is not None:
            dm1 = diis.update(dm1)
        mo_occ, rot = np.linalg.eigh(dm1)
        mo_occ, rot = mo_occ[::-1], rot[:, ::-1]
        nocc = np.count_nonzero(mf.mo_occ > 0)
        if abs(mo_occ[nocc - 1] - mo_occ[nocc]) < 1e-8:
            self.log.critical("p-DMET MO occupation numbers (occupied):\n%s", mo_occ[:nocc])
            self.log.critical("p-DMET MO occupation numbers (virtual):\n%s", mo_occ[nocc:])
            raise RuntimeError("Degeneracy in MO occupation!")
        else:
            self.log.debug("p-DMET MO occupation numbers (occupied):\n%s", mo_occ[:nocc])
            self.log.debug("p-DMET MO occupation numbers (virtual):\n%s", mo_occ[nocc:])
        mo_coeff = np.dot(self._mo_orig, rot)
        mo_coeff = fix_orbital_sign(mo_coeff)[0]
        return mo_coeff


class PDMET_UHF(PDMET_RHF):
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

    def update_mo_coeff(self, mf, diis=None):
        dma, dmb = self.get_rdm1()
        mo_coeff = self.emb.mo_coeff
        ovlp = self.emb.get_ovlp()
        # Transform DM to original MO basis
        ra = dot(mo_coeff[0].T, ovlp, self._mo_orig[0])
        rb = dot(mo_coeff[1].T, ovlp, self._mo_orig[1])
        dma = dot(ra.T, dma, ra)
        dmb = dot(rb.T, dmb, rb)
        if diis is not None:
            assert dma.shape == dmb.shape
            dma, dmb = diis.update(np.asarray((dma, dmb)))
        mo_occ_a, rot_a = np.linalg.eigh(dma)
        mo_occ_b, rot_b = np.linalg.eigh(dmb)
        mo_occ_a, rot_a = mo_occ_a[::-1], rot_a[:, ::-1]
        mo_occ_b, rot_b = mo_occ_b[::-1], rot_b[:, ::-1]
        nocc_a = np.count_nonzero(mf.mo_occ[0] > 0)
        nocc_b = np.count_nonzero(mf.mo_occ[1] > 0)

        def log_occupation(logger):
            logger("p-DMET MO occupation numbers (alpha-occupied):\n%s", mo_occ_a[:nocc_a])
            logger("p-DMET MO occupation numbers (beta-occupied):\n%s", mo_occ_b[:nocc_b])
            logger("p-DMET MO occupation numbers (alpha-virtual):\n%s", mo_occ_a[nocc_a:])
            logger("p-DMET MO occupation numbers (beta-virtual):\n%s", mo_occ_b[nocc_b:])

        if min(abs(mo_occ_a[nocc_a - 1] - mo_occ_a[nocc_a]), abs(mo_occ_b[nocc_b - 1] - mo_occ_b[nocc_b])) < 1e-8:
            log_occupation(self.log.critical)
            raise RuntimeError("Degeneracy in MO occupation!")
        log_occupation(self.log.debugv)

        mo_coeff_a = np.dot(self._mo_orig[0], rot_a)
        mo_coeff_b = np.dot(self._mo_orig[1], rot_b)
        mo_coeff_a = fix_orbital_sign(mo_coeff_a)[0]
        mo_coeff_b = fix_orbital_sign(mo_coeff_b)[0]
        return (mo_coeff_a, mo_coeff_b)
