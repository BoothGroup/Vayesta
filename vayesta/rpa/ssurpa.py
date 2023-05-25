from vayesta.rpa.ssrpa import ssRPA
import numpy as np
import scipy.linalg

from timeit import default_timer as timer
from vayesta.core.util import dot, time_string

import pyscf.ao2mo


class ssURPA(ssRPA):
    @property
    def norb(self):
        return self.mf.mo_coeff.shape[1]

    @property
    def nocc(self):
        return self.mf.nelec

    @property
    def nvir(self):
        return tuple([self.norb - x for x in self.nocc])

    @property
    def ova(self):
        oa, ob = self.nocc
        va, vb = self.nvir
        return oa * va

    @property
    def ovb(self):
        oa, ob = self.nocc
        va, vb = self.nvir
        return ob * vb

    @property
    def mo_coeff_occ(self):
        """Occupied MO coefficients."""
        na, nb = self.nocc
        return self.mo_coeff[0, :, :na], self.mo_coeff[1, :, :nb]

    @property
    def mo_coeff_vir(self):
        """Virtual MO coefficients."""
        na, nb = self.nocc
        return self.mo_coeff[0, :, na:], self.mo_coeff[1, :, nb:]

    def _gen_arrays(self, xc_kernel=None, alpha=1.0):
        t0 = timer()

        nocc_a, nocc_b = self.nocc
        nvir_a, nvir_b = self.nvir
        # Only have diagonal components in canonical basis.
        epsa = np.zeros((nocc_a, nvir_a))
        epsa = epsa + self.mf.mo_energy[0, nocc_a:]
        epsa = (epsa.T - self.mf.mo_energy[0, :nocc_a]).T
        epsa = epsa.reshape((self.ova,))

        epsb = np.zeros((nocc_a, nvir_a))
        epsb = epsb + self.mf.mo_energy[1, nocc_b:]
        epsb = (epsb.T - self.mf.mo_energy[1, :nocc_b]).T
        epsb = epsb.reshape((self.ovb,))

        AmB = np.concatenate([epsa, epsb])
        fullv = self.get_k()
        ApB = 2 * fullv * alpha
        # At this point AmB is just epsilon so add in.
        ApB[np.diag_indices_from(ApB)] += AmB

        if xc_kernel is None:
            M = np.einsum("p,pq,q->pq", AmB ** (0.5), ApB, AmB ** (0.5))
        else:
            # Grab A and B contributions for XC kernel.
            ApB_xc, AmB_xc = self.get_xc_contribs(
                xc_kernel, self.mo_coeff_occ, self.mo_coeff_vir, alpha
            )
            ApB = ApB + ApB_xc
            AmB = np.diag(AmB) + AmB_xc
            del ApB_xc, AmB_xc
            AmBrt = scipy.linalg.sqrtm(AmB)
            M = dot(AmBrt, ApB, AmBrt)

        self.log.timing("Time to build RPA arrays: %s", time_string(timer() - t0))
        return M, AmB, ApB, (epsa, epsb), fullv

    def get_k(self):
        vaa, vab, vbb = self.ao2mo()

        nocc_a, nocc_b = self.nocc
        nvir_a, nvir_b = self.nvir
        # Get coulomb interaction in occupied-virtual space.
        vaa = vaa[:nocc_a, nocc_a:, :nocc_a, nocc_a:].reshape((self.ova, self.ova))
        vab = vab[:nocc_a, nocc_a:, :nocc_b, nocc_b:].reshape((self.ova, self.ovb))
        vbb = vbb[:nocc_b, nocc_b:, :nocc_b, nocc_b:].reshape((self.ovb, self.ovb))

        fullv = np.zeros((self.ov, self.ov))
        fullv[: self.ova, : self.ova] = vaa
        fullv[: self.ova, self.ova :] = vab
        fullv[self.ova :, : self.ova] = vab.T
        fullv[self.ova :, self.ova :] = vbb
        return fullv

    def ao2mo(self, mo_coeff=None):
        """Get the ERIs in MO basis"""
        mo_coeff = self.mo_coeff if mo_coeff is None else mo_coeff
        # Call three-times to spin-restricted embedding
        self.log.debugv("Making (aa|aa) ERIs...")
        eris_aa = super().ao2mo(mo_coeff[0])
        self.log.debugv("Making (bb|bb) ERIs...")
        eris_bb = super().ao2mo(mo_coeff[1])
        self.log.debugv("Making (aa|bb) ERIs...")
        eris_ab = super().ao2mo((mo_coeff[0], mo_coeff[0], mo_coeff[1], mo_coeff[1]))
        return (eris_aa, eris_ab, eris_bb)
