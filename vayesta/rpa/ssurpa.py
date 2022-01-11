from vayesta.rpa.ssrpa import ssRPA
import numpy as np
import scipy.linalg

from timeit import default_timer as timer
from vayesta.core.util import *

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


    def _gen_arrays(self, xc_kernel=None):
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

        vaa, vab, vbb = self.ao2mo()
        # Get coulomb interaction in occupied-virtual space.
        vaa = vaa[:nocc_a, nocc_a:, :nocc_a, nocc_a:].reshape((self.ova, self.ova))
        vab = vab[:nocc_a, nocc_a:, :nocc_b, nocc_b:].reshape((self.ova, self.ovb))
        vbb = vbb[:nocc_b, nocc_b:, :nocc_b, nocc_b:].reshape((self.ovb, self.ovb))

        ApB = np.zeros((self.ov, self.ov))
        ApB[:self.ova, :self.ova] = 2 * vaa
        ApB[:self.ova, self.ova:] = 2 * vab
        ApB[self.ova:, :self.ova] = 2 * vab.T
        ApB[self.ova:, self.ova:] = 2 * vbb
        # At this point AmB is just epsilon so add in.
        ApB[np.diag_indices_from(ApB)] += AmB

        if xc_kernel is None:
            M = np.einsum("p,pq,q->pq", AmB ** (0.5), ApB, AmB ** (0.5))
        else:
            AmB = np.diag(AmB)
            # Grab A and B contributions for XC kernel.
            c_o = self.mo_coeff_occ
            c_v = self.mo_coeff_vir
            V_A_aa = einsum("pqrs,pi,qa,rj,sb->iajb", xc_kernel[0], c_o, c_v, c_o, c_v).reshape((ova, ova))
            ApB[:self.ova, :self.ova] += V_A_aa
            AmB[:self.ova, :self.ova] += V_A_aa
            del V_A_aa
            V_B_aa = einsum("pqsr,pi,qa,rj,sb->iajb", xc_kernel[0], c_o, c_v, c_o, c_v).reshape((ova, ova))
            ApB[:self.ova, :self.ova] += V_B_aa
            AmB[:self.ova, :self.ova] -= V_B_aa
            del V_B_aa
            V_A_ab = einsum("pqrs,pi,qa,rj,sb->iajb", xc_kernel[1], c_o, c_v, c_o, c_v).reshape((ova, ovb))
            ApB[:self.ova, self.ova:] += V_A_ab
            ApB[self.ova:, :self.ova] += V_A_ab.T
            AmB[:self.ova, self.ova:] += V_A_ab
            AmB[self.ova:, :self.ova] += V_A_ab.T
            del V_A_ab
            V_B_ab = einsum("pqsr,pi,qa,rj,sb->iajb", xc_kernel[1], c_o, c_v, c_o, c_v).reshape((ova, ovb))
            ApB[:self.ova, self.ova:] += V_B_ab
            ApB[self.ova:, :self.ova] += V_B_ab.T
            AmB[:self.ova, self.ova:] -= V_B_ab
            AmB[self.ova:, :self.ova] -= V_B_ab.T
            del V_B_ab
            V_A_bb = einsum("pqrs,pi,qa,rj,sb->iajb", xc_kernel[2], c_o, c_v, c_o, c_v).reshape((ovb, ovb))
            ApB[self.ova:, self.ova:] += V_A_bb
            AmB[self.ova:, self.ova:] += V_A_bb
            del V_A_bb
            V_B_bb = einsum("pqsr,pi,qa,rj,sb->iajb", xc_kernel[2], c_o, c_v, c_o, c_v).reshape((ovb, ovb))
            ApB[self.ova:, self.ova:] += V_B_bb
            AmB[self.ova:, self.ova:] -= V_B_bb
            del V_B_bb
            AmBrt = scipy.linalg.sqrtm(AmB)
            M = dot(AmBrt, ApB, AmBrt)

        self.log.timing("Time to build RPA arrays: %s", time_string(timer() - t0))
        return M, AmB, ApB, (epsa, epsb), (vaa, vbb)

    def ao2mo(self, mo_coeff = None):
        """Get the ERIs in MO basis
        """
        mo_coeff = self.mo_coeff if mo_coeff is None else mo_coeff
        # Call three-times to spin-restricted embedding
        self.log.debugv("Making (aa|aa) ERIs...")
        eris_aa = super().ao2mo(mo_coeff[0])
        self.log.debugv("Making (bb|bb) ERIs...")
        eris_bb = super().ao2mo(mo_coeff[1])
        self.log.debugv("Making (aa|bb) ERIs...")
        eris_ab = super().ao2mo((mo_coeff[0], mo_coeff[0], mo_coeff[1], mo_coeff[1]))
        return (eris_aa, eris_ab, eris_bb)



