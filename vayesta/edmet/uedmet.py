import numpy as np

from vayesta.dmet import UDMET
from vayesta.edmet import REDMET
from vayesta.edmet.ufragment import UEDMETFragment as Fragment


class UEDMET(REDMET, UDMET):
    @property
    def eps(self):
        noa, nob = self.nocc
        nva, nvb = self.nvir

        epsa = np.zeros((noa, nva))
        epsa = epsa + self.mo_energy[0, noa:]
        epsa = (epsa.T - self.mo_energy[0, :noa]).T
        epsa = epsa.reshape(-1)

        epsb = np.zeros((nob, nvb))
        epsb = epsb + self.mo_energy[1, nob:]
        epsb = (epsb.T - self.mo_energy[1, :nob]).T
        epsb = epsb.reshape(-1)

        return epsa, epsb

    Fragment = Fragment
