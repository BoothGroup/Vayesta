import numpy as np

from vayesta.core.util import *
from vayesta.core.qemb import UFragment
from vayesta.dmet.ufragment import UDMETFragment
from .fragment import EDMETFragment

class UEDMETFragment(UDMETFragment, EDMETFragment):
    def get_rot_to_mf_ov(self):
        r_o, r_v = self.get_overlap_m2c()
        spat_rota = einsum("iJ,aB->iaJB", r_o[0], r_v[0]).reshape((self.ov_mf, self.ov_active)).T
        spat_rotb = einsum("iJ,aB->iaJB", r_o[1], r_v[1]).reshape((self.ov_mf, self.ov_active)).T
        res = np.zeros((2 * self.ov_active, 2 * self.ov_mf))
        res[:self.ov_active, :self.ov_mf] = spat_rota
        res[self.ov_active:2 * self.ov_active, self.ov_mf:2 * self.ov_mf] = spat_rotb
        return res
