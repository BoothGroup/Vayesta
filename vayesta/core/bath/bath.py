"""Bath base class."""

import numpy as np


class Bath:
    def __init__(self, fragment):
        self.fragment = fragment
        assert self.spin_restricted or self.spin_unrestricted

    @property
    def spin_restricted(self):
        return np.ndim(self.mf.mo_coeff[0]) == 1

    @property
    def spin_unrestricted(self):
        return np.ndim(self.mf.mo_coeff[0]) == 2

    @property
    def spinsym(self):
        return self.fragment.spinsym

    @property
    def mf(self):
        return self.fragment.mf

    @property
    def mol(self):
        return self.fragment.mol

    @property
    def log(self):
        return self.fragment.log

    @property
    def base(self):
        return self.fragment.base

    @property
    def c_frag(self):
        return self.fragment.c_frag
