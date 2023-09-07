"""Functionality to calculate zeroth moment via numerical integration """

import numpy as np

from vayesta.core.util import dot, einsum
from vayesta.rpa.rirpa.NI_eval import NumericalIntegratorClenCurInfinite, NIException

from vayesta.rpa.rirpa.momzero_NI import diag_sqrt_contrib, diag_sqrt_grad, diag_sqrt_deriv2


class NIError(Exception):
    pass


class NITrRootMP(NumericalIntegratorClenCurInfinite):
    def __init__(self, D, S_L, S_R, npoints, log):
        self.D = D
        self.S_L = S_L
        self.S_R = S_R
        out_shape = (1,)
        diag_shape = (1,)
        super().__init__(out_shape, diag_shape, npoints, log, True)
        self.diagRI = einsum("np,np->p", self.S_L, self.S_R)
        self.diagmat1 = self.D**2 + self.diagRI
        self.diagmat2 = self.D**2

    @property
    def n_aux(self):
        assert self.S_L.shape == self.S_R.shape
        return self.S_L.shape[0]

    def get_F(self, freq):
        return (self.D**2 + freq**2) ** (-1)

    def get_Q(self, freq):
        """Efficiently construct Q = S_R (D^{-1} G) S_L^T
        This is generally the limiting
        """
        S_L = self.S_L * self.get_F(freq)[None]
        return dot(self.S_R, S_L.T)

    @property
    def diagmat1(self):
        return self._diagmat1

    @diagmat1.setter
    def diagmat1(self, val):
        if val is not None and any(val < 0.0):
            raise NIException("Error in numerical integration; diagonal approximation is non-PSD")
        self._diagmat1 = val

    @property
    def diagmat2(self):
        return self._diagmat2

    @diagmat2.setter
    def diagmat2(self, val):
        if val is not None and any(val < 0.0):
            raise NIException("Error in numerical integration; diagonal approximation is non-PSD")
        self._diagmat2 = val

    def eval_diag_contrib(self, freq):
        Dval = diag_sqrt_contrib(self.diagmat1, freq)
        if not (self.diagmat2 is None):
            Dval -= diag_sqrt_contrib(self.diagmat2, freq)
        F = self.get_F(freq)
        HOval = (freq**2) * (F**2)
        HOval = np.multiply(HOval, self.diagRI) / np.pi
        return np.array([sum(Dval - HOval)])

    def eval_diag_deriv_contrib(self, freq):
        Dval = diag_sqrt_grad(self.diagmat1, freq)
        if not (self.diagmat2 is None):
            Dval -= diag_sqrt_grad(self.diagmat2, freq)
        F = self.get_F(freq)
        HOval = (2 * freq * (F**2)) - (4 * (freq**3) * (F**3))
        HOval = np.multiply(HOval, self.diagRI) / np.pi
        return np.array([sum(Dval - HOval)])

    def eval_diag_deriv2_contrib(self, freq):
        Dval = diag_sqrt_deriv2(self.diagmat1, freq)
        if not (self.diagmat2 is None):
            Dval -= diag_sqrt_deriv2(self.diagmat2, freq)
        F = self.get_F(freq)
        HOval = 2 * F**2 - 20 * freq**2 * F**3 + 24 * freq**4 * F**4
        HOval = np.multiply(HOval, self.diagRI) / np.pi
        return np.array([sum(Dval - HOval)])

    def eval_diag_exact(self):
        Dval = self.diagmat1 ** (0.5)
        if not (self.diagmat2 is None):
            Dval -= self.diagmat2 ** (0.5)
        HOval = 0.5 * np.multiply(self.D ** (-1), self.diagRI)
        return np.array([sum(Dval - HOval)])

    def eval_contrib(self, freq):
        Q = self.get_Q(freq)
        F = self.get_F(freq)
        val_aux = np.linalg.inv(np.eye(self.n_aux) + Q) - np.eye(self.n_aux)
        lhs = dot(val_aux, self.S_L * F[None])
        res = np.tensordot(lhs, self.S_R * F[None], ((0, 1), (0, 1)))
        res = (freq**2) * res / np.pi
        return np.array([res])


class NITrRootMP_dRHF(NITrRootMP):
    """All provided quantities are now in spatial orbitals. This actually only requires an additional factor in the
    get_Q method."""

    def get_Q(self, freq):
        # Have equal contributions from both spin channels.
        return 2 * super().get_Q(freq)

    def eval_contrib(self, freq):
        return 2 * super().eval_contrib(freq)
