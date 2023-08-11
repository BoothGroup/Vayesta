"""Functionality to calculate zeroth moment via numerical integration """

import numpy as np

from vayesta.core.util import dot, einsum
from vayesta.rpa.rirpa.NI_eval import (
    NumericalIntegratorClenCurInfinite,
    NumericalIntegratorClenCurSemiInfinite,
    NumericalIntegratorGaussianSemiInfinite,
    NumericalIntegratorBase,
    NIException,
)


class NIMomZero(NumericalIntegratorClenCurInfinite):
    def __init__(self, D, S_L, S_R, target_rot, npoints, log):
        self.D = D
        self.S_L = S_L
        self.S_R = S_R
        self.target_rot = target_rot
        out_shape = self.target_rot.shape
        diag_shape = self.D.shape
        self.diagmat1 = self.diagmat2 = None
        super().__init__(out_shape, diag_shape, npoints, log, True)

    @property
    def n_aux(self):
        assert self.S_L.shape == self.S_R.shape
        return self.S_L.shape[0]

    def get_F(self, freq):
        return (self.D**2 + freq**2) ** (-1)

    def get_Q(self, freq):
        """Efficiently construct Q = S_R F S_L^T
        This is generally the limiting step.
        """
        S_L = self.S_L * self.get_F(freq)[None]
        return dot(self.S_R, S_L.T)

    @property
    def diagmat1(self):
        return self._diagmat1

    @diagmat1.setter
    def diagmat1(self, val):
        if val is not None and any(val < 0.0):
            raise NIException(
                "Error in numerical integration; diagonal approximation is non-PSD"
            )
        self._diagmat1 = val

    @property
    def diagmat2(self):
        return self._diagmat2

    @diagmat2.setter
    def diagmat2(self, val):
        if val is not None and any(val < 0.0):
            raise NIException(
                "Error in numerical integration; diagonal approximation is non-PSD"
            )
        self._diagmat2 = val


class MomzeroDeductNone(NIMomZero):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.diagmat1 = self.D**2 + einsum("np,np->p", self.S_L, self.S_R)

    def eval_diag_contrib(self, freq):
        val = diag_sqrt_contrib(self.diagmat1, freq)
        if not (self.diagmat2 is None):
            val -= diag_sqrt_contrib(self.diagmat2, freq)
        return val

    def eval_diag_deriv_contrib(self, freq):
        val = diag_sqrt_grad(self.diagmat1, freq)
        if not (self.diagmat2 is None):
            val -= diag_sqrt_grad(self.diagmat2, freq)
        return val

    def eval_diag_deriv2_contrib(self, freq):
        val = diag_sqrt_deriv2(self.diagmat1, freq)
        if not (self.diagmat2 is None):
            val -= diag_sqrt_deriv2(self.diagmat2, freq)
        return val

    def eval_diag_exact(self):
        val = self.diagmat1 ** (0.5)
        if not (self.diagmat2 is None):
            val -= self.diagmat2 ** (0.5)
        return val

    def eval_contrib(self, freq):
        if not (self.diagmat2 is None):
            raise ValueError(
                "Diagonal deducted quantity specified without being included in full contribution "
                "evaluation; please update overwrite .eval_contrib() for subclass."
            )
        F = self.get_F(freq)
        Q = self.get_Q(freq)

        rrot = F
        lrot = self.target_rot * rrot[None]
        val_aux = np.linalg.inv(np.eye(self.n_aux) + Q)
        lres = dot(lrot, self.S_L.T)
        res = dot(dot(lres, val_aux), self.S_R * rrot[None])
        return (self.target_rot + (freq**2) * (res - lrot)) / np.pi


class MomzeroDeductD(MomzeroDeductNone):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.diagmat2 = self.D**2

    def eval_contrib(self, freq):
        Q = self.get_Q(freq)
        F = self.get_F(freq)

        rrot = F
        lrot = einsum("lq,q->lq", self.target_rot, F)
        val_aux = np.linalg.inv(np.eye(self.n_aux) + Q)
        res = dot(
            dot(dot(lrot, self.S_L.T), val_aux), einsum("np,p->np", self.S_R, rrot)
        )
        res = (freq**2) * res / np.pi
        return res


class MomzeroDeductHigherOrder(MomzeroDeductD):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.diagRI = einsum("np,np->p", self.S_L, self.S_R)

    # Just calculate diagonals via expression for D-deducted quantity, minus diagonal approximation for higher-order
    # terms.
    def eval_diag_contrib(self, freq):
        Dval = super().eval_diag_contrib(freq)
        F = self.get_F(freq)
        HOval = (freq**2) * (F**2)
        HOval = np.multiply(HOval, self.diagRI) / np.pi
        return Dval - HOval

    def eval_diag_deriv_contrib(self, freq):
        Dval = super().eval_diag_deriv_contrib(freq)
        F = self.get_F(freq)
        HOval = (2 * freq * (F**2)) - (4 * (freq**3) * (F**3))
        HOval = np.multiply(HOval, self.diagRI) / np.pi
        return Dval - HOval

    def eval_diag_deriv2_contrib(self, freq):
        Dval = super().eval_diag_deriv2_contrib(freq)
        F = self.get_F(freq)
        HOval = 2 * F**2 - 20 * freq**2 * F**3 + 24 * freq**4 * F**4
        HOval = np.multiply(HOval, self.diagRI) / np.pi
        return Dval - HOval

    def eval_diag_exact(self):
        Dval = super().eval_diag_exact()
        HOval = 0.5 * np.multiply(self.D ** (-1), self.diagRI)
        return Dval - HOval

    def eval_contrib(self, freq):
        Q = self.get_Q(freq)
        F = self.get_F(freq)

        rrot = F
        lrot = einsum("lq,q->lq", self.target_rot, F)
        val_aux = np.linalg.inv(np.eye(self.n_aux) + Q) - np.eye(self.n_aux)
        res = dot(
            dot(dot(lrot, self.S_L.T), val_aux), self.S_R * rrot[None]
        )
        res = (freq**2) * res / np.pi
        return res


class BaseMomzeroOffset(NumericalIntegratorBase):
    """NB this is an abstract class!"""

    def __init__(self, D, S_L, S_R, target_rot, npoints, log):
        self.D = D
        self.S_L = S_L
        self.S_R = S_R
        self.target_rot = target_rot
        out_shape = self.target_rot.shape
        diag_shape = self.D.shape
        super().__init__(out_shape, diag_shape, npoints, log)
        self.diagRI = einsum("np,np->p", self.S_L, self.S_R)

    def get_offset(self):
        return np.zeros(self.out_shape)

    def eval_contrib(self, freq):
        # This should be real currently, so can safely do this.
        expval = np.exp(-freq * self.D)
        lrot = self.target_rot * expval[None]
        rrot = expval
        res = dot(dot(lrot, self.S_L.T), self.S_R * rrot[None])
        return res

    def eval_diag_contrib(self, freq):
        expval = np.exp(-2 * freq * self.D)
        return np.multiply(expval, self.diagRI)

    def eval_diag_deriv_contrib(self, freq):
        expval = -2 * (np.multiply(self.D, np.exp(-2 * freq * self.D)))
        return np.multiply(expval, self.diagRI)

    def eval_diag_deriv2_contrib(self, freq):
        expval = 4 * (np.multiply(self.D**2, np.exp(-2 * freq * self.D)))
        return np.multiply(expval, self.diagRI)

    def eval_diag_exact(self):
        return 0.5 * np.multiply(self.D ** (-1), self.diagRI)


class MomzeroOffsetCalcGaussLag(
    BaseMomzeroOffset, NumericalIntegratorGaussianSemiInfinite
):
    pass


class MomzeroOffsetCalcCC(BaseMomzeroOffset, NumericalIntegratorClenCurSemiInfinite):
    pass


def diag_sqrt_contrib(D, freq):
    M = (D + freq**2) ** (-1)
    return (np.full_like(D, fill_value=1.0) - (freq**2) * M) / np.pi


def diag_sqrt_grad(D, freq):
    M = (D + freq**2) ** (-1)
    return (2 * ((freq**3) * M**2 - freq * M)) / np.pi


def diag_sqrt_deriv2(D, freq):
    M = (D + freq**2) ** (-1)
    return (-2 * M + 10 * (freq**2) * (M**2) - 8 * (freq**4) * (M**3)) / np.pi


# Subclass for performing calculations with RHF quantities.

class MomzeroDeductHigherOrder_dRHF(MomzeroDeductHigherOrder):
    """All provided quantities are now in spatial orbitals. This actually only requires an additional factor in the
    get_Q method."""
    def get_Q(self, freq):
        # Have equal contributions from both spin channels.
        return 2 * super().get_Q(freq)
