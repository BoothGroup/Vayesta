"""Functionality to calculate zeroth moment via numerical integration """



from vayesta.rpa.rirpa.NI_eval import NumericalIntegratorClenCur
from vayesta.rpa.rirpa import momzero_calculation
from vayesta.core.util import *

import numpy as np

class NIMomZero(NumericalIntegratorClenCur):
    def __init__(self,  D, S_L, S_R, target_rot, npoints):
        self.D = D
        self.S_L = S_L
        self.S_R = S_R
        self.target_rot = target_rot
        out_shape = self.target_rot.shape
        diag_shape = self.D.shape
        super().__init__(out_shape, diag_shape, npoints)

    @property
    def n_aux(self):
        assert(self.S_L.shape == self.S_R.shape)
        return self.S_L.shape[0]

    def get_G(self, freq):
        return construct_G(freq, self.D)

    def get_Q(self, freq):
        return construct_Q(freq, self.D, self.S_L, self.S_R)

class MomzeroDeductNone(NIMomZero):

    @property
    def diagmat1(self):
        return self.D + einsum("np,np->p", self.S_L, self.S_R)
    @property
    def diagmat2(self):
        return None

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
            raise ValueError("Diagonal deducted quantity specified without being included in full contribution "
                             "evaluation; please update overwrite .eval_contrib() for subclass.")
        D = self.D
        G = self.get_G(freq)
        Q = self.get_Q(freq)

        rrot = np.multiply(G, D**(-1))
        lrot = einsum("lq,q->lq", self.target_rot, rrot)
        val_aux = np.linalg.inv(np.eye(self.n_aux) + Q)
        lres = np.dot(lrot, self.S_L.T)
        res = dot(dot(lres, val_aux), einsum("np,p->np", self.S_R, rrot))
        return (self.target_rot + (freq ** 2) * (res - lrot)) / np.pi

class MomzeroDeductD(MomzeroDeductNone):

    @property
    def diagmat2(self):
        return self.D

    def eval_contrib(self, freq):
        D = self.D
        G = self.get_G(freq)
        Q = self.get_Q(freq)

        rrot = np.multiply(G, D**(-1))
        lrot = einsum("lq,q->lq", self.target_rot, rrot)
        val_aux = np.linalg.inv(np.eye(self.n_aux) + Q)
        lrot = dot(lrot, self.S_L.T)
        res = dot(dot(lrot, val_aux), einsum("np,p->np", self.S_R, rrot))
        res = (freq ** 2) * res / np.pi
#        diff = abs(res - momzero_calculation.eval_eta0_contrib_diff2(freq, self.S_L, self.S_R, self.D, self.target_rot)).max()
        return res


def construct_G(freq, D):
    """Evaluate G = D (D**2 + \omega**2 I)**(-1), given frequency and diagonal of D."""
    return np.multiply(D, (D ** 2 + freq ** 2) ** (-1))

def construct_Q(freq, D, S_L, S_R):
    """Efficiently construct Q = S_R (D^{-1} G) S_L^T
    This is generally the limiting
    """
    intermed = np.multiply(construct_G(freq, D), D ** (-1))
    S_L = einsum("np,p->np", S_L, intermed)
    return dot(S_R,S_L.T)

def diag_sqrt_contrib(D, freq):
    M = (D + freq ** 2) ** (-1)
    return (np.full_like(D, fill_value=1.0) - (freq ** 2) * M) / np.pi

def diag_sqrt_grad(D, freq):
    M = (D + freq ** 2) ** (-1)
    return (2 * ((freq ** 3) * M**2 - freq * M)) / np.pi

def diag_sqrt_deriv2(D, freq):
    M = (D + freq ** 2) ** (-1)
    return (- 2 * M + 10 * (freq ** 2) * (M ** 2) - 8 * (freq ** 4) * (M**3)) / np.pi
