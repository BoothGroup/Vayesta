"""Functionality to calculate zeroth moment via numerical integration """
import scipy.optimize

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

    def get_F(self, freq):
        return (self.D ** 2 + freq ** 2) ** (-1)

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
        F = self.get_F(freq)
        Q = self.get_Q(freq)

        rrot = F
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
        Q = self.get_Q(freq)
        F = self.get_F(freq)

        rrot = F
        lrot = einsum("lq,q->lq", self.target_rot, F)
        val_aux = np.linalg.inv(np.eye(self.n_aux) + Q)
        res = dot(dot(dot(lrot, self.S_L.T), val_aux), einsum("np,p->np", self.S_R, rrot))
        res = (freq ** 2) * res / np.pi
#        diff = abs(res - momzero_calculation.eval_eta0_contrib_diff2(freq, self.S_L, self.S_R, self.D, self.target_rot)).max()
        return res

class MomzeroDeductHigherOrder(MomzeroDeductNone):

    def eval_contrib(self, freq):
        Q = self.get_Q(freq)
        F = self.get_F(freq)

        rrot = F
        lrot = einsum("lq,q->lq", self.target_rot, F)
        val_aux = np.linalg.inv(np.eye(self.n_aux) + Q) - np.eye(self.n_aux)
        res = dot(dot(dot(lrot, self.S_L.T), val_aux), einsum("np,p->np", self.S_R, rrot))
        res = (freq ** 2) * res / np.pi
        return res

class MomzeroOffsetCalc(MomzeroDeductNone):

    def __init__(self, *args):
        super().__init__(*args)
        self.alpha = None
        self.diagRI = einsum("np,np->p", self.S_L, self.S_R)

    def fix_params(self):

        try:
            self.alpha = self.opt_alpha()
        except ConvergenceError:
            self.alpha = self.D.mean()
            print(
                "Failed to find stationary point for alpha optimisation; falling back to heuristic value {:6.4e}.".format(
                    self.alpha))

    def opt_alpha(self):
        def get_penalty(alpha):
            intermed = (2*self.D)**(-1) - \
                       2 * (self.D + alpha * np.full_like(self.D, fill_value=1.0))**(-1) + \
                       0.5 * (alpha)**(-1) * np.full_like(self.D, fill_value=1.0)
            return sum(np.multiply(intermed, self.diagRI))
        def get_grad(alpha):
            intermed = +2 * (self.D + alpha * np.full_like(self.D, fill_value=1.0))**(-2) - \
                       0.5 * (alpha)**(-2) * np.full_like(self.D, fill_value=1.0)
            return sum(np.multiply(intermed, self.diagRI))
        def get_deriv2(alpha):
            intermed = -4 * (self.D + alpha * np.full_like(self.D, fill_value=1.0))**(-3) + \
                       1.0 * (
                           alpha)**(-3) * np.full_like(self.D, fill_value=1.0)
            return sum(np.multiply(intermed, self.diagRI))
        #delta = 1e-6
        #print(self.alpha,get_penalty(self.alpha),get_penalty(self.alpha/2), get_penalty(self.alpha*2))
        #print((get_penalty(self.alpha + delta) - get_penalty(self.alpha - delta)) / (2 * delta), get_grad(self.alpha))
        #print((get_grad(self.alpha + delta) - get_grad(self.alpha - delta)) / (2 * delta), get_deriv2(self.alpha))

        #print("Test:")
        #print(get_penalty(self.D.min()), get_penalty(self.D.max()))
        #print(get_grad(self.D.min()), get_grad(self.D.max()))
        #print(get_penalty(1e6))

        # Want to minimise the trace of the integral to be evaluated; our penalty function here computes this value.
        # Seek minimum for alpha>0; can show that gradient will have opposite sign at minimum and maximum values of D.
        trRI = self.diagRI.sum()
        baretr = np.multiply(self.diagRI, 0.5 * self.D**(-1)).sum()
        lo, hi = self.D.min(), self.D.max()
        gradlo, gradhi = get_grad(lo), get_grad(hi)
        i=0
        while gradlo * trRI > 0:
            print("Shifted lo for grad!")
            print(lo,gradlo, trRI)
            lo = lo / 10
            gradlo = get_grad(lo)
            i += 1
            if i > 5:
                raise ConvergenceError
        i=0
        while gradhi * trRI < 0:
            print("Shifted hi for grad!")
            print(hi,gradhi, trRI)
            hi *= 10
            gradhi = get_grad(hi)
            i += 1
            if i > 5:
                raise ConvergenceError

        root, res = scipy.optimize.brentq(get_grad, lo, hi, full_output=True)

        if not res.converged:
            raise ConvergenceError
        # Check if our stationary point trace value has a different sign to the original one; if so then we need to
        # find the root where integrand becomes traceless.
        penroot = get_penalty(root)

        if penroot * baretr > 0:
            print("Alpha set to {:6.4e} to minimise integral trace from {:6.4e} to {:6.4e}.".format(
                root, baretr, penroot))
            return root
        else:
            # Sign change in trace; can find a root!
            # There will in fact be two roots, one on either side of the stationary point, as the penalty function is
            # positive at both 0 and +infinity. For now we choose the lower value, corresponding to a larger offset,
            # since the penalty rises steeply for alpha->0 but slowly for alpha->infinity.
            penlo = get_penalty(lo)
            while penlo * baretr < 0:
                print("Shifted lo for pen!")
                print(lo, penlo, baretr)
                lo = lo / 10
                penlo = get_penalty(lo)

            root2, res = scipy.optimize.brentq(get_penalty, lo, root, full_output=True)
            if not res.converged:
                raise ConvergenceError
            print("Alpha set to {:6.4e} to make integral traceless.".format(root2))
            return root2

    def get_offset(self):
        res = dot(self.target_rot, self.S_L.T, einsum("np,p->np", self.S_R, (self.D+self.alpha)**(-1)))
        res += dot(self.target_rot, einsum("np,p->pn", self.S_L, (self.D+self.alpha)**(-1)), self.S_R)
        res -= self.alpha**(-1) * dot(dot(self.target_rot, self.S_L.T), self.S_R) / 2
        return res

    def eval_contrib(self, freq):
        expval = np.exp(-freq * self.D) - np.full_like(self.D, fill_value=np.exp(- freq * self.alpha))
        lrot = einsum("lp,p->lp", self.target_rot, expval)
        rrot = expval
        res = dot(dot(lrot, self.S_L.T), einsum("np,p->np", self.S_R, rrot))
        return res


def construct_F(freq, D):
    return (D ** 2 + freq ** 2) ** (-1)

def construct_G(freq, D):
    """Evaluate G = D (D**2 + \omega**2 I)**(-1), given frequency and diagonal of D."""
    return np.multiply(D, construct_F(freq, D))

def construct_Q(freq, D, S_L, S_R):
    """Efficiently construct Q = S_R (D^{-1} G) S_L^T
    This is generally the limiting
    """
    S_L = einsum("np,p->np", S_L, construct_F(freq, D))
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
