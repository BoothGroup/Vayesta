import numpy as np
import scipy.optimize

class NumericalIntegratorClenCur:
    """Abstract base class for numerical integration of even functions from -infty to +infty; subclasses
    need to define:
        .eval_contrib
        .eval_diag_contrib
        .eval_diag_deriv_contrib
        .eval_diag_deriv2_contrib
        .eval_diag_exact
    A new .__init__ assigning any required attributes and a .fix_params may also be required, depending upon the
    particular form of the integral to be approximated.
    Might be able to write this as a factory class, but this'll do for now.
    """
    def __init__(self, out_shape, diag_shape, npoints):
        self.out_shape = out_shape
        self.diag_shape = diag_shape
        self.npoints = npoints

    def get_quad(self, a):
        """Generate the Clenshaw-Curtis quadrature with a cot distribution function."""
        return gen_ClenCur_quad(a, self.npoints, even=True)

    def eval_contrib(self, freq):
        """Evaluate contribution to numerical integral of result at given frequency point."""
        raise NotImplementedError

    def eval_diag_contrib(self, freq):
        """Evaluate contribution to integral of diagonal approximation at given frequency point."""
        raise NotImplementedError

    def eval_diag_deriv_contrib(self, freq):
        """Evaluate gradient of contribution to integral of diagonal approximation at given frequency point,
        w.r.t that frequency point."""
        raise NotImplementedError

    def eval_diag_deriv2_contrib(self, freq):
        """Evaluate second derivative of contribution to integral of diagonal approximation at given frequency point,
        w.r.t that frequency point."""
        raise NotImplementedError

    def eval_diag_exact(self):
        """Provides an exact evaluation of the integral for the diagonal approximation."""
        raise NotImplementedError

    def _NI_eval(self, a, res_shape, evaluator):
        """Base function to perform numerical integration with provided quadrature grid."""
        quadrature = self.get_quad(a)
        integral = np.zeros(res_shape)
        for point, weight in quadrature:
            contrib = evaluator(point)
            assert(contrib.shape == res_shape)
            integral += weight * contrib
        return integral

    def _NI_eval_deriv(self, a, res_shape, evaluator):
        """Base function to perform numerical integration with provided quadrature grid."""
        quadrature = self.get_quad(a)
        integral = np.zeros(res_shape)
        for point, weight in quadrature:
            contrib = evaluator(point, weight, a)
            assert(contrib.shape == res_shape)
            integral += contrib
        return integral

    def eval_NI_approx(self, a):
        """Evaluate the NI approximation of the integral with a provided quadrature."""
        return self._NI_eval(a, self.out_shape, self.eval_contrib)

    def eval_diag_NI_approx(self, a):
        """Evaluate the NI approximation to the diagonal approximation of the integral."""
        return self._NI_eval(a, self.diag_shape, self.eval_diag_contrib)

    def eval_diag_NI_approx_grad(self, a):
        """Evaluate the gradient w.r.t a of NI diagonal expression."""
        def get_grad_contrib(freq, weight, a):
            contrib = self.eval_diag_contrib(freq)
            deriv = self.eval_diag_deriv_contrib(freq)
            return (weight/a) * contrib + weight * (freq / a) * deriv
        return self._NI_eval_deriv(a, self.diag_shape, get_grad_contrib)

    def eval_diag_NI_approx_deriv2(self, a):
        """Evaluate the second derivative w.r.t a of NI diagonal expression."""
        def get_deriv2_contrib(freq, weight, a):
            deriv = self.eval_diag_deriv_contrib(freq)
            deriv2 = self.eval_diag_deriv2_contrib(freq)
            return 2 * (weight/a) * (freq / a) * deriv + weight * (freq / a)**2 * deriv2
        return self._NI_eval_deriv(a, self.diag_shape, get_deriv2_contrib)

    def test_diag_derivs(self, a, delta=1e-6):
        grad_1 = self.eval_diag_NI_approx_grad(a)
        deriv2_1 = self.eval_diag_NI_approx_deriv2(a)
        grad_2 = (self.eval_diag_NI_approx(a+delta/2) - self.eval_diag_NI_approx(a-delta/2))/delta
        deriv2_2 = (self.eval_diag_NI_approx_grad(a + delta / 2) - self.eval_diag_NI_approx_grad(a - delta / 2)) / delta
        print("Max Grad Error={:6.4e}".format(abs(grad_1 - grad_2).max()))
        print("Max Deriv2 Error={:6.4e}".format(abs(deriv2_1 - deriv2_2).max()))

    def opt_quadrature_diag(self, ainit=1.0):
        """Optimise the quadrature to exactly integrate a diagonal approximation to the integral"""
        def get_val(a):
            return (self.eval_diag_NI_approx(a) - self.eval_diag_exact()).sum()
        def get_grad(a):
            return self.eval_diag_NI_approx_grad(a).sum()
        def get_deriv2(a):
            return self.eval_diag_NI_approx_deriv2(a).sum()
        solve = scipy.optimize.newton(get_val, x0=ainit, fprime=get_grad, fprime2=get_deriv2)
        print("!", solve)
        return solve

    def fix_params(self):
        """If required set parameters within ansatz; defined to ensure hook for functionality in future, will
        not always be needed."""
        pass

    def kernel(self, a = 1.0, opt_quad = True):
        """Perform numerical integration. Put simply, fix any arbitrary parameters in the integral to be evaluated,
        optimise the quadrature grid to ensure a diagonal approximation is exactly integrated then evaluate full
        expression."""
        self.fix_params()
        if opt_quad:
            a = self.opt_quadrature_diag(a)
        return self.eval_NI_approx(a)

def gen_ClenCur_quad(a, npoints, even = False):
    symfac = 1.0 + even
    # If even we only want points up to t <= pi/2
    tvals = [(j/npoints) * (np.pi / symfac ) for j in range(1, npoints+1)]

    points = [a/np.tan(t) for t in tvals]
    weights = [a * np.pi * symfac / (2 * npoints * (np.sin(t)**2)) for t in tvals]
    if even: weights[-1] /= 2
    return points, weights


