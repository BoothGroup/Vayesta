import numpy as np
import scipy.integrate
import scipy.optimize
import scipy.special


class NIException(BaseException):
    pass


class NumericalIntegratorBase:
    """Abstract base class for numerical integration over semi-infinite and infinite limits.
    Subclasses implementing a specific quadrature need to define

    Subclasses implementing specific evaluations need to define:
        .eval_contrib
        .eval_diag_contrib
        .eval_diag_deriv_contrib
        .eval_diag_deriv2_contrib
        .eval_diag_exact
    A new .__init__ assigning any required attributes and a .fix_params may also be required, depending upon the
    particular form of the integral to be approximated.
    Might be able to write this as a factory class, but this'll do for now.
    """

    def __init__(self, out_shape, diag_shape, npoints, log):
        self.log = log
        self.out_shape = out_shape
        self.diag_shape = diag_shape
        self.npoints = npoints

    @property
    def npoints(self):
        return self._npoints

    @npoints.setter
    def npoints(self, value):
        self._npoints = value

    def get_quad(self, a):
        """Generate the appropriate Clenshaw-Curtis quadrature points and weights."""
        return NotImplementedError

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
        for point, weight in zip(*quadrature):
            contrib = evaluator(point)
            assert (contrib.shape == res_shape)
            integral += weight * contrib

        return integral

    def _NI_eval_w_error(self, *args):
        raise NotImplementedError(
            "Error estimation only available with naturally nested quadratures (current just Clenshaw-Curtis).")

    def _NI_eval_deriv(self, a, res_shape, evaluator):
        """Base function to perform numerical integration with provided quadrature grid."""
        quadrature = self.get_quad(a)
        integral = np.zeros(res_shape)
        for point, weight in zip(*quadrature):
            contrib = evaluator(point, weight, a)
            assert (contrib.shape == res_shape)
            integral += contrib
        return integral

    def eval_NI_approx(self, a):
        """Evaluate the NI approximation of the integral with a provided quadrature."""
        return self._NI_eval(a, self.out_shape, self.eval_contrib), None

    def eval_diag_NI_approx(self, a):
        """Evaluate the NI approximation to the diagonal approximation of the integral."""
        return self._NI_eval(a, self.diag_shape, self.eval_diag_contrib)

    def eval_diag_NI_approx_grad(self, a):
        """Evaluate the gradient w.r.t a of NI diagonal expression.
        Note that for all quadratures the weights and quadrature point positions are proportional to the arbitrary
        parameter `a', so we can use the same expressions for the derivatives."""

        def get_grad_contrib(freq, weight, a):
            contrib = self.eval_diag_contrib(freq)
            deriv = self.eval_diag_deriv_contrib(freq)
            return (weight / a) * contrib + weight * (freq / a) * deriv

        return self._NI_eval_deriv(a, self.diag_shape, get_grad_contrib)

    def eval_diag_NI_approx_deriv2(self, a):
        """Evaluate the second derivative w.r.t a of NI diagonal expression.
        Note that for all quadratures the weights and quadrature point positions are proportional to the arbitrary
        parameter `a', so we can use the same expressions for the derivatives."""

        def get_deriv2_contrib(freq, weight, a):
            deriv = self.eval_diag_deriv_contrib(freq)
            deriv2 = self.eval_diag_deriv2_contrib(freq)
            return 2 * (weight / a) * (freq / a) * deriv + weight * (freq / a) ** 2 * deriv2

        return self._NI_eval_deriv(a, self.diag_shape, get_deriv2_contrib)

    def test_diag_derivs(self, a, delta=1e-6):
        freq = np.random.rand() * a
        self.log.info("Testing gradients w.r.t variation of omega at random frequency point=%8.6e:", freq)
        grad_1 = self.eval_diag_deriv_contrib(freq)
        deriv2_1 = self.eval_diag_deriv2_contrib(freq)
        grad_2 = (self.eval_diag_contrib(freq + delta / 2) - self.eval_diag_contrib(freq - delta / 2)) / delta
        deriv2_2 = (self.eval_diag_deriv_contrib(freq + delta / 2) - self.eval_diag_deriv_contrib(
            freq - delta / 2)) / delta
        self.log.info("Max Grad Error=%6.4e", abs(grad_1 - grad_2).max())
        self.log.info("Max Deriv2 Error=%6.4e", abs(deriv2_1 - deriv2_2).max())

        self.log.info("Testing ensemble gradients w.r.t variation of a:")
        grad_1 = self.eval_diag_NI_approx_grad(a)
        deriv2_1 = self.eval_diag_NI_approx_deriv2(a)
        grad_2 = (self.eval_diag_NI_approx(a + delta / 2) - self.eval_diag_NI_approx(a - delta / 2)) / delta
        deriv2_2 = (self.eval_diag_NI_approx_grad(a + delta / 2) - self.eval_diag_NI_approx_grad(a - delta / 2)) / delta
        self.log.info("Max Grad Error=%6.4e", abs(grad_1 - grad_2).max())
        self.log.info("Max Deriv2 Error=%6.4e", abs(deriv2_1 - deriv2_2).max())

    def opt_quadrature_diag(self, ainit=None):
        """Optimise the quadrature to exactly integrate a diagonal approximation to the integral"""

        def get_val(a):
            val = (self.eval_diag_NI_approx(a) - self.eval_diag_exact()).sum()
            return val

        def get_grad(a):
            return self.eval_diag_NI_approx_grad(a).sum()

        def get_deriv2(a):
            return self.eval_diag_NI_approx_deriv2(a).sum()

        def find_good_start(ainit=1e-6, scale_fac=10.0, maxval=1e8, relevance_factor=5):
            """Using a quick exponential search, find the lowest value of the penalty function and from this obtain
            good guesses for the optimum and a good bound on either side.
            Note that the size of resulting bracket will be proportional to both the optimal value and the scaling
            factor."""
            max_exp = int(np.log(maxval / ainit) / np.log(scale_fac))
            vals = np.array([ainit * scale_fac ** x for x in range(max_exp)])
            fvals = np.array([abs(get_val(x)) for x in vals])
            optarg = fvals.argmin()
            optval = fvals[optarg]
            # Now find the values which are within reach of lowest value
            relevant = np.where(fvals < relevance_factor * optval)[0]

            minarg = min(relevant[0], optarg - 1)
            maxarg = min(relevant[-1], optarg + 1)
            return [ainit * scale_fac ** x for x in (optarg, minarg, maxarg)]

        solve = 1
        ainit, mini, maxi = find_good_start()
        try:
            solve, res = scipy.optimize.newton(get_val, x0=ainit, fprime=get_grad, tol=1e-8, maxiter=30,
                                               fprime2=get_deriv2, full_output=True)
        except (RuntimeError, NIException):
            opt_min = True
        else:
            # Did we find a root?
            opt_min = not res.converged
        if opt_min:
            res = scipy.optimize.minimize_scalar(lambda freq: abs(get_val(freq)),
                                                 bounds=(mini, maxi), method="bounded")
            if not res.success:
                raise NIException("Could not optimise `a' value.")
            solve = res.x
            self.log.info(
                "Used minimisation to optimise quadrature grid: a= %.2e  penalty value= %.2e "
                "(smaller is better)", solve, res.fun)
        return solve

    def fix_params(self):
        """If required set parameters within ansatz; defined to ensure hook for functionality in future, will
        not always be needed."""
        pass

    def get_offset(self):
        return np.zeros(self.out_shape)

    def kernel(self, a=None, opt_quad=True):
        """Perform numerical integration. Put simply, fix any arbitrary parameters in the integral to be evaluated,
        optimise the quadrature grid to ensure a diagonal approximation is exactly integrated then evaluate full
        expression."""
        self.fix_params()
        if opt_quad:
            a = self.opt_quadrature_diag(a)
        else:
            if a is None:
                raise ValueError(
                    "A value for the quadrature scaling parameter a must be provided if optimisation is not"
                    "permitted.")
        integral, errors = self.eval_NI_approx(a)
        return integral + self.get_offset(), errors

    def kernel_adaptive(self):
        self.fix_params()
        integral, err, info = scipy.integrate.quad_vec(self.eval_contrib, a=0.0, b=np.inf, norm="max",
                                                       epsabs=1e-4, epsrel=1e-200, full_output=True)
        if not info.success:
            raise NIException("Adaptive gaussian quadrature could not compute integral.")
        else:
            self.log.info(
                "Successfully computed integral via adaptive quadrature using %d evaluations with estimated error of %6.4e",
                info.neval, err)
        return integral + self.get_offset(), err

    def l2_scan(self, freqs):
        return [np.linalg.norm(self.eval_contrib(x)) for x in freqs]

    def max_scan(self, freqs):
        return [abs(self.eval_contrib(x)).max() for x in freqs]

    def get_quad_vals(self, a, l2norm=True):
        quadrature = self.get_quad(a)
        getnorm = np.linalg.norm if l2norm else lambda x: abs(x).max()
        points = [x[0] for x in quadrature]
        vals = [getnorm(self.eval_contrib(p)) for p in points]
        return points, vals


class NumericalIntegratorClenCur(NumericalIntegratorBase):

    @property
    def npoints(self):
        return self._npoints

    @npoints.setter
    def npoints(self, value):
        if value % 4 != 0:
            value += 4 - value % 4
            self.log.warning("Npoints increased to next multiple of 4 (%d) to allow error estimation.", value)
        self._npoints = value

    def _NI_eval_w_error(self, a, res_shape, evaluator):
        """Base function to perform numerical integration with provided quadrature grid.
        Since Clenshaw-Curtis quadrature is naturally nested, we can generate an error estimate straightforwardly."""
        quadrature = self.get_quad(a)
        integral = np.zeros(res_shape)
        integral_half = np.zeros(res_shape)
        integral_quarter = np.zeros(res_shape)

        for i, (point, weight) in enumerate(zip(*quadrature)):
            contrib = evaluator(point)
            assert (contrib.shape == res_shape)
            integral += weight * contrib
            if i % 2 == 0:
                integral_half += 2 * weight * contrib
                if i % 4 == 0:
                    integral_quarter += 4 * weight * contrib

        a = scipy.linalg.norm(integral_quarter - integral)
        b = scipy.linalg.norm(integral_half - integral)
        # Using a simple approximation gives these expressions; we instead use a more complicated cubic expression in
        # calculate_error
        # error = b ** 3 / a ** 2
        # error_error = b ** 2 / (a ** 2 + b ** 2)
        error = self.calculate_error(a, b)
        self.log.info("Numerical Integration performed with estimated L2 norm error %6.4e.",
                      error)
        return integral, error

    def calculate_error(self, a, b):
        """Calculate error by solving cubic equation to model convergence as \alpha e^{-\beta n_p}.
        This relies upon the Cauchy-Schwartz inequality, and assumes all errors are at their maximum values, so
        generally overestimates the resulting error, which suits us well.
        This also overestimates the error since it doesn't account for the effect of quadrature grid optimisation, which
        leads to our actual estimates converging more rapidly than they would with a static grid spacing parameter.
        """
        if a - b < 1e-10:
            self.log.info("RIRPA error numerically zero.")
            return 0.0

        roots = np.roots([1, 0, a / (a - b), - b / (a - b)])

        # Need to choose root with no imaginary part and real part between zero and one; if there are multiple (if this
        # is even possible) take the largest.
        wanted_root = roots[(abs(roots.imag) < 1e-10) & (roots.real <= 1.0) & (roots.real >= 0)].real[-1]
        exp_beta_n = wanted_root ** 4
        # alpha = a * (exp_beta_n + exp_beta_n**(1/4))**(-1)
        error = a * (1 + exp_beta_n ** (-3 / 4)) ** (-1)
        return error

    def eval_NI_approx(self, a):
        """Evaluate the NI approximation of the integral with a provided quadrature."""
        return self._NI_eval_w_error(a, self.out_shape, self.eval_contrib)


class NumericalIntegratorClenCurInfinite(NumericalIntegratorClenCur):
    def __init__(self, out_shape, diag_shape, npoints, log, even):
        super().__init__(out_shape, diag_shape, npoints, log)
        self.even = even

    def get_quad(self, a):
        # Don't care about negative values, since grid should be symmetric about x=0.
        return gen_ClenCur_quad_inf(a, self.npoints, self.even)


class NumericalIntegratorClenCurSemiInfinite(NumericalIntegratorClenCur):
    def __init__(self, out_shape, diag_shape, npoints, log):
        super().__init__(out_shape, diag_shape, npoints, log)

    def get_quad(self, a):
        if a < 0:
            raise NIException("Negative quadrature scaling factor not permitted.")
        return gen_ClenCur_quad_semiinf(a, self.npoints)


class NumericalIntegratorGaussianSemiInfinite(NumericalIntegratorBase):
    def __init__(self, out_shape, diag_shape, npoints, log):
        super().__init__(out_shape, diag_shape, npoints, log)

    @property
    def npoints(self):
        return len(self._points)

    @npoints.setter
    def npoints(self, value):
        """For Gaussian quadrature recalculating the points and weights every time won't be performant;
        instead lets cache them each time npoints is changed."""
        if value > 100:
            self.log.warning("Gauss-Laguerre quadrature with degree over 100 may be problematic due to numerical "
                             "ill-conditioning in the quadrature construction. Watch out for floating-point overflows!")
        self._points, self._weights = np.polynomial.laguerre.laggauss(value)
        self._weights = np.array([w * np.exp(p) for (p, w) in zip(self._points, self._weights)])

    def get_quad(self, a):
        if a < 0:
            raise NIException("Negative quadrature scaling factor not permitted.")
        return a * self._points, a * self._weights


def gen_ClenCur_quad_inf(a, npoints, even=False):
    """Generate quadrature points and weights for Clenshaw-Curtis quadrature over infinite range (-inf to +inf)"""
    symfac = 1.0 + even
    # If even we only want points up to t <= pi/2
    tvals = [(j / npoints) * (np.pi / symfac) for j in range(1, npoints + 1)]

    points = [a / np.tan(t) for t in tvals]
    weights = [a * np.pi * symfac / (2 * npoints * (np.sin(t) ** 2)) for t in tvals]
    if even: weights[-1] /= 2
    return points, weights


def gen_ClenCur_quad_semiinf(a, npoints):
    """Generate quadrature points and weights for Clenshaw-Curtis quadrature over semiinfinite range (0 to +inf)"""
    tvals = [(np.pi * j / (npoints + 1)) for j in range(1, npoints + 1)]
    points = [a / (np.tan(t / 2) ** 2) for t in tvals]
    jsums = [sum([np.sin(j * t) * (1 - np.cos(j * np.pi)) / j for j in range(1, npoints + 1)]) for t in tvals]
    weights = [a * (4 * np.sin(t) / ((npoints + 1) * (1 - np.cos(t)) ** 2)) * s for (t, s) in zip(tvals, jsums)]
    return points, weights


class NICheckInf(NumericalIntegratorClenCurInfinite):
    def __init__(self, exponent, npoints):
        super().__init__((), (), npoints, even=True)
        self.exponent = exponent

    def eval_contrib(self, freq):
        # return np.array(np.exp(-freq*self.exponent))
        return np.array((freq + 0.1) ** (-self.exponent))
