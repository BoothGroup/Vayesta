import numpy as np
import scipy

import vayesta


class ChempotBisection:

    def __init__(self, func, cpt_init=0.0, tol=1e-8, maxiter=30, log=None):
        self.func = func
        self.cpt = cpt_init
        self.converged = False
        # Options
        self.tol = tol
        self.maxiter = maxiter
        self.log = (log or vayesta.log)

    def kernel(self, *args, **kwargs):

        cpt1 = self.cpt
        err1 = self.func(cpt1, *args, **kwargs)
        if abs(err1) <= self.tol:
            self.converged = True
            return cpt1

        # Linear extrapolation for bounds
        cpt2 = cpt1 + np.sign(err1)*1e-3
        err2 = self.func(cpt2, *args, **kwargs)
        m = (err2-err1) / (cpt2-cpt1)
        for powerof2 in range(1, 4):
            cpt3 = cpt1 - (2**powerof2)*err1/m
            err3 = self.func(cpt3, *args, **kwargs)
            if (err1 * err3) <= 0:
                break
            self.log.info('Chemical potential not in [%.8f, %.8f] (errors=[%.3e, %.3e])' % (cpt1, cpt3, err1, err3))
        else:
            raise ValueError('Chemical potential not in [%.8f, %.8f] (errors=[%.3e, %.3e])' % (cpt1, cpt3, err1, err3))
        lower = min(cpt1, cpt3)
        upper = max(cpt1, cpt3)

        it = 1

        def iteration(cpt, *args, **kwargs):
            nonlocal it
            if (cpt == cpt1):
                err = err1
            elif (cpt == cpt3):
                err = err3
            else:
                err = self.func(cpt, *args, **kwargs)
            self.log.info('Chemical potential iteration= %3d  cpt= %+12.8f err= %.3e', it, cpt, err)
            self.cpt = cpt
            it += 1
            if abs(err) <= self.tol:
                raise StopIteration
            return err
        try:
            scipy.optimize.brentq(iteration, lower, upper)
        except StopIteration:
            self.converged = True
        return self.cpt


if __name__ == '__main__':

    def func(cpt):
        return 0.1*cpt + 1

    bisect = ChempotBisection(func)
    cpt = bisect.kernel()
