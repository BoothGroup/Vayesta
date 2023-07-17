import numpy as np
import scipy
import scipy.integrate


def hubbard1d_bethe_energy(t, u, interval=(1e-14, 30.75*np.pi), **kwargs):
    """Exact total energy per site for the 1D Hubbard model in the thermodynamic limit.

    from DOI: 10.1103/PhysRevB.77.045133."""

    kwargs['limit'] = kwargs.get('limit', 100)

    def func(x):
        j0 = scipy.special.jv(0, x)
        j1 = scipy.special.jv(1, x)
        eu = np.exp((x*u)/(2*t))
        f = j0*j1/(x*(1 + eu))
        return f

    e, *res = scipy.integrate.quad(func, *interval, **kwargs)
    e = -4*t*e

    return e

def hubbard1d_bethe_docc(t, u, interval=(1e-14, 30.75*np.pi), **kwargs):
    """Exact on-site double occupancy for the 1D Hubbard model in the thermodynamic limit."""

    kwargs['limit'] = kwargs.get('limit', 100)

    def func(x):
        j0 = scipy.special.jv(0, x)
        j1 = scipy.special.jv(1, x)
        eu = np.exp((x*u)/(2*t))
        #f = j0*j1/x * (-eu)/(1+eu)**2 * x/(2*t)
        # Avoid float overflow:
        emu = np.exp(-(x*u)/(2*t))
        f = j0*j1/x * (-1)/(2+emu+eu) * x/(2*t)
        return f

    e, *res = scipy.integrate.quad(func, *interval, **kwargs)
    e = -4*t*e

    return e


def hubbard1d_bethe_docc_numdiff(t, u, du=1e-10, order=2, **kwargs):
    """Exact on-site double occupancy for the 1D Hubbard model in the thermodynamic limit.

    Calculated from the numerical differentiation of the exact Bethe ansatz energy."""

    if order == 1:
        em1 = hubbard1d_bethe_energy(t, u-du, **kwargs)
        ep1 = hubbard1d_bethe_energy(t, u+du, **kwargs)
        docc = (1/2*ep1 - 1/2*em1) / du
    elif order == 2:
        em2 = hubbard1d_bethe_energy(t, u-2*du, **kwargs)
        em1 = hubbard1d_bethe_energy(t, u-  du, **kwargs)
        ep1 = hubbard1d_bethe_energy(t, u+  du, **kwargs)
        ep2 = hubbard1d_bethe_energy(t, u+2*du, **kwargs)
        docc = (1/12*em2 - 2/3*em1 + 2/3*ep1 - 1/12*ep2) / du
    else:
        raise NotImplementedError()

    return docc

if __name__ == '__main__':
    t = 1.0
    for u in range(0, 13):
        e = hubbard1d_bethe_energy(t, u)
        d = hubbard1d_bethe_docc(t, u)
        print("U= %6.3f:  Energy= %.8f  Double occupancy= %.8f" % (u, e, d))


def hubbard1d_bethe_gap(t, u, interval=(1e-14, 30.75*np.pi), **kwargs):
    """Exact total energy per site for the 1D Hubbard model in the thermodynamic limit.

    from DOI: 10.1103/PhysRevB.77.045133."""

    kwargs['limit'] = kwargs.get('limit', 100)

    U = u/t
    def func(x):
        d = sqrt(x**2 - 1)
        n = sinh(2*np.pi*dx/U)
        return 

    eg, *res = scipy.integrate.quad(func, *interval, **kwargs)
    eg = 16*delta**2/U * eg

    return eg