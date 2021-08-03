import numpy as np
import scipy
import scipy.integrate


def hubbard1d_bethe_energy(t, u, interval=(1e-14, 100), **kwargs):
    """Exact total energy per site for the 1D Hubbard model.

    from DOI: 10.1103/PhysRevB.77.045133."""

    def func(x):
        j0 = scipy.special.jv(0, x)
        j1 = scipy.special.jv(1, x)
        f = j0*j1/(x*(1 + np.exp((x*u)/(2*t))))
        return f

    e, *res = scipy.integrate.quad(func, *interval, **kwargs)
    e = -4*t*e

    return e


def hubbard1d_bethe_double_occ(t, u, du=1e-10, order=2, **kwargs):
    """Exact on-site double occupancy for the 1D Hubbard model.

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
    u = 10.0

    e = hubbard1d_bethe_energy(t, u)
    d = hubbard1d_bethe_double_occ(t, u)
    print("t= %.3f U= %.3f : E(Bethe)= %.8f D(Bethe)= %.8f" % (t, u, e, d))
