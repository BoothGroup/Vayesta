#!/usr/bin/env python3
"""Fitting of equations of states."""

import dataclasses
import numpy as np
import scipy
import scipy.optimize

HARTREE2JOULE = 4.359744e-18
ANGSTROM2METER = 1e-10
# Hartree per cubic Angstrom to Giga-Pascal
HPCA2GPA = (HARTREE2JOULE/ANGSTROM2METER**3) / 1e9


def parabola(v, e0, v0, b0):
    """Quadratic equation of state."""
    ev = 0.5*b0/v0*(v-v0)**2 + e0
    return ev


def birch_murnaghan(v, e0, v0, b0, bp):
    """Third-order Birch-Murnaghan equation of states."""
    vr = (v0/v)**(2.0/3)
    t1 = (vr-1)**3
    t2 = (vr-1)**2 * (6-4*vr)
    ev = e0 + 9/16*v0*b0 * (t1*bp + t2)
    return ev


@dataclasses.dataclass
class FitResult:
    """DOC"""
    e0: float = None
    b0: float = None
    x0: float = None
    v0: float = None
    bp: float = None

    def __repr__(self):
        txt = "Fit results: E0= %.4f Ha" % self.e0
        if self.x0 is not None:
            txt += ("  x0= %.4f A" % self.x0)
        if self.v0 is not None:
            txt += ("  V0= %.4f A^3" % self.v0)
        if self.b0 is not None:
            txt += ("  B0= %.4f GPa" % self.b0)
        if self.bp is not None:
            txt += ("  B'= %.4f" % self.bp)
        return txt


def fit_eos(volumes, energies, fitfunc=birch_murnaghan, plot=True, value_at=None):
    """Fit EOS to volumes, energies data points."""
    # Estimate starting parameters
    minidx = np.argmin(energies)
    e0 = energies[minidx]
    v0 = volumes[minidx]
    b0 = bp = 0.0
    if fitfunc == parabola:
        p0 = (e0, v0, b0)
    else:
        p0 = (e0, v0, b0, bp)

    popt, pcov = scipy.optimize.curve_fit(fitfunc, volumes, energies, p0=p0)
    e0, v0, b0 = popt[:3]
    other = popt[3:]

    # Convert to GPa
    b0 = b0 * HPCA2GPA

    if plot:
        try:
            from matplotlib import pyplot as plt
            grid = np.linspace(volumes[0], volumes[-1], 100)
            y = fitfunc(grid, *popt)
            ax = plt.subplot(111)
            plt.subplots_adjust(left=0.16, bottom=0.12, right=0.98, top=0.98)
            ax.plot(grid, y, label="Fit")
            ax.plot(volumes, energies, label="Data points", marker=".", ls="", markersize=10, markeredgecolor='black')
            ax.plot([v0], [e0], label="Minimum", marker='p', color='C1', markersize=12, markeredgecolor='black', ls='')
            ax.set_xlabel('Unit cell volume ($\mathrm{\AA}^3$)')
            ax.set_ylabel('Unit cell energy ($E_\mathrm{H}$)')
            text = """
            $E_0 = % .6f\,\mathrm{Ha}$
            $V_0 = % .6f\,\mathrm{\AA}^3$
            $B_0 = % .6f\,\mathrm{GPa}$
            """ % (e0, v0, b0)
            if other:
                bp = other[0]
                text += "$B'= % .6f$\n" % bp
            if value_at is not None:
                xv = value_at
                yv = fitfunc(xv, *popt)
                text += "$E(%g\,\mathrm{\AA}^3)= %.6f\,\mathrm{Ha}$" % (xv, yv)
                ax.plot([xv], [yv], label="Value at $V= %g\,\mathrm{\AA}^3$" % xv, marker='d', color='C3', markersize=12, markeredgecolor='black', ls='')

            ax.text(0.3, 0.7,  text, transform=ax.transAxes, ha='left', va='top')
            ax.legend()
            if plot is True:
                plt.show()
            else:
                plt.savefig(plot)
        except Exception as e:
            print("Error while creating plot: %s" % e)

    result = FitResult(e0, b0, v0=v0)
    if other:
        result.bp = other[0]

    return result


def fit_from_file(filename, xcol=0, ycol=1, volume_func=None):
    """Load and fit EOS to textfile."""
    data = np.loadtxt(filename)
    x = data[:,xcol]
    y = data[:,ycol]
    if volume_func is not None:
        x = volume_func(x)
    result = fit_eos(x, y)
    return result


def cmdline_tool():
    import argparse
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument('file')
    parser.add_argument('--lattice')
    parser.add_argument('--bsse')
    parser.add_argument('--xcol', type=int, default=0)
    parser.add_argument('--ycol', type=int, default=1)
    parser.add_argument('--no-plot', dest='plot', action='store_false', default=True)
    parser.add_argument('--value-at', type=float)
    args = parser.parse_args()

    volume_func = lambda x : x
    if args.lattice == 'diamond':
        volume_func = lambda x : (x**3) / 4
        inv_volume_func = lambda v : np.power(4*v, 1.0/3)
    elif args.lattice is not None:
        raise ValueError()

    data = np.loadtxt(args.file)
    print("Loading file %s using columns x= %d y= %d" % (args.file, args.xcol, args.ycol))
    x = data[:,args.xcol]
    y = data[:,args.ycol]
    if args.bsse:
        print("Loading BSSE from file %s" % args.bsse)
        bsse_data = np.loadtxt(args.bsse)
        assert np.allclose(bsse_data[:,args.xcol], x)
        y = y - bsse_data[:,args.ycol]

    x = volume_func(x)
    result = fit_eos(x, y, plot=args.plot, value_at=args.value_at)
    result.x0 = inv_volume_func(result.v0)
    print(result)


if __name__ == '__main__':
    cmdline_tool()
