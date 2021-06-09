"""ModulX for the fitting of the hybridization function"""

import numpy as np
import scipy
import scipy.optimize

from helper import *

def get_init_guess(freq, hyb, npoles, maxfreq=1.0, multiplicity=1):
    #energy = np.linspace(-freq.max()/2, freq.max()/2, npoles)
    if maxfreq is None:
        maxfreq = freq.max()/2
    energy = np.linspace(-maxfreq, maxfreq, npoles)
    energy = np.repeat(energy, multiplicity)

    nimp = hyb.shape[-1]
    coupling = np.full((nimp, npoles, multiplicity), 1e-2)

    return energy, coupling


def kernel(dmft, freq, hyb, init_guess=None, multiplicity=1, freq_weight=-1):

    # Number of fit parameter
    nimp = hyb.shape[-1]

    if init_guess is None:
        init_guess = 5
    if np.isscalar(init_guess):
        e0, c0 = get_init_guess(freq, hyb, init_guess, multiplicity)
        npoles = init_guess
    else:
        e0, c0 = init_guess
        npoles = len(e0)

    # Check if frequencies are uniform
    #uniform = np.allclose(np.diff(freq), freq[1]-freq[0])

    # Calculate frequency grid weights
    bounds = (freq[:-1] + np.diff(freq)/2)
    bounds = np.hstack((0, bounds, bounds[-1] + (bounds[-1]-bounds[-2])/2))
    grid_weight = np.diff(bounds)

    # Frequency weighting
    if np.isscalar(freq_weight):
        freq_weight = abs(np.power(freq, freq_weight))
    elif callable(freq_weight):
        freq_weight = freq_weight(freq)
    else:
        freq_weight = np.asarray(freq_weight)

    # Combine grid_weight and freq_weight
    wgt = grid_weight * freq_weight

    def pack_vec(e, c):
        e = e[::multiplicity]
        vec = np.hstack((e, c.flatten()))
        return vec

    def unpack_vec(vec):
        e, c = np.hsplit(vec, [npoles])
        e = np.repeat(e, multiplicity)
        c = c.reshape(nimp, npoles*multiplicity)
        return e, c

    def make_fit(e, c):
        """Construct fitted hybridization"""
        fit = einsum("ai,wi,bi->wab", c, 1/np.add.outer(1j*freq, -e), c)
        return fit


    def objective_func(vec):

        e, c = unpack_vec(vec)
        fit = make_fit(e, c)
        diff = fit - hyb
        funcval = (einsum("w,wab,wab->", wgt, diff.real, diff.real)
                 + einsum("w,wab,wab->", wgt, diff.imag, diff.imag))
        dmft.log.debug("e= %r c= %r f= %.8e", e, c, funcval)

        # Gradient
        #r = 1/np.add.outer(1j*freq, -e)
        #de = einsum("

        return funcval


    vec0 = pack_vec(e0, c0)

    t0 = timer()
    res = scipy.optimize.minimize(objective_func, vec0)
    print(res.status)
    print(res.message)
    print(res.success)

    energy, coupling = unpack_vec(res.x)

    hybfit = make_fit(energy, coupling)

    return energy, coupling, hybfit

