import functools
from timeit import default_timer as timer
import numpy as np

__all__ = ["timer", "einsum", "save_gf"]

einsum = functools.partial(np.einsum, optimize=True)



def save_gf(filename, freq, gf):
    base, ext = filename.rsplit(".")
    if ext: ext = "." + ext

    # Trick for imaginary part:
    freq = freq + 1j*freq
    data = np.hstack((freq[:,None], gf.reshape(gf.shape[0], -1)))
    np.savetxt("%s-re%s" % (base, ext), data.real)
    np.savetxt("%s-im%s" % (base, ext), data.imag)
