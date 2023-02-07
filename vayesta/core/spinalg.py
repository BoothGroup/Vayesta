"""Some utility to perform operations for RHF and UHF using the
same functions"""

import numpy as np
from vayesta.core import util

__all__ = ['add_numbers', 'hstack_matrices']


def add_numbers(*args):
    # RHF
    if np.all([np.isscalar(arg) for arg in args]):
        return sum(args)
    # UHF
    if not np.any([np.isscalar(arg) for arg in args]):
        return (sum([arg[0] for arg in args]),
                sum([arg[1] for arg in args]))
    raise ValueError

def hstack_matrices(*args, ignore_none=True):
    if ignore_none:
        args = [x for x in args if x is not None]
    ndims = np.asarray([(arg[0].ndim+1) for arg in args])
    # RHF
    if np.all(ndims == 2):
        return util.hstack(*args)
    # UHF
    if np.all(ndims == 3):
        return (util.hstack(*[arg[0] for arg in args]),
                util.hstack(*[arg[1] for arg in args]))
    raise ValueError("ndims= %r" % ndims)

def dot(*args, out=None):
    """Generalizes dot with or without spin channel: ij,jk->ik or Sij,Sjk->Sik

    Additional non spin-dependent matrices can be present, eg. Sij,jk,Skl->Skl.

    Note that unlike numpy.dot, this does not support vectors."""
    maxdim = np.max([np.ndim(x[0]) for x in args]) + 1
    # No spin-dependent arguments present
    if maxdim == 2:
        return util.dot(*args, out=out)
    # Spin-dependent arguments present
    assert (maxdim == 3)
    if out is None:
        out = 2*[None]
    args_a = [(x if np.ndim(x[0]) < 2 else x[0]) for x in args]
    args_b = [(x if np.ndim(x[1]) < 2 else x[1]) for x in args]
    return (util.dot(*args_a, out=out[0]), util.dot(*args_b, out=out[1]))

def eigh(a, b=None, *args, **kwargs):
    ndim = np.ndim(a[0]) + 1
    # RHF
    if ndim == 2:
        return scipy.linalg.eigh(a, b=b, *args, **kwargs)
    # UHF
    if b is None or np.ndim(b[0]) == 1:
        b = (b, b)
    results = (scipy.linalg.eigh(a[0], b=b[0], *args, **kwargs),
               scipy.linalg.eigh(a[1], b=b[1], *args, **kwargs))
    return tuple(zip(*results))

def transpose(a, axes=None):
    if np.ndim(a[0]) == 1:
        return np.transpose(a, axes=axes)
    return (transpose(a[0], axes=axes), transpose(a[1], axes=axes))

T = transpose

def _guess_spinsym(a):
    if isinstance(a, (tuple, list)):
        return 'unrestricted'
    return 'restricted'

def _make_func(func, nargs=1):

    def _func(a, *args, spinsym=None, **kwargs):
        spinsym = spinsym or _guess_spinsym(a)
        if spinsym == 'restricted':
            return func(a, *args, **kwargs)
        if nargs == 1:
            return tuple(func(a[i], *args, **kwargs) for i in range(len(a)))
        if nargs == 2:
            assert len(args) >= 1
            b, args = args[0], args[1:]
            return tuple(func(a[i], b[i], *args, **kwargs) for i in range(len(a)))
        if nargs == 3:
            assert len(args) >= 2
            b, c, args = args[0], args[1], args[2:]
            return tuple(func(a[i], b[i], c[i], *args, **kwargs) for i in range(len(a)))
        raise NotImplementedError
    return _func


zeros_like = _make_func(np.zeros_like)
add = _make_func(np.add, nargs=2)
subtract = _make_func(np.subtract, nargs=2)
multiply = _make_func(np.multiply, nargs=2)
copy = _make_func(np.copy, nargs=1)
norm = _make_func(np.linalg.norm, nargs=1)
