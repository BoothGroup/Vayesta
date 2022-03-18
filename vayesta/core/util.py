import os
import sys
import logging
import dataclasses
import copy
import functools
from timeit import default_timer
from contextlib import contextmanager

try:
    import psutil
except (ModuleNotFoundError, ImportError):
    psutil = None

import numpy as np
import scipy
import scipy.linalg
import scipy.optimize

log = logging.getLogger(__name__)

# util module can be imported as *, such that the following is imported:
__all__ = [
        # General
        'Object', 'NotSet', 'OptionsBase', 'StashBase',
        # NumPy replacements
        'dot', 'einsum', 'hstack',
        # New exceptions
        'AbstractMethodError', 'ConvergenceError', 'OrthonormalityError', 'ImaginaryPartError',
        'NotCalculatedError',
        # Energy
        'energy_string',
        # Time & memory
        'timer', 'time_string', 'log_time', 'memory_string', 'get_used_memory',
        # RHF/UHF abstraction
        'dot_s', 'eigh_s', 'stack_mo', 'stack_mo_coeffs',
        # Other
        'brange',
        'deprecated',
        'replace_attr', 'cached_method', 'break_into_lines', 'fix_orbital_sign',
        ]

class Object:
    pass

class NotSetType:
    def __repr__(self):
        return 'NotSet'
"""Sentinel, use this to indicate that an option/attribute has not been set,
in cases where `None` itself is a valid setting.
"""
NotSet = NotSetType()

# --- NumPy

def dot(*args, out=None):
    """Like NumPy's multi_dot, but variadic"""
    return np.linalg.multi_dot(args, out=out)

def einsum(*args, **kwargs):
    kwargs['optimize'] = kwargs.pop('optimize', True)
    try:
        res = np.einsum(*args, **kwargs)
    except ValueError:
        log.fatal("einsum('%s',...) failed. shapes of arguments:", args[0])
        for i, arg in enumerate(args[1:]):
            log.fatal('%d: %r', i, list(np.asarray(arg).shape))
        raise
    # Unpack scalars (for optimize = True):
    if isinstance(res, np.ndarray) and res.ndim == 0:
        res = res[()]
    return res

def hstack(*args):
    """Like NumPy's hstack, but variadic, ignores any arguments which are None and improved error message."""
    args = [x for x in args if x is not None]
    try:
        return np.hstack(args)
    except ValueError as e:
        log.critical("Exception while trying to stack the following objects:")
        for x in args:
            log.critical("type= %r  shape= %r", type(x), x.shape if hasattr(x, 'shape') else "None")
        raise e

# RHF / UHF abstraction

def dot_s(*args, out=None):
    """Generalizes dot with or without spin channel: ij,jk->ik or Sij,Sjk->Sik

    Additional non spin-dependent matrices can be present, eg. Sij,jk,Skl->Skl.

    Note that unlike numpy.dot, this does not support vectors."""
    maxdim = np.max([np.ndim(x[0]) for x in args]) + 1
    # No spin-dependent arguments present
    if maxdim == 2:
        return dot(*args, out=out)
    # Spin-dependent arguments present
    assert (maxdim == 3)
    if out is None:
        out = 2*[None]
    args_a = [(x if np.ndim(x[0]) < 2 else x[0]) for x in args]
    args_b = [(x if np.ndim(x[1]) < 2 else x[1]) for x in args]
    return (dot(*args_a, out=out[0]), dot(*args_b, out=out[1]))

def eigh_s(a, b=None, *args, **kwargs):
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

def stack_mo(*mo_coeffs):
    ndim = np.ndim(mo_coeffs[0][0]) + 1
    # RHF
    if ndim == 2:
        return hstack(*mo_coeffs)
    # UHF
    if ndim == 3:
        return (hstack(*[c[0] for c in mo_coeffs]),
                hstack(*[c[1] for c in mo_coeffs]))
    raise ValueError("Unknown shape of MO coefficients: ndim= %d" % ndim)

stack_mo_coeffs = stack_mo

def brange(*args, minstep=1, maxstep=None):
    """Similar to PySCF's prange, but returning a slice instead.

    Start, stop, and blocksize can be accessed from each slice blk as
    blk.start, blk.stop, and blk.step.
    """
    if len(args) == 1:
        stop = args[0]
        start = 0
        step = 1
    elif len(args) == 2:
        start, stop = args[:2]
        step = 1
    elif len(args) == 3:
        start, stop, step = args
    else:
        raise ValueError()

    if stop <= start:
        return
    if maxstep is None:
        maxstep = (stop-start)
    step = int(np.clip(step, minstep, maxstep))
    for i in range(start, stop, step):
        blk = np.s_[i:min(i+step, stop)]
        yield blk

def cached_method(cachename, use_cache_default=True, store_cache_default=True):
    """Cache the return value of a class method.

    This adds the parameters `use_cache` and `store_cache` to the method
    signature; the default values for both parameters is `True`."""
    def cached_function(func):
        nonlocal cachename

        def is_cached(self):
            return (hasattr(self, cachename) and get_cache(self) is not None)

        def get_cache(self):
            return getattr(self, cachename)

        def set_cache(self, value):
            return setattr(self, cachename, value)

        @wraps(func)
        def wrapper(self, *args, use_cache=use_cache_default, store_cache=store_cache_default, **kwargs):
            if use_cache and is_cached(self):
                return get_cache(self)
            val = func(self, *args, **kwargs)
            if store_cache:
                set_cache(self, val)
            return val
        return wrapper
    return cached_function


# --- Exceptions

class AbstractMethodError(NotImplementedError):
    pass

class ConvergenceError(RuntimeError):
    pass

class ImaginaryPartError(RuntimeError):
    pass

class OrthonormalityError(RuntimeError):
    pass

class NotCalculatedError(AttributeError):
    """Raise if a necessary attribute has not been calculated."""
    pass

# --- Energy

def energy_string(energy, unit='Ha'):
    if unit == 'eV':
        energy *= 27.211386245988
    if unit: unit = ' %s' % unit
    return '%+16.8f%s' % (energy, unit)

# --- Time and memory

timer = default_timer

@contextmanager
def log_time(logger, message, *args, mintime=None, **kwargs):
    """Log time to execute the body of a with-statement.

    Use as:
        >>> with log_time(log.info, 'Time for hcore: %s'):
        >>>     hcore = mf.get_hcore()

    Parameters
    ----------
    logger
    message
    """
    try:
        t0 = timer()
        yield t0
    finally:
        t = (timer()-t0)
        if mintime is None or t >= mintime:
            logger(message, *args, time_string(t), **kwargs)

def time_string(seconds, show_zeros=False):
    """String representation of seconds."""
    m, s = divmod(seconds, 60)
    if seconds >= 3600 or show_zeros:
        tstr = "%.0f h %.0f min %.0f s" % (divmod(m, 60) + (s,))
    elif seconds >= 60:
        tstr = "%.0f min %.1f s" % (m, s)
    else:
        tstr = "%.3f s" % s
    return tstr

MEMUNITS = {'b': 1, 'kb': 1e3, 'mb': 1e6, 'gb': 1e9, 'tb': 1e12}

def get_used_memory(unit='b'):
    if psutil is not None:
        process = psutil.Process(os.getpid())
        mem = process.memory_info().rss  # in bytes
    # Fallback: use os module
    elif sys.platform.startswith('linux'):
        pagesize = os.sysconf("SC_PAGE_SIZE")
        with open("/proc/%s/statm" % os.getpid()) as f:
            mem = int(f.readline().split()[1])*pagesize
    else:
        mem = 0
    mem /= MEMUNITS[unit.lower()]
    return mem

def memory_string(nbytes, fmt='6.2f'):
    """String representation of nbytes"""
    if isinstance(nbytes, np.ndarray) and nbytes.size > 1:
        nbytes = nbytes.nbytes
    if nbytes < 1e3:
        val = nbytes
        unit = "B"
    elif nbytes < 1e6:
        val = nbytes / 1e3
        unit = "kB"
    elif nbytes < 1e9:
        val = nbytes / 1e6
        unit = "MB"
    elif nbytes < 1e12:
        val = nbytes / 1e9
        unit = "GB"
    else:
        val = nbytes / 1e12
        unit = "TB"
    return "{:{fmt}} {unit}".format(val, unit=unit, fmt=fmt)

# ---

@contextmanager
def replace_attr(obj, **kwargs):
    """Temporary replace attributes and methods of object."""
    orig = {}
    try:
        for name, attr in kwargs.items():
            orig[name] = getattr(obj, name)             # Save originals
            if callable(attr):
                setattr(obj, name, attr.__get__(obj))   # For functions: replace and bind as method
            else:
                setattr(obj, name, attr)                # Just set otherwise
        yield obj
    finally:
        # Restore originals
        for name, attr in orig.items():
            setattr(obj, name, attr)

class SelectNotSetType:
    """Sentinel for implementation of `select` in `Options.replace`.
    Having a dedicated sentinel allows usage of `None` and `NotSet` as `select`.
    """
    def __repr__(self):
        return 'SelectNotSet'
SelectNotSet = SelectNotSetType()

def break_into_lines(string, linelength=80, sep=None, newline='\n'):
    """Break a long string into multiple lines"""
    split = string.split(sep)
    lines = [split[0]]
    for s in split[1:]:
        if (len(lines[-1]) + 1 + len(s)) > linelength:
            # Start new line
            lines.append(s)
        else:
            lines[-1] += ' ' + s
    return newline.join(lines)

def deprecated(message=None):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""
    def decorator(func):
        if message is None:
            msg = "Function %s is deprecated!" % func.__name__
        else:
            msg = message

        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            log.warning(msg)
            return func(*args, **kwargs)
        return wrapped
    return decorator

class OptionsBase:
    """Abstract base class for Option dataclasses.

    This should be inherited and decorated with `@dataclasses.dataclass`.
    One can then define attributes and default values as for any dataclass.
    This base class provides some dictionary-like methods, like `get` and `items`
    and also the method `replace`, in order to update options from another Option object
    or dictionary.
    """

    def __repr__(self):
        return "Options(%r)" % self.asdict()

    def get(self, attr, default=None):
        """Dictionary-like access to attributes.
        Allows the definition of a default value, of the attribute is not present.
        """
        if hasattr(self, attr):
            return getattr(self, attr)
        return default

    def asdict(self):
        """Do not use `dataclasses.asdict(self)`, which creates deep copies of values."""
        return self.__dict__

    def keys(self):
        return self.asdict().keys()

    def values(self):
        return self.asdict().values()

    def items(self):
        return self.asdict().items()

    def replace(self, other, select=SelectNotSet, **kwargs):
        """Replace some or all attributes.

        Parameters
        ----------
        other : Options or dict
            Options object or dictionary defining the attributes which should be replaced.
        select :
            If set, only values which are of the corresponding type will be replaced.
        **kwargs :
            Additional keyword arguments will be added to `other`
        """
        other = copy.deepcopy(other)
        if isinstance(other, OptionsBase):
            other = other.asdict()
        if kwargs:
            other.update(kwargs)

        # Only replace values which are equal to select
        if select is not SelectNotSet:
            keep = {}
            for key, val in self.items():
                if (val is select) and (key in other):
                    #updates[key] = copy.copy(other[key])
                    keep[key] = other[key]
            other = keep

        return dataclasses.replace(self, **other)

class StashBase:
    pass

def fix_orbital_sign(mo_coeff, inplace=True):
    # UHF
    if np.ndim(mo_coeff[0]) == 2:
        mo_coeff_a, sign_a = fix_orbital_sign(mo_coeff[0], inplace=inplace)
        mo_coeff_b, sign_b = fix_orbital_sign(mo_coeff[1], inplace=inplace)
        return (mo_coeff_a, mo_coeff_b), (sign_a, sign_b)
    if not inplace:
        mo_coeff = mo_coeff.copy()
    absmax = np.argmax(abs(mo_coeff), axis=0)
    nmo = mo_coeff.shape[-1]
    swap = mo_coeff[absmax,np.arange(nmo)] < 0
    mo_coeff[:,swap] *= -1
    signs = np.ones((nmo,), dtype=int)
    signs[swap] = -1
    return mo_coeff, signs

if __name__ == '__main__':
    a1 = np.random.rand(2, 3)
    a2 = np.random.rand(2, 3)
    s = np.random.rand(3, 3)
    b1 = np.random.rand(3, 4)
    b2 = np.random.rand(3, 4)

    c1, c2 = dot_s((a1, a2), s, (b1, b2))
    assert np.allclose(c1, dot(a1, s, b1))
    assert np.allclose(c2, dot(a2, s, b2))

    ha = np.random.rand(3,3)
    hb = np.random.rand(3,3)
    ba = np.random.rand(3,3)
    bb = np.random.rand(3,3)
    ba = np.dot(ba, ba.T)
    bb = np.dot(bb, bb.T)
    #b =b a

    ea, va = scipy.linalg.eigh(ha, b=ba)
    eb, vb = scipy.linalg.eigh(hb, b=ba)

    h = (ha, hb)
    e, v = eigh_s(h, ba)
    print(ea)
    print(eb)
    print(e)

    assert np.allclose(e[0], ea)
    assert np.allclose(e[1], eb)



    #d1, d2 = einsum('[s]ij,jk,[s]kl->[S]il', (a1, a2), s, (b1, b2))
    #d1, d2 = einsum('?ij,jk,?kl->?il', (a1, a2), s, (b1, b2))
    #print(d1.shape)
    #print(c1.shape)
    #assert np.allclose(d1, c1)
    #assert np.allclose(d2, c2)
    #assert np.allclose(d1, dot(a1, s, b1))
    #assert np.allclose(d2, dot(a2, s, b2))
