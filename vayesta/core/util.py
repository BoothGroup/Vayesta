import os
import logging
import dataclasses
import copy
import psutil
from functools import wraps
from timeit import default_timer
from contextlib import contextmanager

import numpy as np
import scipy
import scipy.optimize

log = logging.getLogger(__name__)

# util module can be imported as *, such that the following is imported:
__all__ = [
        # General
        'NotSet', 'OptionsBase', 'StashBase',
        # NumPy replacements
        'dot', 'einsum', 'hstack',
        # New exceptions
        'ConvergenceError',
        # Time & memory
        'timer', 'time_string', 'log_time', 'memory_string', 'get_used_memory',
        # Other
        'replace_attr', 'cached_method',
        ]

class NotSetType:
    def __repr__(self):
        return 'NotSet'
"""Sentinel, use this to indicate that an option/attribute has not been set,
in cases where `None` itself is a valid setting.
"""
NotSet = NotSetType()

timer = default_timer

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

@contextmanager
def log_time(logger, message, *args, **kwargs):
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
        logger(message, time_string(t), *args, **kwargs)

# --- NumPy

def dot(*args, **kwargs):
    """Like NumPy's multi_dot, but variadic."""
    return np.linalg.multi_dot(args, **kwargs)

def einsum(*args, **kwargs):
    kwargs['optimize'] = kwargs.pop('optimize', True)
    res = np.einsum(*args, **kwargs)
    # Unpack scalars (for optimize = True):
    if isinstance(res, np.ndarray) and res.ndim == 0:
        res = res[()]
    return res

def hstack(*args):
    """Like NumPy's hstack, but variadic and ignores any arguments which are None."""
    args = [x for x in args if x is not None]
    return np.hstack(args)

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

class ConvergenceError(RuntimeError):
    pass


def get_used_memory():
    process = psutil.Process(os.getpid())
    return(process.memory_info().rss)  # in bytes


def time_string(seconds, show_zeros=False):
    """String representation of seconds."""
    m, s = divmod(seconds, 60)
    if seconds >= 3600 or show_zeros:
        tstr = "%.0f h %.0f min %.0f s" % (divmod(m, 60) + (s,))
    elif seconds >= 60:
        tstr = "%.0f min %.1f s" % (m, s)
    else:
        tstr = "%.2f s" % s
    return tstr


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


class SelectNotSetType:
    """Sentinel for implementation of `select` in `Options.replace`.
    Having a dedicated sentinel allows usage of `None` and `NotSet` as `select`.
    """
    def __repr__(self):
        return 'SelectNotSet'
SelectNotSet = SelectNotSetType()

class OptionsBase:
    """Abstract base class for Option dataclasses.

    This should be inherited and decorated with `@dataclasses.dataclass`.
    One can then define attributes and default values as for any dataclass.
    This base class provides some dictionary-like methods, like `get` and `items`
    and also the method `replace`, in order to update options from another Option object
    or dictionary.
    """

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
