from contextlib import contextmanager
from copy import deepcopy
import dataclasses
import functools
import logging
import os
import re
import string
import sys
from timeit import default_timer


try:
    import psutil
except (ModuleNotFoundError, ImportError):
    psutil = None

import numpy as np
import scipy
import scipy.linalg
import scipy.optimize


modlog = logging.getLogger(__name__)

# util module can be imported as *, such that the following is imported:
__all__ = [
        # General
        'Object', 'OptionsBase', 'brange', 'deprecated', 'cache',
        # NumPy replacements
        'dot', 'einsum', 'hstack',
        # Exceptions
        'AbstractMethodError', 'ConvergenceError', 'OrthonormalityError', 'ImaginaryPartError',
        'NotCalculatedError',
        # String formatting
        'energy_string', 'time_string', 'memory_string',
        # Time & memory
        'timer', 'log_time', 'get_used_memory', 'log_method',
        # Other
        'replace_attr', 'break_into_lines', 'fix_orbital_sign', 'split_into_blocks',
        ]

class Object:
    pass

def cache(maxsize=16, typed=False, copy=False):
    """Adds LRU cache to function or method.

    If the function or method returns a mutable object, e.g. a NumPy array,
    cache hits will return the same object. If the object has been modified
    (for example by the user on the script level), the modified object will be
    returned by future calls. To avoid this, a (deep)copy of the result can be
    performed, by setting copy=True.

    modified from https://stackoverflow.com/questions/54909357
    """
    lru_cache = functools.lru_cache(maxsize, typed)
    if not copy:
        return lru_cache
    def decorator(func):
        cached_func = lru_cache(func)
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            return deepcopy(cached_func(*args, **kwargs))
        wrapper.cache_clear = cached_func.cache_clear
        wrapper.cache_info = cached_func.cache_info
        # Python 3.9+
        if hasattr(cached_func, 'cache_parameters'):
            wrapper.cache_parameters = cached_func.cache_parameters
        return wrapper
    return decorator

# --- NumPy

def dot(*args, out=None, ignore_none=False):
    """Like NumPy's multi_dot, but variadic"""
    if ignore_none:
        args = [a for a in args if a is not None]
    return np.linalg.multi_dot(args, out=out)

def _einsum_replace_decorated_subscripts(subscripts):
    """Support for decorated indices: a!, b$, c3, d123.

    Characters in ',->.()[]{}' cannot be used as decorators.
    """
    free = sorted(set(string.ascii_letters).difference(set(subscripts)))
    keep = (string.ascii_letters + ' ,->.()[]{}')
    replaced = {}
    subscripts_out = []
    for char in subscripts:
        if char in keep:
            subscripts_out += char
            continue
        else:
            last = (subscripts_out.pop() if len(subscripts_out) else '%')
            if last not in string.ascii_letters:
                raise ValueError("Invalid subscripts: '%s'" % subscripts)
            comb = (last + char)
            if comb not in replaced:
                replaced[comb] = free.pop()
            subscripts_out += replaced[comb]
    return ''.join(subscripts_out)

def _ordered_einsum(einsumfunc, subscripts, *operands, **kwargs):
    """Support for parenthesis in einsum subscripts: '(ab,bc),cd->ad'."""

    def resolve(subs, *ops):
        #print('resolve called with %s and %d operands' % (subs, len(ops)))

        idx_right = re.sub('[\]}]', ')', subs).find(')')
        idx_left = re.sub('[\[{]', '(', subs[:idx_right]).rfind('(')

        if idx_left == idx_right == -1:
            return einsumfunc(subs, *ops, **kwargs)
        if (idx_left == -1 or idx_right == -1):
            raise ValueError("Unmatched parenthesis: '%s'" % subs)
        bracket_types = {'(': ')', '[': ']', '{': '}'}
        if subs[idx_right] != bracket_types[subs[idx_left]]:
            raise ValueError("Unmatched parenthesis: '%s'" % subs)

        subs_int = subs[idx_left+1:idx_right]
        subs_left = subs[:idx_left]
        subs_right = subs[idx_right+1:]

        # Split operands
        nops_left = subs_left.count(',')
        nops_right = subs_right.count(',')
        nops_int = subs_int.count(',') + 1
        ops_int = ops[nops_left:nops_left+nops_int]
        ops_left = ops[:nops_left]
        ops_right = ops[nops_left+nops_int:]

        if '->' in subs_int:
            subs_int_in, subs_int_out = subs_int.split('->')
        else:
            subs_int_in = subs_int

            #possible = subs_int_in.replace(',', '').replace(' ', '')
            #subs_int_out = ''.join([x for x in possible if x in (subs_left + subs_right)])
            #subs_int = '->'.join([subs_int_in, subs_int_out])
            subs_int_out = np.core.einsumfunc._parse_einsum_input((subs_int_in, *ops_int))[1]

        # Perform intern einsum
        res_int = einsumfunc(subs_int, *ops_int, **kwargs)
        # Resolve recursively
        subs_ext = subs_left + subs_int_out +  subs_right
        ops_ext = ops_left + (res_int,) + ops_right
        return resolve(subs_ext, *ops_ext)

    res = resolve(subscripts, *operands)
    return res

def einsum(subscripts, *operands, **kwargs):
    subscripts = _einsum_replace_decorated_subscripts(subscripts)

    if np.any([x in subscripts for x in '()[]{}']):
        return _ordered_einsum(einsum, subscripts, *operands, **kwargs)

    kwargs['optimize'] = kwargs.get('optimize', True)
    driver = kwargs.get('driver', np.einsum)
    try:
        res = driver(subscripts, *operands, **kwargs)
    # Better shape information in case of exception:
    except ValueError:
        modlog.fatal("einsum('%s',...) failed. shapes of arguments:", subscripts)
        for i, arg in enumerate(operands):
            modlog.fatal('%d: %r', i, list(np.asarray(arg).shape))
        raise
    # Unpack scalars (for optimize = True):
    if isinstance(res, np.ndarray) and res.ndim == 0:
        res = res[()]
    return res

def hstack(*args, ignore_none=True):
    """Like NumPy's hstack, but variadic, ignores any arguments which are None and improved error message."""
    if ignore_none:
        args = [x for x in args if x is not None]
    try:
        return np.hstack(args)
    except ValueError as e:
        modlog.critical("Exception while trying to stack the following objects:")
        for x in args:
            modlog.critical("type= %r  shape= %r", type(x), x.shape if hasattr(x, 'shape') else "None")
        raise e

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

def split_into_blocks(array, axis=0, blocksize=None, max_memory=int(1e9)):
    size = array.shape[axis]
    axis = axis % array.ndim
    if blocksize is None:
        mem = array.nbytes
        nblocks = max(int(np.ceil(mem/max_memory)), 1)
        blocksize = int(np.ceil(size/nblocks))
    if blocksize >= size:
        yield slice(None), array
        return
    for i in range(0, size, blocksize):
        blk = np.s_[i:min(i+blocksize, size)]
        yield blk, array[axis*(slice(None), ) + (blk,)]

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
        if logger and (mintime is None or t >= mintime):
            logger(message, *args, time_string(t), **kwargs)

def log_method(message='Time for %(classname).%(funcname): %s', log=None):
    def decorator(func):
        @functools.wraps(func)
        def wrapped(self, *args, **kwargs):
            nonlocal message, log
            message = message.replace('%(classname)', type(self).__name__)
            message = message.replace('%(funcname)', func.__name__)
            log = log or getattr(self, 'log', False) or modlog
            log.debugv("Entering method '%s'", func.__name__)
            with log_time(log.timing, message):
                res = func(self, *args, **kwargs)
            log.debugv("Exiting method '%s'", func.__name__)
            return res
        return wrapped
    return decorator

def time_string(seconds, show_zeros=False):
    """String representation of seconds."""
    seconds, sign = abs(seconds), np.sign(seconds)
    m, s = divmod(seconds, 60)
    if seconds >= 3600 or show_zeros:
        tstr = "%.0f h %.0f min" % divmod(m, 60)
    elif seconds >= 60:
        tstr = "%.0f min %.0f s" % (m, s)
    else:
        tstr = "%.1f s" % s
    if sign == -1:
        tstr = '-%s' %  tstr
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

def break_into_lines(string, linelength=100, sep=None, newline='\n'):
    """Break a long string into multiple lines"""
    if len(string) <= linelength:
        return string
    split = string.split(sep)
    lines = [split[0]]
    for s in split[1:]:
        if (len(lines[-1]) + 1 + len(s)) > linelength:
            # Start new line
            lines.append(s)
        else:
            lines[-1] += ' ' + s
    return newline.join(lines)

def deprecated(message=None, replacement=None):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""
    def decorator(func):
        if message is not None:
            msg = message
        else:
            msg = "Function `%s` is deprecated." % func.__name__
            if replacement is not None:
                msg += " Use `%s` instead." % replacement

        @functools.wraps(func)
        def wrapped(*args, **kwargs):
            if len(args) > 0 and hasattr(args[0], 'log'):
                log = args[0].log
            else:
                log = modlog
            log.warning(msg)
            return func(*args, **kwargs)
        return wrapped
    return decorator

@dataclasses.dataclass
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

    @classmethod
    def get_default(cls, field):
        for x in dataclasses.fields(cls):
            if (x.name == field):
                return x.default
        raise TypeError

    @classmethod
    def get_default_factory(cls, field):
        for x in dataclasses.fields(cls):
            if (x.name == field):
                return x.default_factory
        raise TypeError

    def replace(self, **kwargs):
        keys = self.keys()
        for key, val in kwargs.items():
            if key not in keys:
                raise TypeError("replace got an unexpected keyword argument '%s'" % key)
            if isinstance(val, dict) and isinstance(getattr(self, key), dict):
                setattr(self, key, {**getattr(self, key), **val})
            else:
                setattr(self, key, val)
        return self

    def update(self, **kwargs):
        keys = self.keys()
        for key, val in kwargs.items():
            if key not in keys:
                continue
            if isinstance(val, dict) and isinstance(getattr(self, key), dict):
                #getattr(self, key).update(val)
                setattr(self, key, {**getattr(self, key), **val})
            else:
                setattr(self, key, val)
        return self

    @staticmethod
    def dict_with_defaults(**kwargs):
        return dataclasses.field(default_factory=lambda: kwargs)

    @classmethod
    def change_dict_defaults(cls, field, **kwargs):
        defaults = cls.get_default_factory(field)()
        return cls.dict_with_defaults(**{**defaults, **kwargs})

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
