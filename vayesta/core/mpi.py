import logging
import functools
from timeit import default_timer
from contextlib import contextmanager

import numpy as np

from vayesta.core.util import *

log = logging.getLogger(__name__)

try:
    #raise ImportError()
    import mpi4py
    mpi4py.rc.threads = False
    from mpi4py import MPI
    mpi_world = MPI.COMM_WORLD
    mpi_rank = mpi_world.Get_rank()
    mpi_size = mpi_world.Get_size()
    timer = MPI.Wtime
except (ModuleNotFoundError, ImportError):
    MPI = None
    mpi_world = None
    mpi_rank = 0
    mpi_size = 1
    timer = default_timer

class MPI_Operators:
    pass

mpi_ops = MPI_Operators()
for op in ['max', 'min', 'sum', 'prod', 'land', 'lor', 'band', 'bor', 'maxloc', 'minloc']:
    setattr(mpi_ops, op, getattr(MPI, op.upper()) if MPI is not None else None)

class MPI_Interface:

    MPI = MPI
    world = mpi_world
    rank = mpi_rank
    size = mpi_size
    timer = timer

    op = mpi_ops

    def __init__(self):
        self._tag = -1

    def __len__(self):
        return self.size

    def __bool__(self):
        return self.enabled

    @property
    def enabled(self):
        return (self.size > 1)

    @property
    def disabled(self):
        return not self.enabled

    @property
    def is_master(self):
        return (self.rank == 0)

    def get_new_tag(self):
        self._tag += 1
        return self._tag

    # --- Function wrapper at embedding level
    # ---------------------------------------

    def with_reduce(self, **mpi_kwargs):
        def decorator(func):
            # No MPI:
            if self.disabled:
                return func
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                res = func(*args, **kwargs)
                res = self.world.reduce(res, **mpi_kwargs)
                return res
            return wrapper
        return decorator

    def with_allreduce(self, **mpi_kwargs):
        def decorator(func):
            # No MPI:
            if self.disabled:
                return func
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                res = func(*args, **kwargs)
                res = self.world.allreduce(res, **mpi_kwargs)
                return res
            return wrapper
        return decorator

    def only_master(self):
        def decorator(func):
            # No MPI:
            if self.disabled:
                return func
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if not mpi.is_master:
                    return None
                return func(*args, **kwargs)
            return wrapper
        return decorator

    # --- Function wrapper at fragment level
    # --------------------------------------

    def with_send(self, source, dest=0, tag=None, **mpi_kwargs):
        def decorator(func):
            # No MPI:
            if self.disabled:
                return func
            # With MPI:
            tag2 = self.get_new_tag() if tag is None else tag
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                if callable(source):
                    src = source(*args)
                else:
                    src = source
                if self.rank == src:
                    res = func(*args, **kwargs)
                    if (self.rank != dest):
                        log.debugv("MPI[%d]<send>: func=%s dest=%d", self.rank, func.__name__, dest)
                        self.world.send(res, dest=dest, tag=tag2, **mpi_kwargs)
                        log.debugv("MPI[%d]<send>: done", self.rank)
                    return res
                elif self.rank == dest:
                    log.debugv("MPI[%d] <recv>: func=%s source=%d", self.rank, func.__name__, src)
                    res = self.world.recv(source=src, tag=tag2, **mpi_kwargs)
                    log.debugv("MPI[%d] <recv>: type= %r done!", self.rank, type(res))
                    return res
                else:
                    log.debugv("MPI[%d] <do nothing> func=%s source=%d ", self.rank, func.__name__, src)
                return None
            return wrapper
        return decorator

    def create_rma_dict(self, dictionary):
        return RMA_Dict.from_dict(self, dictionary)

class RMA_Dict:

    class RMA_DictElement:

        def __init__(self, collection, location, data=None, shape=None, dtype=None):
            self.collection = collection
            self.location = location
            self.shape = shape
            self.dtype = dtype

            # Allocate RMA Window and put data
            self.win = None
            if self.dtype != type(None):
                if (self.mpi.rank == self.location):
                    self.local_init(data)
                else:
                    self.remote_init()

        def local_init(self, data):
            #if data is not None:
            #    winsize = (data.size * data.dtype.itemsize)
            #else:
            #    winsize = 0
            #self.win = self.mpi.MPI.Win.Allocate(winsize, comm=self.mpi.world)
            #self.win = self.mpi.MPI.Win.Create(data, comm=self.mpi.world)
            #if data is None:
            #    return

            winsize = (data.size * data.dtype.itemsize)
            self.win = self.mpi.MPI.Win.Allocate(winsize, comm=self.mpi.world)
            assert (self.shape == data.shape)
            assert (self.dtype == data.dtype)
            self.rma_lock()
            self.rma_put(data)
            self.rma_unlock()

        def remote_init(self):
            #if self.dtype == type(None):
            #    return
            self.win = self.mpi.MPI.Win.Allocate(0, comm=self.mpi.world)
            #self.win = self.mpi.MPI.Win.Create(None, comm=self.mpi.world)
            #buf = np.empty(self.shape, dtype=self.dtype)
            #self.win = self.mpi.MPI.Win.Create(buf, comm=self.mpi.world)

        @property
        def size(self):
            if self.shape is None:
                return 0
            return np.product(self.shape)

        #@property
        #def itemsize(self):
        #    if self.dtype is type(None):
        #        return 0
        #    return self.dtype.itemsize

        #@property
        #def winsize(self):
        #    return self.size * self.itemsize

        def get(self, shared_lock=True):
            if self.dtype == type(None):
                return None
            buf = np.empty(self.shape, dtype=self.dtype)
            self.rma_lock(shared_lock=shared_lock)
            self.rma_get(buf)
            self.rma_unlock()
            return buf

        @property
        def mpi(self):
            return self.collection.mpi

        def rma_lock(self, shared_lock=False, **kwargs):
            if shared_lock:
                return self.win.Lock(self.location, lock_type=self.mpi.MPI.LOCK_SHARED, **kwargs)
            return self.win.Lock(self.location)

        def rma_unlock(self, **kwargs):
            return self.win.Unlock(self.location, **kwargs)

        def rma_put(self, data, **kwargs):
            return self.win.Put(data, target_rank=self.location, **kwargs)

        def rma_get(self, buf, **kwargs):
            return self.win.Get(buf, target_rank=self.location, **kwargs)

        def free(self):
            return self.win.Free()

    def __init__(self, mpi):
        self.mpi = mpi
        self._writable = False
        self.local_data = {}
        self._elements = {}

    @classmethod
    def from_dict(cls, mpi, dictionary):
        rma_dict = RMA_Dict(mpi)
        with rma_dict.writable():
            for key, val in dictionary.items():
                rma_dict[key] = val
        return rma_dict

    @property
    def readable(self):
        return not self._writable

    def __getitem__(self, key):
        if not self.readable:
            raise AttributeError("Cannot read from ArrayCollection from inside with-statement.")
        # Is local access without going via MPI.Get safe?
        #if key in self.local_data:
        #    return self.local_data[key]
        if mpi.disabled:
            return self._elements[key]
        element = self._elements[key]
        log.debugv("RMA: origin= %d, target= %d, key= %r, shape= %r, dtype= %r", self.mpi.rank, element.location, key, element.shape, element.dtype)
        return element.get()

    def __setitem__(self, key, value):
        if not self._writable:
            raise AttributeError("Cannot write to ArrayCollection outside of with-statement.")
        if not isinstance(value, (np.ndarray, type(None))):
            #value = np.asarray(value)
            raise ValueError("Invalid type= %r" % type(value))
        if mpi.disabled:
            self._elements[key] = value
            return
        self.local_data[key] = value

    def __delitem__(self, key):
        if not self._writable:
            raise AttributeError("Cannot write to ArrayCollection outside of with-statement.")
        del self._elements[key]

    def __enter__(self):
        self._writable = True
        return self

    def __exit__(self, type, value, traceback):
        self._writable = False
        self.synchronize()

    def clear(self):
        if self.mpi.enabled:
            for item in self.values():
                item.free()
        self._elements.clear()

    @contextmanager
    def writable(self):
        try:
            yield self.__enter__()
        finally:
            self.__exit__(None, None, None)

    def _get_metadata(self):
        """Get shapes and datatypes of local data."""
        #return {key: (getattr(val, 'shape', None), getattr(val, 'dtype', type(None))) for key, val in self.local_data.items()}
        mdata = {}
        for key, val in self.local_data.items():
            shape = getattr(val, 'shape', None)
            dtype = getattr(val, 'dtype', type(None))
            mdata[key] = (shape, dtype)
        return mdata

    def __contains__(self, key):
        return key in self.keys()

    def __len__(self):
        return len(self.keys())

    def keys(self):
        if not self.readable:
            raise RuntimeError("Cannot access keys inside of with-statement.""")
        return self._elements.keys()

    def values(self):
        if not self.readable:
            raise RuntimeError("Cannot access values inside of with-statement.""")
        return self._elements.values()

    def get_location(self, key):
        return self._elements[key].location

    def get_shape(self, key):
        return self._elements[key].shape

    def get_dtype(self, key):
        return self._elements[key].dtype

    def synchronize(self):
        """Synchronize keys and metadata over all MPI ranks."""
        if self.mpi.disabled:
            return
        self.mpi.world.Barrier()
        mdata = self._get_metadata()
        allmdata = self.mpi.world.allgather(mdata)
        assert (len(allmdata) == len(mpi))
        elements = {}
        for rank, d in enumerate(allmdata):
            for key, mdata in d.items():
                #print("Rank %d has key: %r" % (rank, key))
                if key in elements:
                    raise AttributeError("Key '%s' used multiple times. Keys need to be unique." % key)
                shape, dtype = mdata
                if rank == self.mpi.rank:
                    data = self.local_data[key]
                    elements[key] = self.RMA_DictElement(self, location=rank, data=data, shape=shape, dtype=dtype)
                else:
                    elements[key] = self.RMA_DictElement(self, location=rank, shape=shape, dtype=dtype)
        self._elements.update(elements)
        self.mpi.world.Barrier()
        self.local_data = {}


mpi = MPI_Interface()
