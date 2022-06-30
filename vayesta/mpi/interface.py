import logging
import functools
from timeit import default_timer

import vayesta
from vayesta.core.util import *
from .rma import RMA_Dict
from .scf import scf_with_mpi


class MPI_Interface:

    def __init__(self, mpi, required=False, log=None):
        self.log = log or logging.getLogger(__name__)
        if mpi == 'mpi4py':
            mpi = self._import_mpi4py(required=required)
        if mpi:
            self.MPI = mpi
            self.world = mpi.COMM_WORLD
            self.rank = self.world.Get_rank()
            self.size = self.world.Get_size()
            self.timer = mpi.Wtime
        else:
            self.MPI = None
            self.world = None
            self.rank = 0
            self.size = 1
            self.timer = default_timer
        self._tag = -1

    def _import_mpi4py(self, required=True):
        try:
            import mpi4py
            mpi4py.rc.threads = False
            from mpi4py import MPI as mpi
            return mpi
        except (ModuleNotFoundError, ImportError) as e:
            if required:
                self.log.critical("mpi4py not found.")
                raise e
            self.log.debug("mpi4py not found.")
            return None

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

    def nreduce(self, *args, target=None, logfunc=None, **kwargs):
        """(All)reduce multiple arguments.

        TODO:
        * Use Allreduce/Reduce for NumPy types
        * combine multiple *args of same dtype into a single array,
        to reduce communication overhead.
        """
        if logfunc is None:
            logfunc = vayesta.log.timingv
        if target is None:
            with log_time(logfunc, "Time for MPI allreduce: %s"):
                res = [self.world.allreduce(x, **kwargs) for x in args]
        else:
            with log_time(logfunc, "Time for MPI reduce: %s"):
                res = [self.world.reduce(x, root=target, **kwargs) for x in args]
        if len(res) == 1:
            return res[0]
        return tuple(res)

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
                        self.log.debugv("MPI[%d]<send>: func=%s dest=%d", self.rank, func.__name__, dest)
                        self.world.send(res, dest=dest, tag=tag2, **mpi_kwargs)
                        self.log.debugv("MPI[%d]<send>: done", self.rank)
                    return res
                elif self.rank == dest:
                    self.log.debugv("MPI[%d] <recv>: func=%s source=%d", self.rank, func.__name__, src)
                    res = self.world.recv(source=src, tag=tag2, **mpi_kwargs)
                    self.log.debugv("MPI[%d] <recv>: type= %r done!", self.rank, type(res))
                    return res
                else:
                    self.log.debugv("MPI[%d] <do nothing> func=%s source=%d ", self.rank, func.__name__, src)
                return None
            return wrapper
        return decorator

    def create_rma_dict(self, dictionary):
        return RMA_Dict.from_dict(self, dictionary)

    # --- PySCF decorators
    # --------------------

    def scf(self, mf, mpi_rank=0, log=None):
        return scf_with_mpi(self, mf, mpi_rank=mpi_rank, log=log)
