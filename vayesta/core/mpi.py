import logging
import functools
from timeit import default_timer

log = logging.getLogger(__name__)

try:
    #raise ImportError()
    from mpi4py import MPI
    mpi_world = MPI.COMM_WORLD
    mpi_rank = mpi_world.Get_rank()
    mpi_size = mpi_world.Get_size()
    mpi_timer = MPI.Wtime
except (ModuleNotFoundError, ImportError):
    MPI = None
    mpi_world = None
    mpi_rank = 0
    mpi_size = 1
    mpi_timer = default_timer

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
    timer = mpi_timer

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

mpi = MPI_Interface()
