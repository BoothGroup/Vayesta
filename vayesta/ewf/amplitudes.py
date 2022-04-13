import functools

import numpy as np

from vayesta.core.util import *
from vayesta.core.mpi import mpi

def _mpi_reduce(log, *args, mpi_target=None):
    if mpi_target is None:
        with log_time(log.timingv, "Time for MPI allreduce: %s"):
            res = [mpi.world.allreduce(x) for x in args]
    else:
        with log_time(log.timingv, "Time for MPI reduce: %s"):
            res = [mpi.world.reduce(x, root=mpi_target) for x in args]
    if len(res) == 1:
        return res[0]
    return tuple(res)

def get_global_t1_rhf(emb, get_lambda=False, mpi_target=None):
    """Get global CCSD T1 amplitudes from fragment calculations.

    Parameters
    ----------
    get_lambda: bool, optional
        If True, return L1 amplitudes. Default: False.
    mpi_target: int or None, optional
        If set to an integer, the result will only be available at the specified MPI rank.
        If set to None, an MPI allreduce will be performed and the result will be available
        at all MPI ranks. Default: None.

    Returns
    -------
    t1: (n(occ), n(vir)) array
        Global T1 amplitudes.
    """
    t1 = np.zeros((emb.nocc, emb.nvir))
    # Add fragment WFs in intermediate normalization
    for x in emb.get_fragments(mpi_rank=mpi.rank):
        emb.log.debugv("Now adding projected %s amplitudes of fragment %s", ("L" if get_lambda else "T"), x)
        ro, rv = x.get_overlap_m2c()
        pwf = x.results.pwf.restore()
        t1x = pwf.l1 if get_lambda else pwf.t1
        if t1x is None:
            raise NotCalculatedError
        t1 += einsum('ia,Ii,Aa->IA', t1x, ro, rv)
    # --- MPI
    if mpi:
        t1 = _mpi_reduce(emb.log, t1, mpi_target=mpi_target)
    return t1

def get_global_t2_rhf(emb, get_lambda=False, symmetrize=True, mpi_target=None):
    """Get global CCSD T2 amplitudes from fragment calculations.

    Parameters
    ----------
    get_lambda: bool, optional
        If True, return L1 amplitudes. Default: False.
    mpi_target: int or None, optional
        If set to an integer, the result will only be available at the specified MPI rank.
        If set to None, an MPI allreduce will be performed and the result will be available
        at all MPI ranks. Default: None.

    Returns
    -------
    t2: (n(occ), n(occ), n(vir), n(vir)) array
        Global T2 amplitudes.
    """
    t2 = np.zeros((emb.nocc, emb.nocc, emb.nvir, emb.nvir))
    # Add fragment WFs in intermediate normalization
    for x in emb.get_fragments(mpi_rank=mpi.rank):
        emb.log.debugv("Now adding projected %s-amplitudes of fragment %s", ("L" if get_lambda else "T"), x)
        ro, rv = x.get_overlap_m2c()
        pwf = x.results.pwf.restore()
        t2x = pwf.l2 if get_lambda else pwf.t2
        if t2x is None:
            raise NotCalculatedError
        t2 += einsum('ijab,Ii,Jj,Aa,Bb->IJAB', t2x, ro, ro, rv, rv)
    # --- MPI
    if mpi:
        t2 = _mpi_reduce(emb.log, t2, mpi_target=mpi_target)
    return t2

def get_global_t1_uhf(emb, get_lambda=False, mpi_target=None):
    """Get global CCSD T1 from fragment calculations.

    Parameters
    ----------
    get_lambda: bool, optional
        If True, return L1 amplitudes. Default: False.
    mpi_target: int or None, optional
        If set to an integer, the result will only be available at the specified MPI rank.
        If set to None, an MPI allreduce will be performed and the result will be available
        at all MPI ranks. Default: None.

    Returns
    -------
    t1: tuple(2) of (n(occ), n(vir)) array
        Global T1 amplitudes.
    """
    t1a = np.zeros((emb.nocc[0], emb.nvir[0]))
    t1b = np.zeros((emb.nocc[1], emb.nvir[1]))
    # Add fragment WFs in intermediate normalization
    for x in emb.get_fragments(mpi_rank=mpi.rank):
        emb.log.debugv("Now adding projected %s-amplitudes of fragment %s", ("L" if get_lambda else "T"), x)
        (roa, rob), (rva, rvb) = x.get_overlap_m2c()
        pwf = x.results.pwf.restore()
        t1xa, t1xb = pwf.l1 if get_lambda else pwf.t1
        if t1xa is None:
            raise NotCalculatedError
        t1a += einsum('ia,Ii,Aa->IA', t1xa, roa, rva)
        t1b += einsum('ia,Ii,Aa->IA', t1xb, rob, rvb)
    # --- MPI
    if mpi:
        t1a, t1b = _mpi_reduce(emb.log, t1a, t1b, mpi_target=mpi_target)
    return (t1a, t1b)

def get_global_t2_uhf(emb, get_lambda=False, symmetrize=True, mpi_target=None):
    """Get global CCSD T2 amplitudes from fragment calculations.

    Parameters
    ----------
    get_lambda: bool, optional
        If True, return L1 amplitudes. Default: False.
    mpi_target: int or None, optional
        If set to an integer, the result will only be available at the specified MPI rank.
        If set to None, an MPI allreduce will be performed and the result will be available
        at all MPI ranks. Default: None.

    Returns
    -------
    t2: tuple(3) of (n(occ), n(occ), n(vir), n(vir)) array
        Global T2 amplitudes.
    """
    t2aa = np.zeros((emb.nocc[0], emb.nocc[0], emb.nvir[0], emb.nvir[0]))
    t2ab = np.zeros((emb.nocc[0], emb.nocc[1], emb.nvir[0], emb.nvir[1]))
    t2bb = np.zeros((emb.nocc[1], emb.nocc[1], emb.nvir[1], emb.nvir[1]))
    # Add fragment WFs in intermediate normalization
    for x in emb.get_fragments(mpi_rank=mpi.rank):
        emb.log.debugv("Now adding projected %s-amplitudes of fragment %s", ("L" if get_lambda else "T"), x)
        (roa, rob), (rva, rvb) = x.get_overlap_m2c()
        pwf = x.results.pwf.restore()
        t2xaa, t2xab, t2xbb = pwf.l2 if get_lambda else pwf.t2
        if t2xaa is None:
            raise NotCalculatedError
        t2aa += einsum('ijab,Ii,Jj,Aa,Bb->IJAB', t2xaa, roa, roa, rva, rva)
        t2ab += einsum('ijab,Ii,Jj,Aa,Bb->IJAB', t2xab, roa, rob, rva, rvb)
        t2bb += einsum('ijab,Ii,Jj,Aa,Bb->IJAB', t2xbb, rob, rob, rvb, rvb)
    # --- MPI
    if mpi:
        t2aa, t2ab, t2bb = _mpi_reduce(emb.log, t2aa, t2ab, t2bb, mpi_target=mpi_target)
    return (t2aa, t2ab, t2bb)
