import functools

import numpy as np

from vayesta.core.util import *
from vayesta.core.mpi import mpi


def get_global_t1_rhf(emb, get_lambda=False, mpi_target=None, ao_basis=False):
    """Get global CCSD T1 amplitudes from fragment calculations.

    Runtime: N(frag)/N(MPI) * N^2

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
        pwf = x.results.pwf.restore().as_ccsd()
        t1x = pwf.l1 if get_lambda else pwf.t1
        if t1x is None:
            raise NotCalculatedError
        t1 += dot(ro, t1x, rv.T)
    # --- MPI
    if mpi:
        t1 = mpi.nreduce(t1, target=mpi_target, logfunc=emb.log.timingv)
    if ao_basis:
        t1 = dot(emb.mo_coeff_occ, t1, emb.mo_coeff_vir.T)
    return t1

def get_global_t2_rhf(emb, get_lambda=False, symmetrize=True, mpi_target=None, ao_basis=False):
    """Get global CCSD T2 amplitudes from fragment calculations.

    Runtime: N(frag)/N(MPI) * N^4

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
        pwf = x.results.pwf.restore().as_ccsd()
        t2x = pwf.l2 if get_lambda else pwf.t2
        if t2x is None:
            raise NotCalculatedError
        t2 += einsum('ijab,Ii,Jj,Aa,Bb->IJAB', t2x, ro, ro, rv, rv)
    # --- MPI
    if mpi:
        t2 = mpi.nreduce(t2, target=mpi_target, logfunc=emb.log.timingv)
    if ao_basis:
        t2 = einsum('Ii,Jj,ijab,Aa,Bb->IJAB', emb.mo_coeff_occ, emb.mo_coeff_occ, t2, emb.mo_coeff_vir, emb.mo_coeff_vir)
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
        pwf = x.results.pwf.restore().as_ccsd()
        t1xa, t1xb = pwf.l1 if get_lambda else pwf.t1
        if t1xa is None:
            raise NotCalculatedError
        t1a += einsum('ia,Ii,Aa->IA', t1xa, roa, rva)
        t1b += einsum('ia,Ii,Aa->IA', t1xb, rob, rvb)
    # --- MPI
    if mpi:
        t1a, t1b = mpi.nreduce(t1a, t1b, target=mpi_target, logfunc=emb.log.timingv)
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
        pwf = x.results.pwf.restore().as_ccsd()
        t2xaa, t2xab, t2xbb = pwf.l2 if get_lambda else pwf.t2
        if t2xaa is None:
            raise NotCalculatedError
        t2aa += einsum('ijab,Ii,Jj,Aa,Bb->IJAB', t2xaa, roa, roa, rva, rva)
        t2ab += einsum('ijab,Ii,Jj,Aa,Bb->IJAB', t2xab, roa, rob, rva, rvb)
        t2bb += einsum('ijab,Ii,Jj,Aa,Bb->IJAB', t2xbb, rob, rob, rvb, rvb)
    # --- MPI
    if mpi:
        t2aa, t2ab, t2bb = mpi.nreduce(t2aa, t2ab, t2bb, target=mpi_target, logfunc=emb.log.timingv)
    return (t2aa, t2ab, t2bb)
