import numpy as np

from vayesta.core.util import *


def get_t1(emb, get_lambda=False, partition=None):
    return get_t12(emb, calc_t2=False, get_lambda=get_lambda, partition=partition)[0]

def get_t2(emb, get_lambda=False, partition=None, symmetrize=True):
    return get_t12(emb, calc_t1=False, get_lambda=get_lambda, partition=partition, symmetrize=symmetrize)[1]

def get_t1_uhf(emb, get_lambda=False, partition=None):
    return get_t12_uhf(emb, calc_t2=False, get_lambda=get_lambda, partition=partition)[0]

def get_t2_uhf(emb, get_lambda=False, partition=None, symmetrize=True):
    return get_t12_uhf(emb, calc_t1=False, get_lambda=get_lambda, partition=partition, symmetrize=symmetrize)[1]

def get_t12(emb, calc_t1=True, calc_t2=True, get_lambda=False, partition=None, symmetrize=True):
    """Get global CCSD wave function (T1 and T2 amplitudes) from fragment calculations.

    Parameters
    ----------
    partition: ['first-occ', 'first-vir', 'democratic']
        Partitioning scheme of the T amplitudes. Default: 'first-occ'.

    Returns
    -------
    t1: (n(occ), n(vir)) array
        Global T1 amplitudes.
    t2: (n(occ), n(occ), n(vir), n(vir)) array
        Global T2 amplitudes.
    """
    if partition is None: partition = emb.opts.wf_partition
    t1 = np.zeros((emb.nocc, emb.nvir)) if calc_t1 else None
    t2 = np.zeros((emb.nocc, emb.nocc, emb.nvir, emb.nvir)) if calc_t2 else None
    # Add fragment WFs in intermediate normalization
    for f in emb.fragments:
        emb.log.debugv("Now adding projected %s-amplitudes of fragment %s", ("L" if get_lambda else "T"), f)
        ro, rv = f.get_rot_to_mf()
        if calc_t1:
            t1f = (f.results.l1 if get_lambda else f.results.get_t1())
            if t1f is None: raise RuntimeError("Amplitudes not found for %s" % f)
            t1f = f.project_amplitude_to_fragment(t1f, partition=partition)
            t1 += einsum('ia,iI,aA->IA', t1f, ro, rv)
        if calc_t2:
            t2f = (f.results.l2 if get_lambda else f.results.get_t2())
            if t2f is None: raise RuntimeError("Amplitudes not found for %s" % f)
            t2f = f.project_amplitude_to_fragment(t2f, partition=partition, symmetrize=symmetrize)
            t2 += einsum('ijab,iI,jJ,aA,bB->IJAB', t2f, ro, ro, rv, rv)
    #t2 = (t2 + t2.transpose(1,0,3,2))/2
    #assert np.allclose(t2, t2.transpose(1,0,3,2))
    return t1, t2

def get_t12_uhf(emb, calc_t1=True, calc_t2=True, get_lambda=False, partition=None, symmetrize=True):
    """Get global CCSD wave function (T1 and T2 amplitudes) from fragment calculations.

    Parameters
    ----------
    partition: ['first-occ', 'first-vir', 'democratic']
        Partitioning scheme of the T amplitudes. Default: 'first-occ'.

    Returns
    -------
    t1: tuple(2) of (n(occ), n(vir)) array
        Global T1 amplitudes.
    t2: tuple(3) of (n(occ), n(occ), n(vir), n(vir)) array
        Global T2 amplitudes.
    """
    if partition is None: partition = emb.opts.wf_partition
    t1a = np.zeros((emb.nocc[0], emb.nvir[0])) if calc_t1 else None
    t1b = np.zeros((emb.nocc[1], emb.nvir[1])) if calc_t1 else None
    t2aa = np.zeros((emb.nocc[0], emb.nocc[0], emb.nvir[0], emb.nvir[0])) if calc_t2 else None
    t2ab = np.zeros((emb.nocc[0], emb.nocc[1], emb.nvir[0], emb.nvir[1])) if calc_t2 else None
    t2bb = np.zeros((emb.nocc[1], emb.nocc[1], emb.nvir[1], emb.nvir[1])) if calc_t2 else None
    # Add fragment WFs in intermediate normalization
    for f in emb.fragments:
        emb.log.debugv("Now adding projected %s-amplitudes of fragment %s", ("L" if get_lambda else "T"), f)
        (roa, rob), (rva, rvb) = f.get_rot_to_mf()
        if calc_t1:
            t1f = (f.results.l1 if get_lambda else f.results.get_t1())
            if t1f is None: raise RuntimeError("Amplitudes not found for %s" % f)
            t1fa, t1fb = f.project_amplitude_to_fragment(t1f, partition=partition)
            t1a += einsum('ia,iI,aA->IA', t1fa, roa, rva)
            t1b += einsum('ia,iI,aA->IA', t1fb, rob, rvb)
        if calc_t2:
            t2f = (f.results.l2 if get_lambda else f.results.get_t2())
            if t2f is None: raise RuntimeError("Amplitudes not found for %s" % f)
            t2faa, t2fab, t2fbb = f.project_amplitude_to_fragment(t2f, partition=partition, symmetrize=symmetrize)
            t2aa += einsum('ijab,iI,jJ,aA,bB->IJAB', t2faa, roa, roa, rva, rva)
            t2ab += einsum('ijab,iI,jJ,aA,bB->IJAB', t2fab, roa, rob, rva, rvb)
            t2bb += einsum('ijab,iI,jJ,aA,bB->IJAB', t2fbb, rob, rob, rvb, rvb)
    #t2 = (t2 + t2.transpose(1,0,3,2))/2
    #assert np.allclose(t2, t2.transpose(1,0,3,2))
    return (t1a, t1b), (t2aa, t2ab, t2bb)
