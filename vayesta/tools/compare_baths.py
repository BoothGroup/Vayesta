import numpy as np
from vayesta.core.util import dot
import scipy.linalg


def get_incremental_overlap(ca, cb, oa=None, ob=None, s=None, incre_tol=1e-2):
    """For two sets of coefficients, compute the overlap between the spaces spanned as the number of coefficients,
    when truncated at different points.
    parameters:
        ca,cb: np.array
            Coefficients for baths A and B, where c[:, i] corresponds to the ith bath orbital in the ranking.
        oa, ob: np.array
            Ranking metric for baths; used to identify degenerate bath orbitals. If None (default) then all orbitals
            are assumed to be non-degenerate.
        s: np.array
            Overlap matrix for bath orbitals. If None (default) then the overlap is assumed to be the identity.
        incre_tol: float
            Tolerance for identifying degenerate orbitals. If two orbitals have a ratio of 1 +- incre_tol, then they
            are assumed to be degenerate. Default: 1e-2
    returns:
        overlap: np.array
            Overlap between the spaces spanned by the first i bath orbitals, for i in increments.
        increments: np.array
            Number of bath orbitals included in each overlap calculation, ie. the numbers of bath orbitals which don't
            split any degenerate sets.
    """

    if s is not None:
        srt = scipy.linalg.sqrtm(s)
        ca = dot(srt, ca)
        cb = dot(srt.T, cb)

    # Normalise
    ca = ca / np.sqrt(dot(ca.T, ca).trace()/ca.shape[1])
    cb = cb / np.sqrt(dot(cb.T, cb).trace()/cb.shape[1])

    if oa is not None and ob is not None:
        increments = get_pair_increments(oa, ob, tol=incre_tol)
    else:
        increments = range(1, ca.shape[1]+1)


    overlap = np.asarray([compute_overlap(ca[:, :i], cb[:, :i]) for i in increments])
    return overlap, increments


def compute_overlap(ca, cb, s=None):
    # Could compute as dot(ca.T, cb).trace()
    # But more efficient to instead compute as
    def get_ovlp(ca, cb):
        if s is not None:
            return dot(ca.T, s, cb)
        return dot(ca.T, cb)
    sab = get_ovlp(ca, cb)
    return dot(sab.T, sab).trace() / np.sqrt(get_ovlp(ca, ca).trace() * get_ovlp(cb, cb).trace())


def get_pair_increments(oa, ob, tol=1e-3):
    inca = set(get_increments(oa, tol=tol))
    incb = set(get_increments(ob, tol=tol))
    comb = inca.intersection(incb)
    return sorted(list(comb))


def get_increments(vals, tol=1e-3):
    preval = None
    breaks = []
    for i, v in enumerate(vals):
        if preval is not None:
            if abs(1 - (v/preval)) > tol:
                breaks.append(i)
        preval = v
    breaks.append(len(vals))
    return breaks
