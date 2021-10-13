"""Routines to generate reduced density-matrices (RDMs) from spin-unrestricted quantum embedding calculations."""

import numpy as np

import pyscf
import pyscf.cc

from vayesta.core.util import *


def make_rdm1_demo(emb, ao_basis=False, add_mf=False, symmetrize=True):
    """Make democratically partitioned one-particle reduced density-matrix from fragment calculations.

    Warning: A democratically partitioned DM is only expected to yield reasonable results
    for full fragmentations (eg, Lowdin-AO or IAO+PAO fragmentation).

    Parameters
    ----------
    ao_basis: bool, optional
        Return the density-matrix in the AO basis. Default: False.
    add_mf: bool, optional
        Add the mean-field contribution to the density-matrix (double counting is accounted for).
        Is only used if `partition = 'dm'`. Default: False.
    symmetrize: bool, optional
        Symmetrize the density-matrix at the end of the calculation. Default: True.

    Returns
    -------
    dm1: tuple of (n, n) arrays
        Alpha- and beta one-particle reduced density matrix in AO (if `ao_basis=True`) or MO basis (default).
    """
    ovlp = emb.get_ovlp()
    mo_coeff = emb.mo_coeff
    if add_mf:
        sca = np.dot(ovlp, mo_coeff[0])
        scb = np.dot(ovlp, mo_coeff[1])
        dm1a_mf, dm1b_mf = emb.mf.make_rdm1()
        dm1a_mf = dot(sca.T, dm1a_mf, sca)
        dm1b_mf = dot(scb.T, dm1b_mf, scb)
        dm1a = dm1a_mf.copy()
        dm1b = dm1b_mf.copy()
    else:
        dm1a = np.zeros((emb.nmo[0], emb.nmo[0]))
        dm1b = np.zeros((emb.nmo[1], emb.nmo[1]))
    for f in emb.fragments:
        emb.log.debugv("Now adding projected DM of fragment %s", f)
        if f.results.dm1 is None:
            raise RuntimeError("DM1 not calculated for fragment %s!" % f)
        if emb.opts.dm_with_frozen:
            cf = f.mo_coeff
        else:
            cf = f.c_active
        rfa = dot(mo_coeff[0].T, ovlp, cf[0])
        rfb = dot(mo_coeff[1].T, ovlp, cf[1])
        if add_mf:
            # Subtract double counting:
            ddma = (f.results.dm1[0] - dot(rfa.T, dm1a_mf, rfa))
            ddmb = (f.results.dm1[1] - dot(rfb.T, dm1b_mf, rfb))
        else:
            ddma, ddmb = f.results.dm1
        pfa, pfb = f.get_fragment_projector(cf)
        dm1a += einsum('xi,ij,px,qj->pq', pfa, ddma, rfa, rfa)
        dm1b += einsum('xi,ij,px,qj->pq', pfb, ddmb, rfb, rfb)
    if ao_basis:
        dm1a = dot(mo_coeff[0], dm1a, mo_coeff[0].T)
        dm1b = dot(mo_coeff[1], dm1b, mo_coeff[1].T)
    if symmetrize:
        dm1a = (dm1a + dm1a.T)/2
        dm1b = (dm1b + dm1b.T)/2
    return (dm1a, dm1b)
