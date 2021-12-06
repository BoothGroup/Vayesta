"""Democratically partitioned RDMs"""

import numpy as np

from vayesta.core.util import *


def make_rdm1_demo_rhf(emb, ao_basis=False, add_mf=False, symmetrize=True):
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
    dm1: (n, n) array
        One-particle reduced density matrix in AO (if `ao_basis=True`) or MO basis (default).
    """
    ovlp = emb.get_ovlp()
    mo_coeff = emb.mo_coeff
    if add_mf:
        sc = np.dot(ovlp, mo_coeff)
        dm1_mf = emb.mf.make_rdm1()
        dm1_mf = dot(sc.T, dm1_mf, sc)
        dm1 = dm1_mf.copy()
    else:
        dm1 = np.zeros((emb.nmo, emb.nmo))
    for f in emb.fragments:
        emb.log.debugv("Now adding projected DM of fragment %s", f)
        if f.results.dm1 is None:
            raise RuntimeError("DM1 not calculated for fragment %s!" % f)
        if emb.opts.dm_with_frozen:
            cf = f.mo_coeff
        else:
            cf = f.cluster.c_active
        rf = dot(mo_coeff.T, ovlp, cf)
        if add_mf:
            # Subtract double counting:
            ddm = (f.results.dm1 - dot(rf.T, dm1_mf, rf))
        else:
            ddm = f.results.dm1
        pf = f.get_fragment_projector(cf)
        dm1 += einsum('xi,ij,px,qj->pq', pf, ddm, rf, rf)
    if ao_basis:
        dm1 = dot(mo_coeff, dm1, mo_coeff.T)
    if symmetrize:
        dm1 = (dm1 + dm1.T)/2
    return dm1

def make_rdm1_demo_uhf(emb, ao_basis=False, add_mf=False, symmetrize=True):
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

# --- Two-particle
# ----------------

def make_rdm2_demo_rhf(emb, ao_basis=False, add_mf=True, symmetrize=True):
    """Make democratically partitioned two-particle reduced density-matrix from fragment calculations.

    Warning: A democratically partitioned DM is only expected to yield reasonable results
    for full fragmentations (eg, Lowdin-AO or IAO+PAO fragmentation).

    Parameters
    ----------
    ao_basis: bool, optional
        Return the density-matrix in the AO basis. Default: False.
    add_mf: bool, optional
        Add the mean-field contribution to the density-matrix (double counting is accounted for).
        Is only used if `partition = 'dm'`. Default: True.
    symmetrize: bool, optional
        Symmetrize the density-matrix at the end of the calculation. Default: True.

    Returns
    -------
    dm2: (n, n, n, n) array
        Two-particle reduced density matrix in AO (if `ao_basis=True`) or MO basis (default).
    """
    if add_mf:
        #dm2_mf = np.zeros(4*[emb.nmo])
        #for i in range(emb.nocc):
        #    for j in range(emb.nocc):
        #        dm2_mf[i,i,j,j] += 4.0
        #        dm2_mf[i,j,j,i] -= 2.0
        sc = np.dot(emb.get_ovlp(), emb.mo_coeff)
        dm1_mf = np.linalg.multi_dot((sc.T, emb.mf.make_rdm1(), sc))
        dm2_mf = einsum('ij,kl->ijkl', dm1_mf, dm1_mf) - einsum('il,jk->ijkl', dm1_mf, dm1_mf)/2
        dm2 = dm2_mf.copy()
    else:
        dm2 = np.zeros((emb.nmo, emb.nmo, emb.nmo, emb.nmo))

    for f in emb.fragments:
        if f.results.dm2 is None:
            raise RuntimeError("DM2 not calculated for fragment %s!" % f)
        if emb.opts.dm_with_frozen:
            cf = f.mo_coeff
        else:
            cf = f.cluster.c_active
        rf = np.linalg.multi_dot((emb.mo_coeff.T, emb.get_ovlp(), cf))
        if add_mf:
            # Subtract double counting:
            ddm = (f.results.dm2 - einsum('IJKL,Ii,Jj,Kk,Ll->ijkl', dm2_mf, rf, rf, rf, rf))
        else:
            ddm = f.results.dm2
        pf = f.get_fragment_projector(cf)
        dm2 += einsum('xi,ijkl,px,qj,rk,sl->pqrs', pf, ddm, rf, rf, rf, rf)
    if ao_basis:
        dm2 = einsum('ijkl,pi,qj,rk,sl->pqrs', dm2, *(4*[emb.mo_coeff]))
    if symmetrize:
        dm2 = (dm2 + dm2.transpose(1,0,3,2))/2
    return dm2


