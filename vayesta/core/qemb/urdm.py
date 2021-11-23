"""Routines to generate reduced density-matrices (RDMs) from spin-unrestricted quantum embedding calculations."""

import numpy as np

import pyscf
import pyscf.cc

from vayesta.core.util import *
from vayesta.core.mpi import mpi
from .rdm import _mpi_reduce


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

def make_rdm1_ccsd_new2(emb, ao_basis=False, t_as_lambda=False, symmetrize=True, with_mf=True, mpi_target=None):
    """Make one-particle reduced density-matrix from partitioned fragment CCSD wave functions.

    MPI parallelized.

    Parameters
    ----------
    ao_basis: bool, optional
        Return the density-matrix in the AO basis. Default: False.
    t_as_lambda: bool, optional
        Use T-amplitudes instead of Lambda-amplitudes for CCSD density matrix. Default: False.

    Returns
    -------
    dm1: (n, n) array
        One-particle reduced density matrix in AO (if `ao_basis=True`) or MO basis (default).
    """

    # --- Fast algorithm via fragment-fragment loop:
    # T1/L1-amplitudes can be summed directly
    t1a, t1b = emb.get_global_t1()
    l1a, l1b = emb.get_global_l1() if not t_as_lambda else (t1a, t1b)

    # --- Loop over pairs of fragments and add projected density-matrix contributions:
    nocca, nvira = t1a.shape
    noccb, nvirb = t1b.shape
    dooa = np.zeros((nocca, nocca))
    doob = np.zeros((noccb, noccb))
    dvva = np.zeros((nvira, nvira))
    dvvb = np.zeros((nvirb, nvirb))
    dova = np.zeros((nocca, nvira))
    dovb = np.zeros((noccb, nvirb))
    # MPI loop

    for frag in emb.get_fragments(mpi_rank=mpi.rank):
        t2xaa, t2xab, t2xba, t2xbb = frag.results.t2x
        l2xaa, l2xab, l2xba, l2xbb = frag.results.l2x if not t_as_lambda else frag.results.t2x
        # Mean-field to cluster (occupied/virtual):
        (coa, cob), (cva, cvb) = frag.get_overlap_m2c()
        # Mean-field to fragment (occupied):
        foa, fob = frag.get_overlap_m2f()[0]
        dooa -= einsum('kiab,kjab,Ii,Jj->IJ', l2xaa, t2xaa, coa, coa) / 2
        doob -= einsum('kiab,kjab,Ii,Jj->IJ', l2xbb, t2xbb, cob, cob) / 2
        dooa -= einsum('ikab,jkab,Ii,Jj->IJ', l2xba, t2xba, coa, coa)
        doob -= einsum('kiab,kjab,Ii,Jj->IJ', l2xab, t2xab, cob, cob)

        if not symmetrize:
            # vv 1)
            dvva += einsum('ijac,ijbc,Aa,Bb->AB', t2xaa, l2xaa, cva, cva) / 2
            dvvb += einsum('ijac,ijbc,Aa,Bb->AB', t2xbb, l2xbb, cvb, cvb) / 2
            # vv 2)
            dvva += einsum('ijac,ijbc,Aa,Bb->AB', t2xab, l2xab, cva, cva)
            dvvb += einsum('ijca,ijcb,Aa,Bb->AB', t2xba, l2xba, cvb, cvb)
            # OR:
            #dvva += einsum('ijac,ijbc,Aa,Bb->AB', t2xba, l2xba, cva, cva)
            #dvvb += einsum('ijca,ijcb,Aa,Bb->AB', t2xab, l2xab, cvb, cvb)
            # ov 1)
            dova += einsum('ijab,Ii,Jj,Aa,Bb,JB->IA', t2xaa, foa, coa, cva, cva, l1a)
            dovb += einsum('ijab,Ii,Jj,Aa,Bb,JB->IA', t2xbb, fob, cob, cvb, cvb, l1b)
            # ov 2)
            dova += einsum('ijab,Ii,Jj,Aa,Bb,JB->IA', t2xab, foa, cob, cva, cvb, l1b)
            dovb += einsum('jiba,Ii,Jj,Aa,Bb,JB->IA', t2xba, fob, coa, cvb, cva, l1a)
        else:
            # vv 1)
            #dvva += einsum('ijac,ijbc,Aa,Bb->AB', t2xaa, l2xaa, cva, cva) / 4
            #dvva += einsum('jica,jicb,Aa,Bb->AB', t2xaa, l2xaa, cva, cva) / 4
            #dvvb += einsum('ijac,ijbc,Aa,Bb->AB', t2xbb, l2xbb, cvb, cvb) / 4
            #dvvb += einsum('jica,jicb,Aa,Bb->AB', t2xbb, l2xbb, cvb, cvb) / 4
            # This symmetrization does not seem to do anything for UHF... Do same as above:
            dvva += einsum('ijac,ijbc,Aa,Bb->AB', t2xaa, l2xaa, cva, cva) / 2
            dvvb += einsum('ijac,ijbc,Aa,Bb->AB', t2xbb, l2xbb, cvb, cvb) / 2
            # vv 2)
            dvva += einsum('ijac,ijbc,Aa,Bb->AB', t2xab, l2xab, cva, cva) / 2
            dvva += einsum('ijac,ijbc,Aa,Bb->AB', t2xba, l2xba, cva, cva) / 2
            dvvb += einsum('ijca,ijcb,Aa,Bb->AB', t2xab, l2xab, cvb, cvb) / 2
            dvvb += einsum('ijca,ijcb,Aa,Bb->AB', t2xba, l2xba, cvb, cvb) / 2
            # ov 1)
            dova += einsum('ijab,Ii,Jj,Aa,Bb,JB->IA', t2xaa, foa, coa, cva, cva, l1a) / 2
            dova += einsum('jiba,Jj,Ii,Aa,Bb,JB->IA', t2xaa, foa, coa, cva, cva, l1a) / 2
            dovb += einsum('ijab,Ii,Jj,Aa,Bb,JB->IA', t2xbb, fob, cob, cvb, cvb, l1b) / 2
            dovb += einsum('jiba,Jj,Ii,Aa,Bb,JB->IA', t2xbb, fob, cob, cvb, cvb, l1b) / 2
            # ov 2)
            dova += einsum('ijab,Ii,Jj,Aa,Bb,JB->IA', t2xab, foa, cob, cva, cvb, l1b) / 2
            dova += einsum('ijab,Jj,Ii,Aa,Bb,JB->IA', t2xba, fob, coa, cva, cvb, l1b) / 2
            dovb += einsum('jiba,Ii,Jj,Aa,Bb,JB->IA', t2xba, fob, coa, cvb, cva, l1a) / 2
            dovb += einsum('jiba,Jj,Ii,Aa,Bb,JB->IA', t2xab, foa, cob, cvb, cva, l1a) / 2

    # MPI reduce here; the remaining terms involve L1/T1 only
    if mpi:
        dooa, doob, dvva, dvvb, dova, dovb = _mpi_reduce(
                emb.log, dooa, doob, dvva, dvvb, dova, dovb, mpi_target=mpi_target)
        if mpi_target not in (None, mpi.rank):
            return None

    # Note: the corresponding dvv-t1 term only gets added later,
    # as the t1*l1 term needs to be added to dvv first
    # Note the + sign as we use dooa/b, rather than xt1a/b (see PySCF)
    dova += einsum('IJ,IA->JA', dooa, t1a)
    dovb += einsum('IJ,IA->JA', doob, t1b)

    dooa -= einsum('IA,JA->IJ', l1a, t1a)
    doob -= einsum('IA,JA->IJ', l1b, t1b)
    dvva += einsum('IA,IB->AB', t1a, l1a)
    dvvb += einsum('IA,IB->AB', t1b, l1b)

    dova -= einsum('IB,AB->IA', t1a, dvva)
    dovb -= einsum('IB,AB->IA', t1b, dvvb)

    dova += (t1a + l1a)
    dovb += (t1b + l1b)

    # Alpha
    occ, vir = np.s_[:nocca], np.s_[nocca:]
    nmo = (nocca + nvira)
    dm1a = np.zeros((nmo, nmo))
    dm1a[occ,occ] = (dooa + dooa.T)
    dm1a[vir,vir] = (dvva + dvva.T)
    dm1a[occ,vir] = dova
    dm1a[vir,occ] = dova.T
    dm1a /= 2.0
    # Beta
    occ, vir = np.s_[:noccb], np.s_[noccb:]
    nmo = (noccb + nvirb)
    dm1b = np.zeros((nmo, nmo))
    dm1b[occ,occ] = (doob + doob.T)
    dm1b[vir,vir] = (dvvb + dvvb.T)
    dm1b[occ,vir] = dovb
    dm1b[vir,occ] = dovb.T
    dm1b /= 2.0

    if with_mf:
        dm1a[np.diag_indices(nocca)] += 1.0
        dm1b[np.diag_indices(noccb)] += 1.0
    if ao_basis:
        dm1a = dot(emb.mo_coeff[0], dm1a, emb.mo_coeff[0].T)
        dm1b = dot(emb.mo_coeff[1], dm1b, emb.mo_coeff[1].T)

    return (dm1a, dm1b)
