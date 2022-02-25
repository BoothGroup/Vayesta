"""Routines to generate reduced density-matrices (RDMs) from spin-unrestricted quantum embedding calculations."""

import numpy as np

import pyscf
import pyscf.cc

from vayesta.core.util import *
from vayesta.core.mpi import mpi
from .rdm import _mpi_reduce


def make_rdm1_ccsd(emb, ao_basis=False, t_as_lambda=False, symmetrize=True, with_mf=True, mpi_target=None, mp2=False):
    """Make one-particle reduced density-matrix from partitioned fragment CCSD wave functions.

    MPI parallelized.

    Parameters
    ----------
    ao_basis: bool, optional
        Return the density-matrix in the AO basis. Default: False.
    t_as_lambda: bool, optional
        Use T-amplitudes instead of Lambda-amplitudes for CCSD density matrix. Default: False.
    symmetrize: bool, optional
        Use Symemtrized equations, if possible. Default: True.
    with_mf: bool, optional
        If False, only the difference to the mean-field density-matrix is returned. Default: True.
    mpi_target: integer, optional
        If set to an integer, the density-matrix will only be constructed on the corresponding MPI rank.
        Default: None.

    Returns
    -------
    dm1: tuple(2) of (n, n) arrays
        (alpha, beta) one-particle reduced density matrix in AO (if `ao_basis=True`) or MO basis (default).
    """

    nocca, noccb = emb.nocc
    nvira, nvirb = emb.nvir
    # --- Fast algorithm via fragment-fragment loop:
    # T1/L1-amplitudes can be summed directly
    if mp2:
        t_as_lambda = True
    else:
        t1a, t1b = emb.get_global_t1()
        l1a, l1b = emb.get_global_l1() if not t_as_lambda else (t1a, t1b)

    # --- Loop over pairs of fragments and add projected density-matrix contributions:
    dooa = np.zeros((nocca, nocca))
    doob = np.zeros((noccb, noccb))
    dvva = np.zeros((nvira, nvira))
    dvvb = np.zeros((nvirb, nvirb))
    if not mp2:
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

        # D(occ,occ) and D(vir,vir)
        # aa/bb -> dooa/doob
        dooa -= einsum('kiab,kjab,Ii,Jj->IJ', l2xaa, t2xaa, coa, coa) / 2
        doob -= einsum('kiab,kjab,Ii,Jj->IJ', l2xbb, t2xbb, cob, cob) / 2
        # ba/ab -> dooa/doob
        dooa -= einsum('ikab,jkab,Ii,Jj->IJ', l2xba, t2xba, coa, coa)
        doob -= einsum('kiab,kjab,Ii,Jj->IJ', l2xab, t2xab, cob, cob)
        # aa/bb - > dvva/dvvb
        dvva += einsum('ijac,ijbc,Aa,Bb->AB', t2xaa, l2xaa, cva, cva) / 2
        dvvb += einsum('ijac,ijbc,Aa,Bb->AB', t2xbb, l2xbb, cvb, cvb) / 2
        # We cannot symmetrize the ba/ab -> dooa/doob part between t/l2xab and t/l2xba (and keep a single fragment loop);
        # instead dooa only uses the "ba" amplitudes and doob only the "ab" amplitudes.
        # In order to ensure the correct number of alpha and beta electrons, we should use the same choice for the
        # virtual-virtual parts (even though here we could symmetrize them as:
        #dvva += einsum('ijac,ijbc,Aa,Bb->AB', t2xba, l2xba, cva, cva) / 2
        #dvva += einsum('ijac,ijbc,Aa,Bb->AB', t2xab, l2xab, cva, cva) / 2
        #dvvb += einsum('ijca,ijcb,Aa,Bb->AB', t2xab, l2xab, cvb, cvb) / 2
        #dvvb += einsum('ijca,ijcb,Aa,Bb->AB', t2xba, l2xba, cvb, cvb) / 2
        # ba/ab -> dooa/doob
        dvva += einsum('ijac,ijbc,Aa,Bb->AB', t2xba, l2xba, cva, cva)
        dvvb += einsum('ijca,ijcb,Aa,Bb->AB', t2xab, l2xab, cvb, cvb)

        # D(occ,vir)
        if not mp2:
            if not symmetrize:
                # aa/bb
                dova += einsum('ijab,Ii,Jj,Aa,Bb,JB->IA', t2xaa, foa, coa, cva, cva, l1a)
                dovb += einsum('ijab,Ii,Jj,Aa,Bb,JB->IA', t2xbb, fob, cob, cvb, cvb, l1b)
                # ab/ba
                dova += einsum('ijab,Ii,Jj,Aa,Bb,JB->IA', t2xab, foa, cob, cva, cvb, l1b)
                dovb += einsum('jiba,Ii,Jj,Aa,Bb,JB->IA', t2xba, fob, coa, cvb, cva, l1a)
            else:
                # aa/bb
                dova += einsum('ijab,Ii,Jj,Aa,Bb,JB->IA', t2xaa, foa, coa, cva, cva, l1a) / 2
                dova += einsum('jiba,Jj,Ii,Aa,Bb,JB->IA', t2xaa, foa, coa, cva, cva, l1a) / 2
                dovb += einsum('ijab,Ii,Jj,Aa,Bb,JB->IA', t2xbb, fob, cob, cvb, cvb, l1b) / 2
                dovb += einsum('jiba,Jj,Ii,Aa,Bb,JB->IA', t2xbb, fob, cob, cvb, cvb, l1b) / 2
                # ab/baAA (here we can use t2xab and t2xba in a symmetric fashion:
                dova += einsum('ijab,Ii,Jj,Aa,Bb,JB->IA', t2xab, foa, cob, cva, cvb, l1b) / 2
                dova += einsum('ijab,Jj,Ii,Aa,Bb,JB->IA', t2xba, fob, coa, cva, cvb, l1b) / 2
                dovb += einsum('jiba,Ii,Jj,Aa,Bb,JB->IA', t2xba, fob, coa, cvb, cva, l1a) / 2
                dovb += einsum('jiba,Jj,Ii,Aa,Bb,JB->IA', t2xab, foa, cob, cvb, cva, l1a) / 2

    # MPI reduce here; the remaining terms involve L1/T1 only
    if mpi:
        if mp2:
            dooa, doob, dvva, dvvb = _mpi_reduce(
                    emb.log, dooa, doob, dvva, dvvb, mpi_target=mpi_target)
        else:
            dooa, doob, dvva, dvvb, dova, dovb = _mpi_reduce(
                    emb.log, dooa, doob, dvva, dvvb, dova, dovb, mpi_target=mpi_target)
        if mpi_target not in (None, mpi.rank):
            return None

    # Note: the corresponding dvv-t1 term only gets added later,
    # as the t1*l1 term needs to be added to dvv first
    # Note the + sign as we use dooa/b, rather than xt1a/b (see PySCF)
    if not mp2:
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
    if not mp2:
        dm1a[occ,vir] = dova
        dm1a[vir,occ] = dova.T
    dm1a /= 2.0
    # Beta
    occ, vir = np.s_[:noccb], np.s_[noccb:]
    nmo = (noccb + nvirb)
    dm1b = np.zeros((nmo, nmo))
    dm1b[occ,occ] = (doob + doob.T)
    dm1b[vir,vir] = (dvvb + dvvb.T)
    if not mp2:
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
