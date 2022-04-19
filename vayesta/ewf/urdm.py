"""Routines to generate reduced density-matrices (RDMs) from spin-unrestricted quantum embedding calculations."""

import numpy as np

import pyscf
import pyscf.cc

from vayesta.core.util import *
from vayesta.core.mpi import mpi


def make_rdm1_ccsd(emb, ao_basis=False, t_as_lambda=False, symmetrize=True, with_mf=True, mpi_target=None, mp2=False, ba_order='ba'):
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
        if ba_order == 'ab':
            t2xba = t2xba.transpose(1,0,3,2)
            l2xba = l2xba.transpose(1,0,3,2)
        # Mean-field to cluster (occupied/virtual):
        (coa, cob), (cva, cvb) = frag.get_overlap_m2c()
        # Mean-field to fragment (occupied):
        foa, fob = frag.get_overlap_m2f()[0]

        # D(occ,occ) and D(vir,vir)
        # aa/bb -> dooa/doob
        dooa -= einsum('kiab,kjab,Ii,Jj->IJ', l2xaa, t2xaa, coa, coa) / 2
        doob -= einsum('kiab,kjab,Ii,Jj->IJ', l2xbb, t2xbb, cob, cob) / 2
        # ba/ab -> dooa/doob
        dooa -= einsum('kiab,kjab,Ii,Jj->IJ', l2xba, t2xba, coa, coa)
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
        dvva += einsum('ijca,ijcb,Aa,Bb->AB', t2xba, l2xba, cva, cva)
        dvvb += einsum('ijca,ijcb,Aa,Bb->AB', t2xab, l2xab, cvb, cvb)

        # D(occ,vir)
        if not mp2:
            if not symmetrize:
                # aa/bb
                dova += einsum('ijab,Ii,Jj,Aa,Bb,JB->IA', t2xaa, foa, coa, cva, cva, l1a)
                dovb += einsum('ijab,Ii,Jj,Aa,Bb,JB->IA', t2xbb, fob, cob, cvb, cvb, l1b)
                # ab/ba
                dova += einsum('ijab,Ii,Jj,Aa,Bb,JB->IA', t2xab, foa, cob, cva, cvb, l1b)
                dovb += einsum('ijab,Ii,Jj,Aa,Bb,JB->IA', t2xba, fob, coa, cvb, cva, l1a)
            else:
                # aa/bb
                dova += einsum('ijab,Ii,Jj,Aa,Bb,JB->IA', t2xaa, foa, coa, cva, cva, l1a) / 2
                dova += einsum('jiba,Jj,Ii,Aa,Bb,JB->IA', t2xaa, foa, coa, cva, cva, l1a) / 2
                dovb += einsum('ijab,Ii,Jj,Aa,Bb,JB->IA', t2xbb, fob, cob, cvb, cvb, l1b) / 2
                dovb += einsum('jiba,Jj,Ii,Aa,Bb,JB->IA', t2xbb, fob, cob, cvb, cvb, l1b) / 2
                # ab/baAA (here we can use t2xab and t2xba in a symmetric fashion:
                dova += einsum('ijab,Ii,Jj,Aa,Bb,JB->IA', t2xab, foa, cob, cva, cvb, l1b) / 2
                dova += einsum('jiba,Jj,Ii,Aa,Bb,JB->IA', t2xba, fob, coa, cva, cvb, l1b) / 2
                dovb += einsum('ijab,Ii,Jj,Aa,Bb,JB->IA', t2xba, fob, coa, cvb, cva, l1a) / 2
                dovb += einsum('jiba,Jj,Ii,Aa,Bb,JB->IA', t2xab, foa, cob, cvb, cva, l1a) / 2

    # MPI reduce here; the remaining terms involve L1/T1 only
    if mpi:
        if mp2:
            dooa, doob, dvva, dvvb = mpi.nreduce(
                    dooa, doob, dvva, dvvb, target=mpi_target, logfunc=emb.log.timingv)
        else:
            dooa, doob, dova, dovb, dvva, dvvb = mpi.nreduce(
                    dooa, doob, dova, dovb, dvva, dvvb, target=mpi_target, logfunc=emb.log.timingv)
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


def make_rdm1_ccsd_global_wf(emb, ao_basis=False, t_as_lambda=False, slow=True,
        with_mf=True, ovlp_tol=1e-10, symmetrize=True, with_t1=True):
    """Make one-particle reduced density-matrix from partitioned fragment CCSD wave functions.

    NOT MPI READY

    Parameters
    ----------
    ao_basis: bool, optional
        Return the density-matrix in the AO basis. Default: False.
    t_as_lambda: bool, optional
        Use T-amplitudes instead of Lambda-amplitudes for CCSD density matrix. Default: False.
    slow: bool, optional
        Combine to global CCSD wave function first, then build density matrix.
        Equivalent, but does not scale well. Default: False

    Returns
    -------
    dm1: (n, n) array
        One-particle reduced density matrix in AO (if `ao_basis=True`) or MO basis (default).
    """

    def finalize(dm1):
        dm1a, dm1b = dm1
        if ao_basis:
            dm1a = dot(emb.mo_coeff[0], dm1a, emb.mo_coeff[0].T)
            dm1b = dot(emb.mo_coeff[1], dm1b, emb.mo_coeff[1].T)
        dm1a = (dm1a + dm1a.T)/2
        dm1b = (dm1b + dm1b.T)/2
        return (dm1a, dm1b)

    fragments = emb.fragments

    # --- Slow N^5 algorithm:
    if slow:
        t1 = emb.get_global_t1()
        t2 = emb.get_global_t2()
        fakecc = Object()
        fakecc.mo_coeff = emb.mo_coeff
        if t_as_lambda:
            l1, l2 = t1, t2
        else:
            l1 = emb.get_global_l1()
            l2 = emb.get_global_l2()
        if not with_t1:
            t1 = l1 = (np.zeros_like(t1[0]), np.zeros_like(t1[1]))
        dm1 = pyscf.cc.uccsd_rdm.make_rdm1(fakecc, t1=t1, t2=t2, l1=l1, l2=l2,
                with_frozen=False, with_mf=with_mf)
        return finalize(dm1)

    raise NotImplementedError

    ## --- Fast algorithm via fragment-fragment loop:
    ## T1/L1-amplitudes can be summed directly
    #if with_t1:
    #    t1 = emb.get_global_t1()
    #    l1 = (t1 if t_as_lambda else emb.get_global_l1())
    #else:
    #    raise NotImplementedError

    ## --- Preconstruct some C^T.S.C rotation matrices:
    #f2fo, f2fv = emb.get_overlap_c2c()
    #f2mo, f2mv = emb.get_overlap_m2c()

    ## --- Loop over pairs of fragments and add projected density-matrix contributions:
    #nocca, nvira = t1[0].shape
    #noccb, nvirb = t1[1].shape
    #dooa = np.zeros((nocca, nocca))
    #doob = np.zeros((noccb, noccb))
    #dvva = np.zeros((nvira, nvira))
    #dvvb = np.zeros((nvirb, nvirb))
    #if with_t1:
    #    #dov = (t1 + l1 - einsum('ie,me,ma->ia', t1, l1, t1))
    #    dova = np.zeros((nocca, nvira))
    #    dovb = np.zeros((noccb, nvirb))

    #for x in emb.get_fragments(mpi_rank=mpi.rank):

    #    wfx = x.results.pwf.restore()
    #    t2xaa, t2xab, t2xbb = wfx.t2

    #    dooax = np.zeros((, noccb

    #    for y in emb.get_fragments(mpi_rank=mpi.rank):
    #    
    #        wfy = y.results.pwf.restore()
    #        l2xaa, l2xab, l2xbb = wfy.l2




    #    theta = (2*theta - theta.transpose(0,1,3,2))
    #    theta = f1.project_amplitude_to_fragment(theta, symmetrize=symmetrize)
    #    # Intermediates - leave left index in cluster basis:
    #    doo_f1 = np.zeros((f1.cluster.nocc_active, nocc))
    #    dvv_f1 = np.zeros((f1.cluster.nvir_active, nvir))
    #    if with_t1:
    #        dov += einsum('imae,Pi,Mm,Qa,Ee,ME->PQ', theta, f2mo[i1], f2mo[i1], f2mv[i1], f2mv[i1], l1)
    #    for i2, f2 in enumerate(fragments):
    #        if i1 >= i2:
    #            f2fo12 = f2fo[i1][i2]
    #            f2fv12 = f2fv[i1][i2]
    #        else:
    #            f2fo12 = f2fo[i2][i1].T
    #            f2fv12 = f2fv[i2][i1].T
    #        if min(abs(f2fo12).max(), abs(f2fv12).max()) < ovlp_tol:
    #            emb.log.debugv("Overlap between %s and %s below %.2e; skipping.", f1, f2, ovlp_tol)
    #            continue
    #        l2 = (f2.results.get_t2() if t_as_lambda else f2.results.l2)
    #        if l2 is None:
    #            raise RuntimeError("No L2 amplitudes found for %s!" % f2)
    #        l2 = f2.project_amplitude_to_fragment(l2, symmetrize=symmetrize)
    #        # Theta_jk^ab * l_ik^ab -> ij
    #        doo_f1 -= einsum('jkab,IKAB,kK,aA,bB,qI->jq', theta, l2, f2fo12, f2fv12, f2fv12, f2mo[i2])
    #        # Theta_ji^ca * l_ji^cb -> ab
    #        dvv_f1 += einsum('jica,JICB,jJ,iI,cC,qB->aq', theta, l2, f2fo12, f2fo12, f2fv12, f2mv[i2])
    #    doo += np.dot(f2mo[i1], doo_f1)
    #    dvv += np.dot(f2mv[i1], dvv_f1)

    #if with_t1:
    #    dov += einsum('im,ma->ia', doo, t1)
    #    dov -= einsum('ie,ae->ia', t1, dvv)
    #    doo -= einsum('ja,ia->ij', t1, l1)
    #    dvv += einsum('ia,ib->ab', t1, l1)

    #nmo = (nocc + nvir)
    #occ, vir = np.s_[:nocc], np.s_[nocc:]
    #dm1 = np.zeros((nmo, nmo))
    #dm1[occ,occ] = (doo + doo.T)
    #dm1[vir,vir] = (dvv + dvv.T)
    #if with_t1:
    #    dm1[occ,vir] = dov
    #    dm1[vir,occ] = dov.T
    #if with_mf:
    #    dm1[np.diag_indices(nocc)] += 2

    #return finalize(dm1)

def make_rdm2_ccsd_proj_lambda(emb, ao_basis=False, t_as_lambda=False, mpi_target=None):
    """Make two-particle reduced density-matrix from partitioned fragment CCSD wave functions.

    Without 1DM!

    MPI parallelized.

    Parameters
    ----------
    ao_basis: bool, optional
        Return the density-matrix in the AO basis. Default: False.
    t_as_lambda: bool, optional
        Use T-amplitudes instead of Lambda-amplitudes for CCSD density matrix. Default: False.
    with_mf: bool, optional
        If False, only the difference to the mean-field density-matrix is returned. Default: True.
    mpi_target: integer, optional
        If set to an integer, the density-matrix will only be constructed on the corresponding MPI rank.
        Default: None.

    Returns
    -------
    dm1: (n, n) array
        One-particle reduced density matrix in AO (if `ao_basis=True`) or MO basis (default).
    """
    # --- Loop over pairs of fragments and add projected density-matrix contributions:
    nmoa, nmob = emb.nmo
    dm2aa = np.zeros(4*[nmoa])
    dm2ab = np.zeros(2*[nmoa]+2*[nmob])
    dm2bb = np.zeros(4*[nmob])
    ovlp = emb.get_ovlp()
    for x in emb.get_fragments(mpi_rank=mpi.rank):
        dm2xaa, dm2xab, dm2xbb = x.make_partial_dm2(t_as_lambda=t_as_lambda)
        ra, rb = x.get_mo2co()
        dm2aa += einsum('ijkl,Ii,Jj,Kk,Ll->IJKL', dm2xaa, ra, ra, ra, ra)
        dm2ab += einsum('ijkl,Ii,Jj,Kk,Ll->IJKL', dm2xab, ra, ra, rb, rb)
        dm2bb += einsum('ijkl,Ii,Jj,Kk,Ll->IJKL', dm2xbb, rb, rb, rb, rb)
    if mpi:
        dm2aa, dm2ab, dm2bb = mpi.nreduce(dm2aa, dm2ab, dm2bb, target=mpi_target, logfunc=emb.log.timingv)
        if mpi_target not in (None, mpi.rank):
            return None
    if ao_basis:
        dm2aa = einsum('ijkl,pi,qj,rk,sl->pqrs', dm2aa, *(4*[emb.mo_coeff[0]]))
        dm2ab = einsum('ijkl,pi,qj,rk,sl->pqrs', dm2ab, *(2*[emb.mo_coeff[0]] + 2*[emb.mo_coeff[1]]))
        dm2bb = einsum('ijkl,pi,qj,rk,sl->pqrs', dm2bb, *(4*[emb.mo_coeff[1]]))
    return (dm2aa, dm2ab, dm2bb)
