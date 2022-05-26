"""Routines to generate reduced density-matrices (RDMs) from spin-unrestricted quantum embedding calculations."""

import numpy as np

import pyscf
import pyscf.cc

from vayesta.core.util import *
from vayesta.mpi import mpi


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
        coa, cob = frag.get_overlap('mo[occ]|cluster[occ]')
        cva, cvb = frag.get_overlap('mo[vir]|cluster[vir]')
        # Mean-field to fragment (occupied):
        foa, fob = frag.get_overlap('mo[occ]|frag')

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


def make_rdm1_ccsd_global_wf(emb, ao_basis=False, with_mf=True, t_as_lambda=False, with_t1=True,
        svd_tol=1e-3, ovlp_tol=None, use_sym=True, late_t2_sym=True, mpi_target=None, slow=True):
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
    t_init = timer()

    if t_as_lambda is None:
        t_as_lambda = emb.opts.t_as_lambda

    # === Slow algorithm (O(N^5)?): Form global N^4 T2/L2-amplitudes first
    if slow:
        t1 = emb.get_global_t1()
        t2 = emb.get_global_t2()
        mockcc = Object()
        mockcc.frozen = None
        mockcc.mo_coeff = emb.mo_coeff
        if t_as_lambda:
            l1, l2 = t1, t2
        else:
            l1 = emb.get_global_l1()
            l2 = emb.get_global_l2()
        if not with_t1:
            t1 = l1 = (np.zeros_like(t1[0]), np.zeros_like(t1[1]))
        dm1 = pyscf.cc.uccsd_rdm.make_rdm1(mockcc, t1=t1, t2=t2, l1=l1, l2=l2, ao_repr=ao_basis, with_mf=with_mf)
        emb.log.timing("Time for make_rdm1: %s", time_string(timer()-t_init))
        return dm1

    # TODO
    raise NotImplementedError

def make_rdm2_ccsd_global_wf(emb, ao_basis=False, symmetrize=False, t_as_lambda=False, slow=True, with_dm1=True):
    """Recreate global two-particle reduced density-matrix from fragment calculations.

    Parameters
    ----------
    ao_basis: bool, optional
        Return the density-matrix in the AO basis. Default: False.
    symmetrize: bool, optional
        Symmetrize the density-matrix at the end of the calculation. Default: True.
    t_as_lambda: bool, optional
        Use T-amplitudes instead of Lambda-amplitudes for CCSD density matrix. Default: False.
    slow: bool, optional
        Combine to global CCSD wave function first, then build density matrix.
        Equivalent, but does not scale well. Default: True.

    Returns
    -------
    dm2: (n, n, n, n) array
        Two-particle reduced density matrix in AO (if `ao_basis=True`) or MO basis (default).
    """

    if slow:
        t1 = emb.get_global_t1()
        t2 = emb.get_global_t2()
        cc = pyscf.cc.uccsd.UCCSD(emb.mf)
        if t_as_lambda:
            l1, l2 = t1, t2
        else:
            l1 = emb.get_global_t1(get_lambda=True)
            l2 = emb.get_global_t2(get_lambda=True)
        dm2 = cc.make_rdm2(t1=t1, t2=t2, l1=l1, l2=l2, with_frozen=False, with_dm1=with_dm1)
    else:
        raise NotImplementedError()
    if ao_basis:
        raise NotImplementedError()
        #dm2 = einsum('ijkl,pi,qj,rk,sl->pqrs', dm2, *(4*[emb.mo_coeff]))
    if symmetrize:
        raise NotImplementedError()
        #dm2 = (dm2 + dm2.transpose(1,0,3,2))/2
    return dm2

def make_rdm2_ccsd_proj_lambda(emb, ao_basis=False, t_as_lambda=False, with_dm1=True, mpi_target=None):
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
        dm2xaa, dm2xab, dm2xbb = x.make_fragment_dm2cumulant(t_as_lambda=t_as_lambda)
        ra, rb =  x.get_overlap('mo|cluster')
        dm2aa += einsum('ijkl,Ii,Jj,Kk,Ll->IJKL', dm2xaa, ra, ra, ra, ra)
        dm2ab += einsum('ijkl,Ii,Jj,Kk,Ll->IJKL', dm2xab, ra, ra, rb, rb)
        dm2bb += einsum('ijkl,Ii,Jj,Kk,Ll->IJKL', dm2xbb, rb, rb, rb, rb)
    if mpi:
        dm2aa, dm2ab, dm2bb = mpi.nreduce(dm2aa, dm2ab, dm2bb, target=mpi_target, logfunc=emb.log.timingv)
        if mpi_target not in (None, mpi.rank):
            return None
    if with_dm1:
        if with_dm1 is True:
            dm1a, dm1b = emb.make_rdm1()
        else:
            dm1a, dm1b = (with_dm1[0].copy(), with_dm1[1].copy())
        nocca, noccb = emb.nocc
        dm1a[np.diag_indices(nocca)] -= 0.5
        dm1b[np.diag_indices(noccb)] -= 0.5
        for i in range(nocca):
            dm2aa[i,i,:,:] += dm1a
            dm2aa[:,:,i,i] += dm1a
            dm2aa[:,i,i,:] -= dm1a
            dm2aa[i,:,:,i] -= dm1a.T
            dm2ab[i,i,:,:] += dm1b
        for i in range(noccb):
            dm2bb[i,i,:,:] += dm1b
            dm2bb[:,:,i,i] += dm1b
            dm2bb[:,i,i,:] -= dm1b
            dm2bb[i,:,:,i] -= dm1b.T
            dm2ab[:,:,i,i] += dm1a

    if ao_basis:
        dm2aa = einsum('ijkl,pi,qj,rk,sl->pqrs', dm2aa, *(4*[emb.mo_coeff[0]]))
        dm2ab = einsum('ijkl,pi,qj,rk,sl->pqrs', dm2ab, *(2*[emb.mo_coeff[0]] + 2*[emb.mo_coeff[1]]))
        dm2bb = einsum('ijkl,pi,qj,rk,sl->pqrs', dm2bb, *(4*[emb.mo_coeff[1]]))
    return (dm2aa, dm2ab, dm2bb)
