"""Routines to generate reduced density-matrices (RDMs) from spin-restricted quantum embedding calculations."""

import numpy as np

import pyscf
import pyscf.cc

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
            cf = f.c_active
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


def make_rdm1_ccsd(emb, ao_basis=False, t_as_lambda=False, slow=False,
        ovlp_tol=1e-10, symmetrize=True):
    """Make one-particle reduced density-matrix from partitioned fragment CCSD wave functions.

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
        if ao_basis:
            dm1 = dot(emb.mo_coeff, dm1, emb.mo_coeff.T)
        dm1 = (dm1 + dm1.T)/2
        return dm1

    fragments = emb.fragments

    # --- Slow N^5 algorithm:
    if slow:
        t1 = emb.get_global_t1()
        t2 = emb.get_global_t2()
        cc = pyscf.cc.ccsd.CCSD(emb.mf)
        #cc.conv_tol = 1e-12
        #cc.conv_tol_normt = 1e-10
        if t_as_lambda:
            l1, l2 = t1, t2
        else:
            l1 = emb.get_global_l1()
            l2 = emb.get_global_l2()
        dm1 = cc.make_rdm1(t1=t1, t2=t2, l1=l1, l2=l2, with_frozen=False)
        return finalize(dm1)

    # --- Fast algorithm via fragment-fragment loop:
    # T1/L1-amplitudes can be summed directly
    t1 = emb.get_global_t1()
    l1 = (t1 if t_as_lambda else emb.get_global_l1())

    # --- Preconstruct some C^T.S.C rotation matrices:
    f2fo, f2fv = emb.get_overlap_c2c()
    f2mo, f2mv = emb.get_overlap_m2c()

    # --- Loop over pairs of fragments and add projected density-matrix contributions:
    nocc, nvir = t1.shape
    doo = np.zeros((nocc, nocc))
    dvv = np.zeros((nvir, nvir))
    dov = (t1 + l1 - einsum('ie,me,ma->ia', t1, l1, t1))
    for i1, f1 in enumerate(fragments):
        theta = f1.results.get_t2()
        if theta is None:
            raise RuntimeError("No T2 amplitudes found for %s!" % f1)
        theta = (2*theta - theta.transpose(0,1,3,2))
        theta = f1.project_amplitude_to_fragment(theta, symmetrize=symmetrize)
        # Intermediates - leave left index in cluster basis:
        doo_f1 = np.zeros((f1.n_active_occ, nocc))
        dvv_f1 = np.zeros((f1.n_active_vir, nvir))
        dov += einsum('imae,Pi,Mm,Qa,Ee,ME->PQ', theta, f2mo[i1], f2mo[i1], f2mv[i1], f2mv[i1], l1)
        for i2, f2 in enumerate(fragments):
            if i1 >= i2:
                f2fo12 = f2fo[i1][i2]
                f2fv12 = f2fv[i1][i2]
            else:
                f2fo12 = f2fo[i2][i1].T
                f2fv12 = f2fv[i2][i1].T
            if min(abs(f2fo12).max(), abs(f2fv12).max()) < ovlp_tol:
                emb.log.debugv("Overlap between %s and %s below %.2e; skipping.", f1, f2, ovlp_tol)
                continue
            l2 = (f2.results.get_t2() if t_as_lambda else f2.results.l2)
            if l2 is None:
                raise RuntimeError("No L2 amplitudes found for %s!" % f2)
            l2 = f2.project_amplitude_to_fragment(l2, symmetrize=symmetrize)
            # Theta_jk^ab * l_ik^ab -> ij
            doo_f1 -= einsum('jkab,IKAB,kK,aA,bB,qI->jq', theta, l2, f2fo12, f2fv12, f2fv12, f2mo[i2])
            # Theta_ji^ca * l_ji^cb -> ab
            dvv_f1 += einsum('jica,JICB,jJ,iI,cC,qB->aq', theta, l2, f2fo12, f2fo12, f2fv12, f2mv[i2])
        doo += np.dot(f2mo[i1], doo_f1)
        dvv += np.dot(f2mv[i1], dvv_f1)

    dov += einsum('im,ma->ia', doo, t1)
    dov -= einsum('ie,ae->ia', t1, dvv)
    doo -= einsum('ja,ia->ij', t1, l1)
    dvv += einsum('ia,ib->ab', t1, l1)

    nmo = (nocc + nvir)
    occ, vir = np.s_[:nocc], np.s_[nocc:]
    dm1 = np.zeros((nmo, nmo))
    dm1[occ,occ] = (doo + doo.T)
    dm1[vir,vir] = (dvv + dvv.T)
    dm1[occ,vir] = dov
    dm1[vir,occ] = dov.T
    dm1[np.diag_indices(nocc)] += 2

    return finalize(dm1)

def make_rdm1_ccsd_new(emb, ao_basis=False, t_as_lambda=False, slow=False, symmetrize=False,
        ovlp_tol=1e-10, cluster_local=False):
    """Make one-particle reduced density-matrix from partitioned fragment CCSD wave functions.

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
        if ao_basis:
            dm1 = dot(emb.mo_coeff, dm1, emb.mo_coeff.T)
        dm1 = (dm1 + dm1.T)/2
        return dm1

    # --- Slow N^5 algorithm:
    if slow:
        t1 = emb.get_global_t1()
        t2 = emb.get_global_t2()
        cc = pyscf.cc.ccsd.CCSD(emb.mf)
        #cc.conv_tol = 1e-12
        #cc.conv_tol_normt = 1e-10
        if t_as_lambda:
            l1, l2 = t1, t2
        else:
            l1 = emb.get_global_l1()
            l2 = emb.get_global_l2()
        dm1 = cc.make_rdm1(t1=t1, t2=t2, l1=l1, l2=l2, with_frozen=False)
        return finalize(dm1)

    # --- Fast algorithm via fragment-fragment loop:
    # T1/L1-amplitudes can be summed directly
    t1 = emb.get_global_t1()
    l1 = (t1 if t_as_lambda else emb.get_global_l1())

    # --- Preconstruct some C^T.S.C-type rotation matrices:
    # To cluster/mean-field to cluster:
    rcco, rccv = emb.get_overlap_c2c()
    rmco, rmcv = emb.get_overlap_m2c()
    # To cluster/mean-field to fragment:
    rcfo = emb.get_overlap_c2f()[0]
    rmfo = emb.get_overlap_m2f()[0]

    # --- Loop over pairs of fragments and add projected density-matrix contributions:
    fragments = emb.fragments
    nocc, nvir = t1.shape
    doo = np.zeros((nocc, nocc))
    dvv = np.zeros((nvir, nvir))
    dov = (t1 + l1 - einsum('ie,me,ma->ia', t1, l1, t1))
    for i1, f1 in enumerate(fragments):
        th2x = f1.results.t2x
        th2x = (2*th2x - th2x.transpose(0,1,3,2))
        # Intermediates - leave left index in cluster basis:
        doox = np.zeros((f1.n_active_occ, nocc))
        dvvx = np.zeros((f1.n_active_vir, nvir))
        if symmetrize:
            dov += einsum('Px,xmae,Mm,Qa,Ee,ME->PQ', rmfo[i1], th2x, rmco[i1], rmcv[i1], rmcv[i1], l1) / 2
            dov += einsum('mpea,Mm,Pp,Qa,Ee,ME->PQ', th2x, rmfo[i1], rmco[i1], rmcv[i1], rmcv[i1], l1) / 2
        else:
            dov += einsum('Px,xmae,Mm,Qa,Ee,ME->PQ', rmfo[i1], th2x, rmco[i1], rmcv[i1], rmcv[i1], l1)
        for i2, f2 in enumerate(fragments):
            if i1 >= i2:
                rcco12 = rcco[i1][i2]
                rccv12 = rccv[i1][i2]
                rcfo12 = rcfo[i1][i2]
            else:
                rcco12 = (rcco[i2][i1]).T
                rccv12 = (rccv[i2][i1]).T
            rcfo11 = rcfo[i1][i1]
            rcfo12 = rcfo[i1][i2]
            rcfo21 = rcfo[i2][i1]
            # Screen cluster-cluster loop according to overlap
            if min(abs(rcco12).max(), abs(rccv12).max()) < ovlp_tol:
                emb.log.debugv("Overlap between %s and %s below %.2e; skipping.", f1, f2, ovlp_tol)
                continue
            l2x = (f2.results.t2x if t_as_lambda else f2.results.l2x)

            # Theta_jk^ab * l_ik^ab -> ij
            if not cluster_local:
                #doox -= einsum('jkab,IKAB,kK,aA,bB,QI->jQ', th2x, l2x, rcco12, rccv12, rccv12, rmfo[i2])
                if symmetrize:
                    doox -= einsum('jkab,IKAB,Jj,kK,aA,bB,QI->JQ', th2x, l2x, rcfo11, rcco12, rccv12, rccv12, rmfo[i2]) / 4
                    doox -= einsum('jkab,KIBA,Jj,kK,aA,bB,QI->JQ', th2x, l2x, rcfo11, rcfo12, rccv12, rccv12, rmco[i2]) / 4
                    doox -= einsum('kjba,IKAB,Kk,aA,bB,QI->jQ', th2x, l2x, rcfo21, rccv12, rccv12, rmfo[i2]) / 4
                    if (i1 == i2):
                        doox -= einsum('kjba,kiba,Qi->jQ', th2x, l2x, rmco[i1]) / 4
                else:
                    doox -= einsum('jkab,IKAB,Jj,kK,aA,bB,QI->JQ', th2x, l2x, rcfo11, rcco12, rccv12, rccv12, rmfo[i2])
            elif (i1 == i2):
                doox -= einsum('kjba,kiba,Qi->jQ', th2x, l2x, rmco[i2])

            if not cluster_local:
                # Theta_ji^ca * l_ji^cb -> ab
                if symmetrize:
                    if (i1 == i2):
                        dvvx += einsum('jica,jicb,Qb->aQ', th2x, l2x, rmcv[i1]) / 4
                        dvvx += einsum('ijac,ijbc,Qb->aQ', th2x, l2x, rmcv[i1]) / 4
                    dvvx += einsum('jica,IJBC,Jj,iI,cC,QB->aQ', th2x, l2x, rcfo21, rcfo12, rccv12, rmcv[i2]) / 4
                    dvvx += einsum('ijac,JICB,jJ,Ii,cC,QB->aQ', th2x, l2x, rcfo12, rcfo21, rccv12, rmcv[i2]) / 4
                else:
                    dvvx += einsum('jica,IJBC,Jj,iI,cC,QB->aQ', th2x, l2x, rcfo21, rcfo12, rccv12, rmcv[i2])
            elif (i1 == i2):
                if symmetrize:
                    dvvx += einsum('jica,jicb,Qb->aQ', th2x, l2x, rmcv[i1]) / 2
                    dvvx += einsum('ijac,ijbc,Qb->aQ', th2x, l2x, rmcv[i1]) / 2
                else:
                    dvvx += einsum('jica,jicb,Qb->aQ', th2x, l2x, rmcv[i1])


        #doo += np.dot(rmfo[i1], doox)
        doo += np.dot(rmco[i1], doox)
        dvv += np.dot(rmcv[i1], dvvx)

    dov += einsum('im,ma->ia', doo, t1)
    dov -= einsum('ie,ae->ia', t1, dvv)
    doo -= einsum('ja,ia->ij', t1, l1)
    dvv += einsum('ia,ib->ab', t1, l1)

    nmo = (nocc + nvir)
    occ, vir = np.s_[:nocc], np.s_[nocc:]
    dm1 = np.zeros((nmo, nmo))
    dm1[occ,occ] = (doo + doo.T)
    dm1[vir,vir] = (dvv + dvv.T)
    dm1[occ,vir] = dov
    dm1[vir,occ] = dov.T
    dm1[np.diag_indices(nocc)] += 2

    return finalize(dm1)

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
    t1 = emb.get_global_t1()
    l1 = emb.get_global_l1() if not t_as_lambda else t1

    # --- Loop over pairs of fragments and add projected density-matrix contributions:
    nocc, nvir = t1.shape
    doo = np.zeros((nocc, nocc))
    dvv = np.zeros((nvir, nvir))
    dov = np.zeros((nocc, nvir))
    # MPI loop
    for frag in emb.get_fragments(mpi_rank=mpi.rank):
        th2x = frag.results.t2x
        th2x = (2*th2x - th2x.transpose(0,1,3,2))
        l2x = frag.results.l2x if not t_as_lambda else frag.results.t2x
        # Mean-field to cluster (occupied/virtual):
        co, cv = frag.get_overlap_m2c()
        # Mean-field to fragment (occupied):
        fo = frag.get_overlap_m2f()[0]
        doo -= einsum('kiba,kjba,Ii,Jj->IJ', th2x, l2x, co, co)
        if not symmetrize:
            dvv += einsum('ijca,ijcb,Aa,Bb->AB', th2x, l2x, cv, cv)
            dov += einsum('ijab,Ii,Jj,Aa,Bb,JB->IA', th2x, fo, co, cv, cv, l1)
        else:
            dvv += einsum('jica,jicb,Aa,Bb->AB', th2x, l2x, cv, cv) / 2
            dvv += einsum('ijac,ijbc,Aa,Bb->AB', th2x, l2x, cv, cv) / 2
            dov += einsum('ijab,Ii,Jj,Aa,Bb,JB->IA', th2x, fo, co, cv, cv, l1) / 2
            dov += einsum('jiba,Jj,Ii,Aa,Bb,JB->IA', th2x, fo, co, cv, cv, l1) / 2

    if mpi:
        doo, dvv, dov = _mpi_reduce(emb.log, doo, dvv, dov, mpi_target=mpi_target)
        if mpi_target not in (None, mpi.rank):
            return None

    dov += einsum('IJ,JA->IA', doo, t1)
    dov -= einsum('IB,AB->IA', t1, dvv)
    dov += (t1 + l1 - einsum('IA,JA,JB->IB', t1, l1, t1))
    doo -= einsum('IA,JA->IJ', l1, t1)
    dvv += einsum('IA,IB->AB', t1, l1)

    nmo = (nocc + nvir)
    occ, vir = np.s_[:nocc], np.s_[nocc:]
    dm1 = np.zeros((nmo, nmo))
    dm1[occ,occ] = (doo + doo.T)
    dm1[vir,vir] = (dvv + dvv.T)
    dm1[occ,vir] = dov
    dm1[vir,occ] = dov.T
    if with_mf:
        dm1[np.diag_indices(nocc)] += 2.0
    if ao_basis:
        dm1 = dot(emb.mo_coeff, dm1, emb.mo_coeff.T)

    return dm1


def make_rdm2_demo(emb, ao_basis=False, add_mf=True, symmetrize=True):
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
            cf = f.c_active
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


def make_rdm2_ccsd(emb, ao_basis=False, symmetrize=True, t_as_lambda=False, slow=True):
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
        cc = pyscf.cc.ccsd.CCSD(emb.mf)
        #if 'l12_full' in partition:
        #    l1 = l2 = None
        if t_as_lambda:
            l1, l2 = t1, t2
        else:
            l1 = emb.get_global_t1(get_lambda=True)
            l2 = emb.get_global_t2(get_lambda=True)
        dm2 = cc.make_rdm2(t1=t1, t2=t2, l1=l1, l2=l2, with_frozen=False)
    else:
        raise NotImplementedError()
    if ao_basis:
        dm2 = einsum('ijkl,pi,qj,rk,sl->pqrs', dm2, *(4*[emb.mo_coeff]))
    if symmetrize:
        dm2 = (dm2 + dm2.transpose(1,0,3,2))/2
    return dm2
