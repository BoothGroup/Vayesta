"""Routines to generate reduced density-matrices (RDMs) from spin-restricted quantum embedding calculations."""

import numpy as np

import pyscf
import pyscf.cc
import pyscf.cc.ccsd_rdm_slow
import pyscf.lib

from vayesta.core.util import *
from vayesta.core.vpyscf import ccsd_rdm
from vayesta.core.types import RMP2_WaveFunction
from vayesta.core.types import RCCSD_WaveFunction
from vayesta.mpi import mpi


def _get_mockcc(mo_coeff, max_memory):
    cc = Object()
    cc.mo_coeff = mo_coeff
    cc.frozen = None
    cc.stdout = None
    cc.verbose = 0
    cc.max_memory = max_memory
    return cc

def make_rdm1_ccsd(emb, ao_basis=False, t_as_lambda=False, symmetrize=True, with_mf=True, mpi_target=None, mp2=False):
    """Make one-particle reduced density-matrix from partitioned fragment CCSD wave functions.

    This utilizes index permutations of the CCSD RDM1 equations, such that the RDM1 can be
    calculated as the sum of single-cluster contributions.

    MPI parallelized.

    Parameters
    ----------
    ao_basis : bool, optional
        Return the density-matrix in the AO basis. Default: False.
    t_as_lambda : bool, optional
        Use T-amplitudes instead of Lambda-amplitudes for CCSD density matrix. Default: False.
    symmetrize : bool, optional
        Use symmetrized equations, if possible. Default: True.
    with_mf : bool, optional
        If False, only the difference to the mean-field density-matrix is returned. Default: True.
    mpi_target : integer, optional
        If set to an integer, the density-matrix will only be constructed on the corresponding MPI rank.
        Default : None.
    mp2 : bool, optional
        Make MP2 density-matrix instead. Default: False.

    Returns
    -------
    dm1 : (n, n) array
        One-particle reduced density matrix in AO (if `ao_basis=True`) or MO basis (default).
    """

    nocc, nvir = emb.nocc, emb.nvir
    # --- Fast algorithm via fragment-fragment loop:
    # T1/L1-amplitudes can be summed directly
    if mp2:
        t_as_lambda = True
    else:
        t1 = emb.get_global_t1()
        l1 = emb.get_global_l1() if not t_as_lambda else t1

    # --- Loop over pairs of fragments and add projected density-matrix contributions:
    doo = np.zeros((nocc, nocc))
    dvv = np.zeros((nvir, nvir))
    if not mp2:
        dov = np.zeros((nocc, nvir))
    # MPI loop
    for frag in emb.get_fragments(active=True, mpi_rank=mpi.rank):
        wfx = frag.results.pwf.as_ccsd()
        th2x = (2*wfx.t2 - wfx.t2.transpose(0,1,3,2))
        l2x = wfx.t2 if t_as_lambda else wfx.l2
        if l2x is None:
            raise NotCalculatedError("L2-amplitudes not calculated for %s" % frag)
        # Mean-field to cluster (occupied/virtual):
        co = frag.get_overlap('mo[occ]|cluster[occ]')
        cv = frag.get_overlap('mo[vir]|cluster[vir]')
        # Mean-field to fragment (occupied):
        fo = frag.get_overlap('mo[occ]|frag')
        doo -= einsum('kiba,kjba,Ii,Jj->IJ', th2x, l2x, co, co)
        if not symmetrize:
            dvv += einsum('ijca,ijcb,Aa,Bb->AB', th2x, l2x, cv, cv)
            if not mp2:
                dov += einsum('ijab,Ii,Jj,Aa,Bb,JB->IA', th2x, fo, co, cv, cv, l1)
        else:
            dvv += einsum('jica,jicb,Aa,Bb->AB', th2x, l2x, cv, cv) / 2
            dvv += einsum('ijac,ijbc,Aa,Bb->AB', th2x, l2x, cv, cv) / 2
            if not mp2:
                dov += einsum('ijab,Ii,Jj,Aa,Bb,JB->IA', th2x, fo, co, cv, cv, l1) / 2
                dov += einsum('jiba,Jj,Ii,Aa,Bb,JB->IA', th2x, fo, co, cv, cv, l1) / 2

    if mpi:
        if mp2:
            doo, dvv = mpi.nreduce(doo, dvv, target=mpi_target, logfunc=emb.log.timingv)
        else:
            doo, dov, dvv = mpi.nreduce(doo, dov, dvv, target=mpi_target, logfunc=emb.log.timingv)
        if mpi_target not in (None, mpi.rank):
            return None

    if not mp2:
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
    if not mp2:
        dm1[occ,vir] = dov
        dm1[vir,occ] = dov.T
    if with_mf:
        dm1[np.diag_indices(nocc)] += 2.0
    if ao_basis:
        dm1 = dot(emb.mo_coeff, dm1, emb.mo_coeff.T)

    return dm1

def make_rdm1_ccsd_proj_lambda(emb, ao_basis=False, t_as_lambda=False, with_mf=True, sym_t2=True, mpi_target=None):
    """Make one-particle reduced density-matrix from partitioned fragment CCSD wave functions.

    MPI parallelized.

    Parameters
    ----------
    ao_basis : bool, optional
        Return the density-matrix in the AO basis. Default: False.
    t_as_lambda : bool, optional
        Use T-amplitudes instead of Lambda-amplitudes for CCSD density matrix. Default: False.
    with_mf : bool, optional
        If False, only the difference to the mean-field density-matrix is returned. Default: True.
    mpi_target : integer, optional
        If set to an integer, the density-matrix will only be constructed on the corresponding MPI rank.
        Default: None.

    Returns
    -------
    dm1 : (n, n) array
        One-particle reduced density matrix in AO (if `ao_basis=True`) or MO basis (default).
    """
    # --- Loop over pairs of fragments and add projected density-matrix contributions:
    dm1 = np.zeros((emb.nmo, emb.nmo))
    for x in emb.get_fragments(active=True, mpi_rank=mpi.rank):
        rx = x.get_overlap('mo|cluster')
        dm1x = x.make_fragment_dm1(t_as_lambda=t_as_lambda, sym_t2=sym_t2)
        dm1 += np.linalg.multi_dot((rx, dm1x, rx.T))
    if mpi:
        dm1 = mpi.nreduce(dm1, target=mpi_target, logfunc=emb.log.timingv)
        if mpi_target not in (None, mpi.rank):
            return None
    if with_mf:
        dm1[np.diag_indices(emb.nocc)] += 2.0
    if ao_basis:
        dm1 = dot(emb.mo_coeff, dm1, emb.mo_coeff.T)
    return dm1

def make_rdm1_ccsd_global_wf(emb, ao_basis=False, with_mf=True, t_as_lambda=None, with_t1=True,
        svd_tol=1e-3, ovlp_tol=None, use_sym=True, late_t2_sym=True, mpi_target=None, slow=False):
    """Make one-particle reduced density-matrix from partitioned fragment CCSD wave functions.

    This replaces make_rdm1_ccsd_old.

    TODO: MPI

    Parameters
    ----------
    ao_basis : bool, optional
        Return the density-matrix in the AO basis. Default: False.
    with_mf : bool, optional
        Add mean-field contribution to the density-matrix. Default: True.
    t_as_lambda : bool, optional
        Use T-amplitudes inplace of Lambda-amplitudes for CCSD density matrix.
        If `None`, `emb.opts.t_as_lambda` will be used. Default: None.
    with_t1 : bool, optional
        If False, T1 and L1 amplitudes are assumed 0. Default: False.
    svd_tol : float, optional
        Left/right singular vectors of singular values of cluster x-y overlap matrices
        smaller than `svd_tol` will be removed to speed up the calculation. Default: 1e-3.
    ovlp_tol : float, optional
        Fragment pairs with a smaller than `ovlp_tol` maximum singular value in their cluster overlap,
        will be skipped. Default: None.
    use_sym : bool, optional
        Make use of symmetry relations, to speed up calculation. Default: True.
    late_t2_sym : bool, optional
        Perform symmetrization of T2/L2-amplitudes late, in order to speed up
        calculation with large N(bath)/N(fragment) ratios. Default: True.
    mpi_target : int or None, optional
    slow : bool, optional
        Combine to global CCSD wave function first, then build density matrix.
        Equivalent, but does not scale well. Default: False

    Returns
    -------
    dm1 : array(n(MO),n(MO)) or array(n(AO),n(AO))
        One-particle reduced density matrix in AO (if `ao_basis=True`) or MO basis (default).
    """
    if t_as_lambda is None:
        t_as_lambda = emb.opts.t_as_lambda

    # === Slow algorithm (O(N^5)?): Form global N^4 T2/L2-amplitudes first
    if slow:
        t1 = emb.get_global_t1()
        t2 = emb.get_global_t2()
        if t_as_lambda:
            l1, l2 = t1, t2
        else:
            l1 = emb.get_global_l1()
            l2 = emb.get_global_l2()
        if not with_t1:
            t1 = l1 = np.zeros_like(t1)
        mockcc = _get_mockcc(emb.mo_coeff, emb.mf.max_memory)
        #dm1 = pyscf.cc.ccsd_rdm.make_rdm1(mockcc, t1=t1, t2=t2, l1=l1, l2=l2, ao_repr=ao_basis, with_mf=with_mf)
        dm1 = ccsd_rdm.make_rdm1(mockcc, t1=t1, t2=t2, l1=l1, l2=l2, ao_repr=ao_basis, with_mf=with_mf)
        return dm1

    # === Fast algorithm via fragment-fragment loop
    # --- Setup
    if ovlp_tol is None:
        ovlp_tol = svd_tol
    total_sv = kept_sv = 0
    total_xy = kept_xy = 0
    # T1/L1-amplitudes can be summed directly
    if with_t1:
        t1 = emb.get_global_t1()
        l1 = (t1 if t_as_lambda else emb.get_global_l1())
    # Preconstruct some matrices, since the construction scales as N^3
    ovlp = emb.get_ovlp()
    cs_occ = np.dot(emb.mo_coeff_occ.T, ovlp)
    cs_vir = np.dot(emb.mo_coeff_vir.T, ovlp)
    # Make projected WF available via remote memory access
    if mpi:
        # TODO: use L-amplitudes of cluster X and T-amplitudes,
        # Only send T-amplitudes via RMA?
        #rma = {x.id: x.results.pwf.pack() for x in emb.get_fragments(active=True, mpi_rank=mpi.rank)}
        rma = {x.id: x.results.pwf.pack() for x in emb.get_fragments(active=True, mpi_rank=mpi.rank, sym_parent=None)}
        rma = mpi.create_rma_dict(rma)

    # --- Loop over pairs of fragments and add projected density-matrix contributions:
    doo = np.zeros((emb.nocc, emb.nocc))
    dvv = np.zeros((emb.nvir, emb.nvir))
    if with_t1:
        dov = np.zeros((emb.nocc, emb.nvir))
    symfilter = dict(sym_parent=None) if use_sym else {}
    maxgen = None if use_sym else 0
    for fx in emb.get_fragments(active=True, mpi_rank=mpi.rank, **symfilter):
        wfx = fx.results.pwf.as_ccsd()
        if not late_t2_sym:
            wfx = wfx.restore()
        theta = (2*wfx.t2 - wfx.t2.transpose(0,1,3,2))

        # Intermediates: leave left index in cluster-x basis:
        doox = np.zeros((fx.cluster.nocc_active, emb.nocc))
        dvvx = np.zeros((fx.cluster.nvir_active, emb.nvir))

        cx_occ = fx.get_overlap('mo[occ]|cluster[occ]')
        cx_vir = fx.get_overlap('mo[vir]|cluster[vir]')
        cfx = fx.get_overlap('cluster[occ]|frag')
        mfx = fx.get_overlap('mo[occ]|frag')

        # Loop over fragments y:
        for fy_parent in emb.get_fragments(active=True, **symfilter):

            if mpi:
                if fy_parent.solver == 'MP2':
                    wfy = RMP2_WaveFunction.unpack(rma[fy_parent.id]).as_ccsd()
                else:
                    wfy = RCCSD_WaveFunction.unpack(rma[fy_parent.id])
            else:
                wfy = fy_parent.results.pwf.as_ccsd()
            if not late_t2_sym:
                wfy = wfy.restore()
            cfy = fy_parent.get_overlap('cluster[occ]|frag')

            for fy, (cy_frag, cy_occ_ao, cy_vir_ao) in fy_parent.loop_symmetry_children((fy_parent.c_frag,
                    fy_parent.cluster.c_occ, fy_parent.cluster.c_vir), include_self=True, maxgen=maxgen):

                cy_occ = np.dot(cs_occ, cy_occ_ao)
                cy_vir = np.dot(cs_vir, cy_vir_ao)
                # Overlap between cluster x and cluster y:
                rxy_occ = np.dot(cx_occ.T, cy_occ)
                rxy_vir = np.dot(cx_vir.T, cy_vir)
                mfy = np.dot(cs_occ, cy_frag)

                if svd_tol is not None:
                    def svd(a):
                        nonlocal total_sv, kept_sv
                        u, s, v = np.linalg.svd(a, full_matrices=False)
                        if svd_tol is not None:
                            keep = (s >= svd_tol)
                            total_sv += len(keep)
                            kept_sv += sum(keep)
                            u, s, v = u[:,keep], s[keep], v[keep]
                        return u, s, v
                    uxy_occ, sxy_occ, vxy_occ = svd(rxy_occ)
                    uxy_vir, sxy_vir, vxy_vir = svd(rxy_vir)
                    uxy_occ *= np.sqrt(sxy_occ)[np.newaxis,:]
                    uxy_vir *= np.sqrt(sxy_vir)[np.newaxis,:]
                    vxy_occ *= np.sqrt(sxy_occ)[:,np.newaxis]
                    vxy_vir *= np.sqrt(sxy_vir)[:,np.newaxis]
                else:
                    nsv = (min(rxy_occ.shape[0], rxy_occ.shape[1])
                         + min(rxy_vir.shape[0], rxy_vir.shape[1]))
                    total_sv = kept_sv = (total_sv + nsv)

                # --- If ovlp_tol is given, the cluster x-y pairs will be screened based on the
                # largest singular value of the occupied and virtual overlap matrices
                total_xy += 1
                if ovlp_tol is not None:
                    if svd_tol:
                        rxy_occ_norm = (sxy_occ[0] if len(sxy_occ) > 0 else 0.0)
                        rxy_vir_norm = (sxy_vir[0] if len(sxy_vir) > 0 else 0.0)
                    else:
                        rxy_occ_norm = np.linalg.norm(rxy_occ, ord=2)
                        rxy_vir_norm = np.linalg.norm(rxy_vir, ord=2)
                    if (min(rxy_occ_norm, rxy_vir_norm) < ovlp_tol):
                        emb.log.debugv("Overlap of fragment pair %s - %s below %.2e; skipping pair.", fx, fy, ovlp_tol)
                        continue
                kept_xy += 1

                l2 = wfy.t2 if (t_as_lambda or fy_parent.solver == 'MP2') else wfy.l2
                if l2 is None:
                    raise RuntimeError("No L2 amplitudes found for %s!" % y)

                # Theta_jk^ab * l_ik^ab -> ij
                #doox -= einsum('jkab,IKAB,kK,aA,bB,QI->jQ', theta, l2, rxy_occ, rxy_vir, rxy_vir, cy_occ)
                ## Theta_ji^ca * l_ji^cb -> ab
                #dvvx += einsum('jica,JICB,jJ,iI,cC,QB->aQ', theta, l2, rxy_occ, rxy_occ, rxy_vir, cy_vir)

                # Only multiply with O(N)-scaling cy_occ/cy_vir in last step:

                if not late_t2_sym:
                    if svd_tol is None:
                        tmp = einsum('(ijab,jJ,aA->iJAb),IJAB->iIbB', theta, rxy_occ, rxy_vir, l2)
                    else:
                        tmp = einsum('(ijab,jS,aP->iSPb),(SJ,PA,IJAB->ISPB)->iIbB', theta, uxy_occ, uxy_vir, vxy_occ, vxy_vir, l2)
                    tmpo = -einsum('iIbB,bB->iI', tmp, rxy_vir)
                    doox += np.dot(tmpo, cy_occ.T)
                    tmpv = einsum('iIbB,iI->bB', tmp, rxy_occ)
                    dvvx += np.dot(tmpv, cy_vir.T)
                else:
                    # Calculate some overlap matrices:
                    if svd_tol is None:
                        cfxy_occ = dot(rxy_occ, cfy)
                        cfyx_occ = dot(rxy_occ.T, cfx)
                        ffxy = dot(cfx.T, rxy_occ, cfy)
                    else:
                        cfxy_occ = dot(uxy_occ, vxy_occ, cfy)
                        cfyx_occ = dot(vxy_occ.T, uxy_occ.T, cfx)
                        ffxy = dot(cfx.T, uxy_occ, vxy_occ, cfy)

                    # --- Occupied
                    # Deal with both virtual overlaps here:
                    if svd_tol is None:
                        t2tmp = einsum('xjab,aA,bB->xjAB', theta, rxy_vir, rxy_vir) # frag * cluster^4
                        l2tmp = l2
                    else:
                        t2tmp = einsum('xjab,aS,bP->xjSP', theta, uxy_vir, uxy_vir)
                        l2tmp = einsum('yjab,Sa,Pb->yjSP', l2, vxy_vir, vxy_vir)
                    # T2 * L2
                    if svd_tol is None:
                        tmp = -einsum('(xjAB,jJ->xJAB),YJAB->xY', t2tmp, rxy_occ, l2tmp)/4
                    else:
                        tmp = -einsum('(xjAB,jS->xSAB),(SJ,YJAB->YSAB)->xY', t2tmp, uxy_occ, vxy_occ, l2tmp)/4
                    doox += dot(cfx, tmp, mfy.T)
                    # T2 * L2.T
                    tmp = -einsum('(xjAB,jY->xYAB),YIBA->xI', t2tmp, cfxy_occ, l2tmp)/4
                    doox += dot(cfx, tmp, cy_occ.T)
                    # T2.T * L2
                    tmp = -einsum('xiBA,(Jx,YJAB->YxAB)->iY', t2tmp, cfyx_occ, l2tmp)/4
                    doox += np.dot(tmp, mfy.T)
                    # T2.T * L2.T
                    tmp = -einsum('xiBA,xY,YIBA->iI', t2tmp, ffxy, l2tmp)/4
                    doox += np.dot(tmp, cy_occ.T)

                    # --- Virtual
                    # T2 * L2 and T2.T * L2.T
                    if svd_tol is None:
                        t2tmp = einsum('xjab,xY,jJ->YJab', theta, ffxy, rxy_occ)
                        tmp = einsum('(YJab,aA->YJAb),YJAB->bB', t2tmp, rxy_vir, l2)/4
                        tmp += einsum('(YIba,aA->YIbA),YIBA->bB', t2tmp, rxy_vir, l2)/4
                    else:
                        t2tmp = einsum('xjab,jS->xSab', theta, uxy_occ)
                        l2tmp = einsum('YJAB,SJ,xY->xSAB', l2, vxy_occ, ffxy)
                        tmp = einsum('(xSab,aP->xSPb),(PA,xSAB->xSPB)->bB', t2tmp, uxy_vir, vxy_vir, l2tmp)/4
                        tmp += einsum('(xSba,aP->xSbP),(PA,xSBA->xSBP)->bB', t2tmp, uxy_vir, vxy_vir, l2tmp)/4
                    # T2 * L2.T and T2.T * L2
                    t2tmp = einsum('xjab,jY->xYab', theta, cfxy_occ)
                    l2tmp = einsum('Jx,YJAB->YxAB', cfyx_occ, l2)
                    if svd_tol is None:
                        tmp += einsum('(xYab,aA->xYAb),YxBA->bB', t2tmp, rxy_vir, l2tmp)/4
                        tmp += einsum('(xYba,aA->xYbA),YxAB->bB', t2tmp, rxy_vir, l2tmp)/4
                    else:
                        tmp += einsum('(xYab,aS->xYSb),(SA,YxBA->YxBS)->bB', t2tmp, uxy_vir, vxy_vir, l2tmp)/4
                        tmp += einsum('(xYba,aS->xYbS),(SA,YxAB->YxSB)->bB', t2tmp, uxy_vir, vxy_vir, l2tmp)/4
                    dvvx += np.dot(tmp, cy_vir.T)

        doo += np.dot(cx_occ, doox)
        dvv += np.dot(cx_vir, dvvx)
        # --- Use symmetry of fragments (rotations and translations)
        if use_sym:
            # Transform right index of intermediates to AO basis:
            doox = dot(doox, emb.mo_coeff_occ.T)
            dvvx = dot(dvvx, emb.mo_coeff_vir.T)
            # Loop over symmetry children of x:
            for fx2, (cx2_occ, cx2_vir, doox2, dvvx2) in fx.loop_symmetry_children(
                    (fx.cluster.c_occ, fx.cluster.c_vir, doox, dvvx), axes=[0,0,1,1]):
                doo += dot(cs_occ, cx2_occ, doox2, cs_occ.T)
                dvv += dot(cs_vir, cx2_vir, dvvx2, cs_vir.T)

        # D[occ,vir] <- "T2 * L1"
        if with_t1:
            l1x = dot(cx_occ.T, l1, cx_vir)
            if not late_t2_sym:
                dovx = einsum('ijab,jb->ia', theta, l1x)
            else:
                dovx1 = einsum('xjab,jb->xa', theta, l1x)/2
                dovx2 = einsum('xiba,(jx,jb->xb)->ia', theta, cfx, l1x)/2
            for fx2, (cx2_frag, cx2_occ, cx2_vir) in fx.loop_symmetry_children(
                    (fx.c_frag, fx.cluster.c_occ, fx.cluster.c_vir), include_self=True, maxgen=maxgen):
                cx2_occ = np.dot(cs_occ, cx2_occ)
                cx2_vir = np.dot(cs_vir, cx2_vir)
                if not late_t2_sym:
                    dov += einsum('ia,Ii,Aa->IA', dovx, cx2_occ, cx2_vir)
                else:
                    dov += dot(cs_occ, cx2_frag, dovx1, cx2_vir.T)
                    dov += dot(cx2_occ, dovx2, cx2_vir.T)

    if mpi:
        rma.clear()
        doo, dov, dvv = mpi.nreduce(doo, dov, dvv, target=mpi_target, logfunc=emb.log.timingv)
        # Make sure no more MPI calls are made after returning some ranks early!
        if mpi_target not in (None, mpi.rank):
            return None

    if with_t1:
        dov += (t1 + l1 - einsum('ie,me,ma->ia', t1, l1, t1))
        dov += einsum('im,ma->ia', doo, t1)
        dov -= einsum('ie,ae->ia', t1, dvv)
        doo -= einsum('ja,ia->ij', t1, l1)
        dvv += einsum('ia,ib->ab', t1, l1)

    # --- Combine full DM:
    occ, vir = np.s_[:emb.nocc], np.s_[emb.nocc:]
    dm1 = np.zeros((emb.nmo, emb.nmo))
    dm1[occ,occ] = (doo + doo.T)
    dm1[vir,vir] = (dvv + dvv.T)
    if with_t1:
        dm1[occ,vir] = dov
        dm1[vir,occ] = dov.T
    if with_mf:
        dm1[np.diag_indices(emb.nocc)] += 2
    if ao_basis:
        dm1 = dot(emb.mo_coeff, dm1, emb.mo_coeff.T)

    # --- Some information:
    emb.log.debug("Cluster-pairs: total= %d  kept= %d (%.1f%%)", total_xy, kept_xy, 100*kept_xy/total_xy)
    emb.log.debug("Singular values: total= %d  kept= %d (%.1f%%)", total_sv, kept_sv, 100*kept_sv/total_sv)

    return dm1


# --- Two-particle
# ----------------

def make_rdm2_ccsd_global_wf(emb, ao_basis=False, symmetrize=True, t_as_lambda=False, slow=True, with_dm1=True):
    """Recreate global two-particle reduced density-matrix from fragment calculations.

    Parameters
    ----------
    ao_basis : bool, optional
        Return the density-matrix in the AO basis. Default: False.
    symmetrize : bool, optional
        Symmetrize the density-matrix at the end of the calculation. Default: True.
    t_as_lambda : bool, optional
        Use T-amplitudes instead of Lambda-amplitudes for CCSD density matrix. Default: False.
    slow : bool, optional
        Combine to global CCSD wave function first, then build density matrix.
        Equivalent, but does not scale well. Default: True.

    Returns
    -------
    dm2 : (n, n, n, n) array
        Two-particle reduced density matrix in AO (if `ao_basis=True`) or MO basis (default).
    """
    if slow:
        t1 = emb.get_global_t1()
        t2 = emb.get_global_t2()
        if t_as_lambda:
            l1, l2 = t1, t2
        else:
            l1 = emb.get_global_t1(get_lambda=True)
            l2 = emb.get_global_t2(get_lambda=True)
        mockcc = _get_mockcc(emb.mo_coeff, emb.mf.max_memory)
        #dm2 = pyscf.cc.ccsd_rdm.make_rdm2(mockcc, t1=t1, t2=t2, l1=l1, l2=l2, with_frozen=False, with_dm1=with_dm1)
        dm2 = ccsd_rdm.make_rdm2(mockcc, t1=t1, t2=t2, l1=l1, l2=l2, with_frozen=False, with_dm1=with_dm1)
    else:
        raise NotImplementedError
    if symmetrize:
        dm2 = (dm2 + dm2.transpose(1,0,3,2))/2
    if ao_basis:
        dm2 = einsum('ijkl,pi,qj,rk,sl->pqrs', dm2, *(4*[emb.mo_coeff]))
    return dm2

def make_rdm2_ccsd_proj_lambda(emb, with_dm1=True, ao_basis=False, t_as_lambda=False, sym_t2=True, sym_dm2=True,
        approx_cumulant=True, mpi_target=None):
    """Make two-particle reduced density-matrix from partitioned fragment CCSD wave functions.

    Without 1DM!

    MPI parallelized.

    Parameters
    ----------
    ao_basis : bool, optional
        Return the density-matrix in the AO basis. Default: False.
    t_as_lambda : bool, optional
        Use T-amplitudes instead of Lambda-amplitudes for CCSD density matrix. Default: False.
    with_mf : bool, optional
        If False, only the difference to the mean-field density-matrix is returned. Default: True.
    mpi_target : integer, optional
        If set to an integer, the density-matrix will only be constructed on the corresponding MPI rank.
        Default: None.

    Returns
    -------
    dm1 : (n, n) array
        One-particle reduced density matrix in AO (if `ao_basis=True`) or MO basis (default).
    """
    # --- Loop over pairs of fragments and add projected density-matrix contributions:
    dm2 = np.zeros((emb.nmo, emb.nmo, emb.nmo, emb.nmo))
    ovlp = emb.get_ovlp()
    for x in emb.get_fragments(active=True, mpi_rank=mpi.rank):
        rx = x.get_overlap('mo|cluster')
        dm2x = x.make_fragment_dm2cumulant(t_as_lambda=t_as_lambda, sym_t2=sym_t2, approx_cumulant=approx_cumulant)
        dm2 += einsum('ijkl,Ii,Jj,Kk,Ll->IJKL', dm2x, rx, rx, rx, rx)
    if mpi:
        dm2 = mpi.nreduce(dm2, target=mpi_target, logfunc=emb.log.timingv)
        if mpi_target not in (None, mpi.rank):
            return None
    if isinstance(with_dm1, np.ndarray) or with_dm1:
        if with_dm1 is True:
            dm1 = emb.make_rdm1()
        else:
            dm1 = with_dm1
        if not approx_cumulant:
            dm2 += (einsum('ij,kl->ijkl', dm1, dm1) - einsum('ij,kl->iklj', dm1, dm1)/2)
        # Remove half of the mean-field contribution
        # (in PySCF the entire MF is removed and afterwards half is added back in a (i,j) loop)
        elif (approx_cumulant in (1, True)):
            dm1 = dm1.copy()
            dm1[np.diag_indices(emb.nocc)] -= 1
            for i in range(emb.nocc):
                dm2[i,i,:,:] += dm1 * 2
                dm2[:,:,i,i] += dm1 * 2
                dm2[:,i,i,:] -= dm1
                dm2[i,:,:,i] -= dm1.T
        elif approx_cumulant == 2:
            raise NotImplementedError
        else:
            raise ValueError

    if ao_basis:
        dm2 = einsum('ijkl,pi,qj,rk,sl->pqrs', dm2, *(4*[emb.mo_coeff]))
    return dm2
