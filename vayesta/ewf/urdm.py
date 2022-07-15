"""Routines to generate reduced density-matrices (RDMs) from spin-unrestricted quantum embedding calculations."""

import numpy as np

import pyscf
import pyscf.cc

from vayesta.core.util import *
from vayesta.core.vpyscf import uccsd_rdm
from vayesta.ewf.rdm import _get_mockcc
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

    for frag in emb.get_fragments(active=True, mpi_rank=mpi.rank):
        wfx = frag.results.pwf.as_ccsd()
        t2xaa, t2xab, t2xba, t2xbb = wfx.t2
        l2xaa, l2xab, l2xba, l2xbb = wfx.l2 if not t_as_lambda else wfx.t2
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
    if t_as_lambda is None:
        t_as_lambda = emb.opts.t_as_lambda

    nocca, noccb = emb.nocc
    nvira, nvirb = emb.nvir

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
            t1 = l1 = (np.zeros_like(t1[0]), np.zeros_like(t1[1]))
        mockcc = _get_mockcc(emb.mo_coeff, emb.mf.max_memory)
        #dm1 = pyscf.cc.uccsd_rdm.make_rdm1(mockcc, t1=t1, t2=t2, l1=l1, l2=l2, ao_repr=ao_basis, with_mf=with_mf)
        dm1a, dm1b = uccsd_rdm.make_rdm1(mockcc, t1=t1, t2=t2, l1=l1, l2=l2, ao_repr=ao_basis, with_mf=with_mf)
        return dm1a, dm1b

    # === Fast algorithm via fragment-fragment loop
    # --- Setup
    if ovlp_tol is None:
        ovlp_tol = svd_tol
    total_sv_a = kept_sv_a = 0
    total_sv_b = kept_sv_b = 0
    total_xy = kept_xy = 0
    # T1/L1-amplitudes can be summed directly
    if with_t1:
        t1a, t1b = emb.get_global_t1()
        l1a, l1b = (t1 if t_as_lambda else emb.get_global_l1())
    # Preconstruct some matrices, since the construction scales as N^3
    if use_sym:
        raise NotImplemented()
        ovlp = emb.get_ovlp()
        cs_occ = np.dot(emb.mo_coeff_occ.T, ovlp)
        cs_vir = np.dot(emb.mo_coeff_vir.T, ovlp)
    # Make projected WF available via remote memory access
    if mpi:
        raise NotImplemented()
        # TODO: use L-amplitudes of cluster X and T-amplitudes,
        # Only send T-amplitudes via RMA?
        rma = {x.id: x.results.pwf.pack() for x in emb.get_fragments(active=True, mpi_rank=mpi.rank)}
        rma = mpi.create_rma_dict(rma)

    # --- Loop over pairs of fragments and add projected density-matrix contributions:



    dooa = np.zeros((nocca, nocca))
    doob = np.zeros((noccb, noccb))
    dvva = np.zeros((nvira, nvira))
    dvvb = np.zeros((nvirb, nvirb))

    xt1a = np.zeros((nvira, nocca))
    xt1b = np.zeros((nvirb, noccb))
    xt2a = np.zeros((nvira, nocca))
    xt2b = np.zeros((nvirb, noccb))

    dvoa = np.zeros((nvira, nocca))
    dvob = np.zeros((nvirb, noccb))

    if with_t1:
        dova = np.zeros((nocca, nvira))
        dovb = np.zeros((noccb, nvirb))
    xfilter = dict(sym_parent=None) if use_sym else {}
    for x in emb.get_fragments(active=True, mpi_rank=mpi.rank, **xfilter):
        wfx = x.results.pwf.as_ccsd()
        if not late_t2_sym:
            wfx = wfx.restore()

        t2aa = wfx.t2aa
        t2ab = wfx.t2ab
        t2bb = wfx.t2bb

        # Intermediates: leave left index in cluster-x basis:
        dooxa = np.zeros((x.cluster.nocc_active[0], emb.nocc[0]))
        dooxb = np.zeros((x.cluster.nocc_active[1], emb.nocc[1]))
        dvvxa = np.zeros((x.cluster.nvir_active[0], emb.nvir[0]))
        dvvxb = np.zeros((x.cluster.nvir_active[1], emb.nvir[1]))
        dvoxa = np.zeros((x.cluster.nvir_active[0], x.cluster.nocc_active[0]))
        dvoxb = np.zeros((x.cluster.nvir_active[1], x.cluster.nocc_active[1]))

        xt1xa = np.zeros((x.cluster.nocc_active[0], nocca))
        xt1xb = np.zeros((x.cluster.nocc_active[1], noccb))
        xt2xa = np.zeros((x.cluster.nvir_active[0], nvira))
        xt2xb = np.zeros((x.cluster.nvir_active[1], nvirb))

        cx_occ_a, cx_occ_b = x.get_overlap('mo[occ]|cluster[occ]')
        cx_vir_a, cx_vir_b = x.get_overlap('mo[vir]|cluster[vir]')

        # Loop over ALL fragments y:
        for y in emb.get_fragments(active=True):

            if mpi:
                if y.solver == 'MP2':
                    wfy = UMP2_WaveFunction.unpack(rma[y.id]).as_ccsd()
                else:
                    wfy = UCCSD_WaveFunction.unpack(rma[y.id])
            else:
                wfy = y.results.pwf.as_ccsd()
            if not late_t2_sym:
                wfy = wfy.restore()

            # Constructing these overlap matrices scales as N(AO)^2,
            # however they are cashed and will only be calculated N(frag) times
            cy_occ_a, cy_occ_b = y.get_overlap('mo[occ]|cluster[occ]')
            cy_vir_a, cy_vir_b = y.get_overlap('mo[vir]|cluster[vir]')
            # Overlap between cluster-x and cluster-y:
            rxy_occ_a, rxy_occ_b = np.dot(cx_occ_a.T, cy_occ_a), np.dot(cx_occ_b.T, cy_occ_b)
            rxy_vir_a, rxy_vir_b = np.dot(cx_vir_a.T, cy_vir_a), np.dot(cx_vir_b.T, cy_vir_b)

            if svd_tol is not None:
                raise NotImplemented()
                def svd(a):
                    #nonlocal total_sv, kept_sv
                    u, s, v = np.linalg.svd(a, full_matrices=False)
                    if svd_tol is not None:
                        keep = (s >= svd_tol)
                        total_sv += len(keep)
                        kept_sv += sum(keep)
                        u, s, v = u[:,keep], s[keep], v[keep]
                    return u, s, v

                uxy_occ_a, sxy_occ_a, vxy_occ_a = svd(rxy_occ_a)
                uxy_vir_a, sxy_vir_a, vxy_vir_a = svd(rxy_vir_a)
                uxy_occ_a *= np.sqrt(sxy_occ_a)[np.newaxis,:]
                uxy_vir_a *= np.sqrt(sxy_vir_a)[np.newaxis,:]
                vxy_occ_a *= np.sqrt(sxy_occ_a)[:,np.newaxis]
                vxy_vir_a *= np.sqrt(sxy_vir_a)[:,np.newaxis]

                uxy_occ_b, sxy_occ_b, vxy_occ_b = svd(rxy_occ_b)
                uxy_vir_b, sxy_vir_b, vxy_vir_b = svd(rxy_vir_b)
                uxy_occ_b *= np.sqrt(sxy_occ_b)[np.newaxis,:]
                uxy_vir_b *= np.sqrt(sxy_vir_b)[np.newaxis,:]
                vxy_occ_b *= np.sqrt(sxy_occ_b)[:,np.newaxis]
                vxy_vir_b *= np.sqrt(sxy_vir_b)[:,np.newaxis]
            else:
                nsv_a = (min(rxy_occ_a.shape[0], rxy_occ_a.shape[1])
                     + min(rxy_vir_a.shape[0], rxy_vir_a.shape[1]))
                total_sv_a = kept_sv_a = (total_sv_a + nsv_a)
                nsv_b = (min(rxy_occ_b.shape[0], rxy_occ_b.shape[1])
                     + min(rxy_vir_b.shape[0], rxy_vir_b.shape[1]))
                total_sv_b = kept_sv_b = (total_sv_b + nsv_b)

            # --- If ovlp_tol is given, the cluster x-y pairs will be screened based on the
            # largest singular value of the occupied and virtual overlap matrices
            total_xy += 1
            if ovlp_tol is not None:
                if svd_tol:
                    rxy_occ_a_norm = (sxy_occ_a[0] if len(sxy_occ_a) > 0 else 0.0)
                    rxy_occ_b_norm = (sxy_occ_b[0] if len(sxy_occ_b) > 0 else 0.0)
                    rxy_vir_a_norm = (sxy_vir_a[0] if len(sxy_vir_a) > 0 else 0.0)
                    rxy_vir_b_norm = (sxy_vir_b[0] if len(sxy_vir_b) > 0 else 0.0)
                else:
                    rxy_occ_a_norm = np.linalg.norm(rxy_occ_a, ord=2)
                    rxy_occ_b_norm = np.linalg.norm(rxy_occ_b, ord=2)
                    rxy_vir_a_norm = np.linalg.norm(rxy_vir_a, ord=2)
                    rxy_vir_b_norm = np.linalg.norm(rxy_vir_b, ord=2)
                if (min(rxy_occ_a_norm, rxy_occ_b_norm, rxy_vir_a_norm, rxy_vir_b_norm) < ovlp_tol):
                    emb.log.debugv("Overlap of fragment pair %s - %s below %.2e; skipping pair.", x, y, ovlp_tol)
                    continue
            kept_xy += 1

            l1ya, l1yb = wfy.t1 if (t_as_lambda or y.solver == 'MP2') else wfy.l1
            l2aa, l2ab, l2bb = wfy.t2 if (t_as_lambda or y.solver == 'MP2') else wfy.l2

            if l2aa is None or l2ab is None or l2bb is None:
                raise RuntimeError("No L2 amplitudes found for %s!" % y)

            # Theta_jk^ab * l_ik^ab -> ij
            #doox -= einsum('jkab,IKAB,kK,aA,bB,QI->jQ', theta, l2, rxy_occ, rxy_vir, rxy_vir, cy_occ)
            ## Theta_ji^ca * l_ji^cb -> ab
            #dvvx += einsum('jica,JICB,jJ,iI,cC,QB->aQ', theta, l2, rxy_occ, rxy_occ, rxy_vir, cy_vir)

            # Only multiply with O(N)-scaling cy_occ/cy_vir in last step:

            if not late_t2_sym:
                if svd_tol is None:
#                     tlaa = einsum('(ijab,jJ,aA->iJAb),IJAB->iIbB', t2aa, rxy_occ_a, rxy_vir_a, l2aa)
#                     tlab = einsum('(ijab,jJ,aA->iJAb),IJAB->iIbB', t2ab, rxy_occ_b, rxy_vir_b, l2ab)
#                     tlbb = einsum('(ijab,jJ,aA->iJAb),IJAB->iIbB', t2bb, rxy_occ_b, rxy_vir_b, l2bb)

#                     tmpA = einsum('(ijab,jJ,aA,bB),(qI,IJAB)->iq', t2ab, rxy_occ_b, rxy_vir_a, rxy_vir_b, cy_occ_a, l2ab)
#                     tmpA += 0.5 * einsum('(ijab,jJ,aA,bB),(qI,IJAB)->iq', t2aa, rxy_occ_a, rxy_vir_a, rxy_vir_a, cy_occ_a, l2aa)

#                     tmpB  = einsum('(ijab,iI,aA,bB),(qJ,IJAB)->jq', t2ab, rxy_occ_a, rxy_vir_a, rxy_vir_b, cy_occ_b, l2ab)
#                     tmpB += 0.5 * einsum('(ijab,jJ,aA,bB),(qI,IJAB)->iq', t2bb, rxy_occ_b, rxy_vir_b, rxy_vir_b, cy_occ_b, l2bb)

#                     tmpC  = einsum('(ijab,iI,jJ,bB),(qA,IJAB)->aq', t2ab, rxy_occ_a, rxy_occ_b, rxy_vir_b, cy_vir_a, l2ab)
#                     tmpC += 0.5 * einsum('(ijab,iI,jJ,bB),(qA,IJAB)->aq', t2aa, rxy_occ_a, rxy_occ_a, rxy_vir_a, cy_vir_a, l2aa)

#                     tmpD  = einsum('(ijab,iI,jJ,aA),(qB,IJAB)->bq', t2ab, rxy_occ_a, rxy_occ_b, rxy_vir_a, cy_vir_b, l2ab)
#                     tmpD += 0.5 * einsum('(ijab,iI,jJ,bB),(qA,IJAB)->aq', t2bb, rxy_occ_b, rxy_occ_b, rxy_vir_b, cy_vir_b, l2bb)

#                     dooxa -= tmpA
#                     dooxb -= tmpB

#                     dvvxa += tmpC
#                     dvvxb += tmpD

#                     xt1xa += tmpA
#                     xt2xa += tmpC

#                     xt1xb += tmpB
#                     xt2xb += tmpD


#                     # OO block
#                     dooxa -= 0.5 * einsum('(ijab,jJ,aA,bB),(qI,IJAB)->iq', t2aa, rxy_occ_a, rxy_vir_a, rxy_vir_a, cy_occ_a, l2aa)
#                     dooxa -= einsum('(ijab,jJ,aA,bB),(qI,IJAB)->iq', t2ab, rxy_occ_b, rxy_vir_a, rxy_vir_b, cy_occ_a, l2ab)
#                     dooxb -= 0.5 * einsum('(ijab,jJ,aA,bB),(qI,IJAB)->iq', t2bb, rxy_occ_b, rxy_vir_b, rxy_vir_b, cy_occ_b, l2bb)
#                     dooxb -= einsum('(ijab,iI,aA,bB),(qJ,IJAB)->jq', t2ab, rxy_occ_a, rxy_vir_a, rxy_vir_b, cy_occ_b, l2ab)

#                     # VV block
#                     dvvxa += 0.5 * einsum('(ijab,iI,jJ,bB),(qA,IJAB)->aq', t2aa, rxy_occ_a, rxy_occ_a, rxy_vir_a, cy_vir_a, l2aa)
#                     dvvxa += einsum('(ijab,iI,jJ,bB),(qA,IJAB)->aq', t2ab, rxy_occ_a, rxy_occ_b, rxy_vir_b, cy_vir_a, l2ab)
#                     dvvxb += 0.5 * einsum('(ijab,iI,jJ,bB),(qA,IJAB)->aq', t2bb, rxy_occ_b, rxy_occ_b, rxy_vir_b, cy_vir_b, l2bb)
#                     dvvxb += einsum('(ijab,iI,jJ,aA),(qB,IJAB)->bq', t2ab, rxy_occ_a, rxy_occ_b, rxy_vir_a, cy_vir_b, l2ab)

#                     # VO block
#                     xt1xa += 0.5 * einsum('(ijab,jJ,aA,bB),(qI,IJAB)->iq', t2aa, rxy_occ_a, rxy_vir_a, rxy_vir_a, cy_occ_a, l2aa)
#                     xt1xa += einsum('(ijab,jJ,aA,bB),(qI,IJAB)->iq', t2ab, rxy_occ_b, rxy_vir_a, rxy_vir_b, cy_occ_a, l2ab)
#                     xt2xa += 0.5 * einsum('(ijab,iI,jJ,bB),(qA,IJAB)->aq', t2aa, rxy_occ_a, rxy_occ_a, rxy_vir_a, cy_vir_a, l2aa)
#                     xt2xa += einsum('(ijab,iI,jJ,bB),(qA,IJAB)->aq', t2ab, rxy_occ_a, rxy_occ_b, rxy_vir_b, cy_vir_a, l2ab)

#                     xt1xb += 0.5 * einsum('(ijab,jJ,aA,bB),(qI,IJAB)->iq', t2bb, rxy_occ_b, rxy_vir_b, rxy_vir_b, cy_occ_b, l2bb)
#                     xt1xb += einsum('(ijab,iI,aA,bB),(qJ,IJAB)->jq', t2ab, rxy_occ_a, rxy_vir_a, rxy_vir_b, cy_occ_b, l2ab)
#                     xt2xb += 0.5 * einsum('(ijab,iI,jJ,bB),(qA,IJAB)->aq', t2bb, rxy_occ_b, rxy_occ_b, rxy_vir_b, cy_vir_b, l2bb)
#                     xt2xb += einsum('(ijab,iI,jJ,aA),(qB,IJAB)->bq', t2ab, rxy_occ_a, rxy_occ_b, rxy_vir_a, cy_vir_b, l2ab)

                    dvoxa += einsum('(ijab,jJ,bB),JB->ai', t2aa, rxy_occ_a, rxy_vir_a, l1ya)
                    dvoxa += einsum('(ijab,jJ,bB),JB->ai', t2ab, rxy_occ_b, rxy_vir_b, l1yb)
                    dvoxb += einsum('(ijab,jJ,bB),JB->ai', t2bb, rxy_occ_b, rxy_vir_b, l1yb)
                    dvoxb += einsum('(ijab,iI,aA),IA->bj', t2ab, rxy_occ_a, rxy_vir_a, l1ya)

                else:
                    tmp = einsum('(ijab,jS,aP->iSPb),(SJ,PA,IJAB->ISPB)->iIbB', theta, uxy_occ, uxy_vir, vxy_occ, vxy_vir, l2)

                # tmpo = -einsum('iIbB,bB->iI', tmp, rxy_vir)
                # doox += np.dot(tmpo, cy_occ.T)
                # tmpv = einsum('iIbB,iI->bB', tmp, rxy_occ)
                # dvvx += np.dot(tmpv, cy_vir.T)
            else:
                raise NotImplemented()
                # Calculate some overlap matrices:
                cfx = x.get_overlap('cluster[occ]|frag')
                cfy = y.get_overlap('cluster[occ]|frag')
                mfx = x.get_overlap('mo[occ]|frag')
                mfy = y.get_overlap('mo[occ]|frag')
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
                    raise NotImplemented()
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
                    raise NotImplemented()
                    tmp += einsum('(xYab,aS->xYSb),(SA,YxBA->YxBS)->bB', t2tmp, uxy_vir, vxy_vir, l2tmp)/4
                    tmp += einsum('(xYba,aS->xYbS),(SA,YxAB->YxSB)->bB', t2tmp, uxy_vir, vxy_vir, l2tmp)/4
                dvvx += np.dot(tmp, cy_vir.T)

        dooa += np.dot(cx_occ_a, dooxa)
        doob += np.dot(cx_occ_b, dooxb)
        dvva += np.dot(cx_vir_a, dvvxa)
        dvvb += np.dot(cx_vir_b, dvvxb)
        xt1a += np.dot(cx_occ_a, xt1xa)
        xt1b += np.dot(cx_occ_b, xt1xb)
        xt2a += np.dot(cx_vir_a, xt2xa)
        xt2b += np.dot(cx_vir_b, xt2xb)
        dvoa += einsum('ij,jk,lk->il', cx_vir_a, dvoxa, cx_occ_a)
        dvob += einsum('ij,jk,lk->il', cx_vir_b, dvoxb, cx_occ_b)
        if use_sym:
            symtree = x.get_symmetry_tree()

        if with_t1:
            #l1x = dot(cx_occ.T, l1, cx_vir)
            if not late_t2_sym:
                pass
                #dovx = einsum('ijab,jb->ia', theta, l1x)
                #dov += einsum('ia,Ii,Aa->IA', dovx, cx_occ, cx_vir)
            else:
                cfx = x.get_overlap('cluster[occ]|frag')
                dovx1 = einsum('xjab,jb->xa', theta, l1x)/2
                dovx2 = einsum('xiba,(jx,jb->xb)->ia', theta, cfx, l1x)/2
                dov += dot(cx_occ, cfx, dovx1, cx_vir.T)
                dov += dot(cx_occ, dovx2, cx_vir.T)
            if use_sym:
                for x2, x2_children in symtree:
                    cx2_occ = x2.get_overlap('mo[occ]|cluster[occ]')
                    cx2_vir = x2.get_overlap('mo[vir]|cluster[vir]')
                    if not late_t2_sym:
                        dov += einsum('ia,Ii,Aa->IA', dovx, cx2_occ, cx2_vir)
                    else:
                        cfx2 = x2.get_overlap('cluster[occ]|frag')
                        dov += dot(cx2_occ, cfx2, dovx1, cx2_vir.T)
                        dov += dot(cx2_occ, dovx2, cx2_vir.T)
                    for x3, x3_children in x2_children:
                        assert not x3_children
                        cx3_occ = x3.get_overlap('mo[occ]|cluster[occ]')
                        cx3_vir = x3.get_overlap('mo[vir]|cluster[vir]')
                        if not late_t2_sym:
                            dov += einsum('ia,Ii,Aa->IA', dovx, cx3_occ, cx3_vir)
                        else:
                            cfx3 = x3.get_overlap('cluster[occ]|frag')
                            dov += dot(cx3_occ, cfx3, dovx1, cx3_vir.T)
                            dov += dot(cx3_occ, dovx2, cx3_vir.T)

        # --- Use symmetry of fragments (e.g. translations)
        if use_sym:
            # Transform right index of intermediates to AO basis:
            doox = dot(doox, emb.mo_coeff_occ.T)
            dvvx = dot(dvvx, emb.mo_coeff_vir.T)
            # Loop over symmetry children of x:
            for x2, x2_children in symtree:
                # Apply x->x2 symmetry to right index:
                doox2 = x2.sym_op(doox, axis=1)
                dvvx2 = x2.sym_op(dvvx, axis=1)
                # Symmetry is automatically applied to left index via `x2.cluster` orbitals
                doo += dot(cs_occ, x2.cluster.c_active_occ, doox2, cs_occ.T)
                dvv += dot(cs_vir, x2.cluster.c_active_vir, dvvx2, cs_vir.T)
                # TODO: scaling? precontraction of sym_op(sym_op(...)) ?
                for x3, x3_children in x2_children:
                    assert not x3_children
                    doox3 = x3.sym_op(doox2, axis=1)
                    dvvx3 = x3.sym_op(dvvx2, axis=1)
                    # Symmetry is automatically applied to left index via `x3.cluster` orbitals
                    doo += dot(cs_occ, x3.cluster.c_active_occ, doox3, cs_occ.T)
                    dvv += dot(cs_vir, x3.cluster.c_active_vir, dvvx3, cs_vir.T)

    if mpi:
        rma.clear()
        doo, dov, dvv = mpi.nreduce(doo, dov, dvv, target=mpi_target, logfunc=emb.log.timingv)
        # Make sure no more MPI calls are made after returning some ranks early!
        if mpi_target not in (None, mpi.rank):
            return None


    """
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
    """
    dooa -= einsum('ie,je->ij', l1a, t1a)
    doob -= einsum('ie,je->ij', l1b, t1b)

    dvva += einsum('ma,mb->ab', t1a, l1a)
    dvvb += einsum('ma,mb->ab', t1b, l1b)

    xt2a += einsum('ma,me->ae', t1a, l1a)
    dvoa -= einsum('mi,ma->ai', xt1a, t1a)
    dvoa -= einsum('ie,ae->ai', t1a, xt2a)
    dvoa += t1a.T

    xt2b += einsum('ma,me->ae', t1b, l1b)
    dvob -= einsum('mi,ma->ai', xt1b, t1b)
    dvob -= einsum('ie,ae->ai', t1b, xt2b)
    dvob += t1b.T

    dova = l1a
    dovb = l1b

    nmoa = nocca + nvira
    dm1a = np.zeros((nmoa,nmoa))
    dm1a[:nocca,:nocca] = dooa + dooa.conj().T
    dm1a[:nocca,nocca:] = (dova + dvoa.conj().T )
    dm1a[nocca:,:nocca] = (dm1a[:nocca,nocca:].conj().T)
    dm1a[nocca:,nocca:] = dvva + dvva.conj().T
    dm1a *= 0.5
    dm1a = (dm1a + dm1a.T)/2
    dm1a[np.diag_indices(nocca)] += 1

    nmob = noccb + nvirb
    dm1b = np.zeros((nmob, nmob))
    dm1b[:noccb,:noccb] = doob + doob.conj().T
    dm1b[:noccb,noccb:] = (dovb + dvob.conj().T )
    dm1b[noccb:,:noccb] = (dm1b[:noccb,noccb:].conj().T)
    dm1b[noccb:,noccb:] = dvvb + dvvb.conj().T
    dm1b *= 0.5
    dm1a = (dm1a + dm1a.T)/2
    dm1b[np.diag_indices(noccb)] += 1

    return dm1a, dm1b



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
        if t_as_lambda:
            l1, l2 = t1, t2
        else:
            l1 = emb.get_global_t1(get_lambda=True)
            l2 = emb.get_global_t2(get_lambda=True)
        mockcc = _get_mockcc(emb.mo_coeff, emb.mf.max_memory)
        #dm2 = pyscf.cc.uccsd_rdm.make_rdm2(mockcc, t1=t1, t2=t2, l1=l1, l2=l2, with_frozen=False, with_dm1=with_dm1)
        dm2 = uccsd_rdm.make_rdm2(mockcc, t1=t1, t2=t2, l1=l1, l2=l2, with_frozen=False, with_dm1=with_dm1)
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
    for x in emb.get_fragments(active=True, mpi_rank=mpi.rank):
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
