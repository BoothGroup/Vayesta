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
        ovlp = emb.get_ovlp()
        cs_occ_a = np.dot(emb.mo_coeff_occ[0].T, ovlp)
        cs_occ_b = np.dot(emb.mo_coeff_occ[1].T, ovlp)
        cs_vir_a = np.dot(emb.mo_coeff_vir[0].T, ovlp)
        cs_vir_b = np.dot(emb.mo_coeff_vir[1].T, ovlp)
    # Make projected WF available via remote memory access
    if mpi:
        # TODO: use L-amplitudes of cluster X and T-amplitudes,
        # Only send T-amplitudes via RMA?
        rma = {x.id: x.results.pwf.pack() for x in emb.get_fragments(active=True, mpi_rank=mpi.rank)}
        rma = mpi.create_rma_dict(rma)

    # --- Loop over pairs of fragments and add projected density-matrix contributions:
    dooa = np.zeros((nocca, nocca))
    doob = np.zeros((noccb, noccb))
    dvva = np.zeros((nvira, nvira))
    dvvb = np.zeros((nvirb, nvirb))

    if with_t1:
        dvoa = np.zeros((nvira, nocca))
        dvob = np.zeros((nvirb, noccb))

    xfilter = dict(sym_parent=None) if use_sym else {}
    for x in emb.get_fragments(active=True, mpi_rank=mpi.rank, **xfilter):
        wfx = x.results.pwf.as_ccsd()
        if not late_t2_sym:
            wfx = wfx.restore()

        t2aa = wfx.t2aa
        t2ab = wfx.t2ab
        t2bb = wfx.t2bb
        if late_t2_sym:
            t2ba = wfx.t2ba

        # Intermediates: leave left index in cluster-x basis:
        dooxa = np.zeros((x.cluster.nocc_active[0], emb.nocc[0]))
        dooxb = np.zeros((x.cluster.nocc_active[1], emb.nocc[1]))
        dvvxa = np.zeros((x.cluster.nvir_active[0], emb.nvir[0]))
        dvvxb = np.zeros((x.cluster.nvir_active[1], emb.nvir[1]))
        dvoxa = np.zeros((x.cluster.nvir_active[0], x.cluster.nocc_active[0]))
        dvoxb = np.zeros((x.cluster.nvir_active[1], x.cluster.nocc_active[1]))

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
                def svd(a,spin):
                    assert(spin == 'a' or spin == 'b')
                    if spin == 'a':
                        nonlocal total_sv_a, kept_sv_a
                    elif spin == 'b':
                        nonlocal total_sv_b, kept_sv_b
                    u, s, v = np.linalg.svd(a, full_matrices=False)
                    if svd_tol is not None:
                        keep = (s >= svd_tol)
                        if spin == 'a':
                            total_sv_a += len(keep)
                            kept_sv_a += sum(keep)
                        elif spin == 'b':
                            total_sv_b += len(keep)
                            kept_sv_b += sum(keep)
                        u, s, v = u[:,keep], s[keep], v[keep]
                    return u, s, v

                uxy_occ_a, sxy_occ_a, vxy_occ_a = svd(rxy_occ_a, 'a')
                uxy_vir_a, sxy_vir_a, vxy_vir_a = svd(rxy_vir_a, 'a')
                uxy_occ_a *= np.sqrt(sxy_occ_a)[np.newaxis,:]
                uxy_vir_a *= np.sqrt(sxy_vir_a)[np.newaxis,:]
                vxy_occ_a *= np.sqrt(sxy_occ_a)[:,np.newaxis]
                vxy_vir_a *= np.sqrt(sxy_vir_a)[:,np.newaxis]

                uxy_occ_b, sxy_occ_b, vxy_occ_b = svd(rxy_occ_b, 'b')
                uxy_vir_b, sxy_vir_b, vxy_vir_b = svd(rxy_vir_b, 'b')
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
            if not late_t2_sym:
                l2aa, l2ab, l2bb = wfy.t2 if (t_as_lambda or y.solver == 'MP2') else wfy.l2
            else:
                l2aa, l2ab, l2ba, l2bb = wfy.t2 if (t_as_lambda or y.solver == 'MP2') else wfy.l2

            if l2aa is None or l2ab is None or l2bb is None:
                raise RuntimeError("No L2 amplitudes found for %s!" % y)

            # Only multiply with O(N)-scaling cy_occ/cy_vir in last step:
            if not late_t2_sym:
                if svd_tol is None:
#                     # OO block
#                     dooxa -= einsum('(ijab,jJ,aA,bB),(qI,IJAB)->iq', t2aa, rxy_occ_a, rxy_vir_a, rxy_vir_a, cy_occ_a, l2aa) * 0.5
#                     dooxa -= einsum('(ijab,jJ,aA,bB),(qI,IJAB)->iq', t2ab, rxy_occ_b, rxy_vir_a, rxy_vir_b, cy_occ_a, l2ab)
#                     dooxb -= einsum('(ijab,jJ,aA,bB),(qI,IJAB)->iq', t2bb, rxy_occ_b, rxy_vir_b, rxy_vir_b, cy_occ_b, l2bb) * 0.5
#                     dooxb -= einsum('(ijab,iI,aA,bB),(qJ,IJAB)->jq', t2ab, rxy_occ_a, rxy_vir_a, rxy_vir_b, cy_occ_b, l2ab)

#                     # VV block
#                     dvvxa += einsum('(ijab,iI,jJ,bB),(qA,IJAB)->aq', t2aa, rxy_occ_a, rxy_occ_a, rxy_vir_a, cy_vir_a, l2aa) * 0.5
#                     dvvxa += einsum('(ijab,iI,jJ,bB),(qA,IJAB)->aq', t2ab, rxy_occ_a, rxy_occ_b, rxy_vir_b, cy_vir_a, l2ab)
#                     dvvxb += einsum('(ijab,iI,jJ,bB),(qA,IJAB)->aq', t2bb, rxy_occ_b, rxy_occ_b, rxy_vir_b, cy_vir_b, l2bb) * 0.5
#                     dvvxb += einsum('(ijab,iI,jJ,aA),(qB,IJAB)->bq', t2ab, rxy_occ_a, rxy_occ_b, rxy_vir_a, cy_vir_b, l2ab)

                    # -------------------------------

                    tlaa = einsum('(ijab,jJ,bB->iJaB),IJAB->iIaA', t2aa, rxy_occ_a, rxy_vir_a, l2aa)
                    tlbb = einsum('(ijab,jJ,bB->iJaB),IJAB->iIaA', t2bb, rxy_occ_b, rxy_vir_b, l2bb)

                    tlab1 = einsum('(ijab,jJ,bB->iJaB),IJAB->iIaA', t2ab, rxy_occ_b, rxy_vir_b, l2ab)
                    tlab2 = einsum('(ijab,iI,aA->IjAb),IJAB->jJbB', t2ab, rxy_occ_a, rxy_vir_a, l2ab)

                else:
#                     # OO block
#                     dooxa -= 0.5 * einsum('(ijab,jX,aY,bZ),(XJ,YA,ZB,qI,IJAB)->iq', t2aa, uxy_occ_a, uxy_vir_a, uxy_vir_a, vxy_occ_a, vxy_vir_a, vxy_vir_a, cy_occ_a, l2aa)
#                     dooxa -= einsum('(ijab,jX,aY,bZ),(XJ,YA,ZB,qI,IJAB)->iq', t2ab, uxy_occ_b, uxy_vir_a, uxy_vir_b, vxy_occ_b, vxy_vir_a, vxy_vir_b, cy_occ_a, l2ab)
#                     dooxb -= 0.5 * einsum('(ijab,jX,aY,bZ),(XJ,YA,ZB,qI,IJAB)->iq', t2bb, uxy_occ_b, uxy_vir_b, uxy_vir_b, vxy_occ_b, vxy_vir_b, vxy_vir_b, cy_occ_b, l2bb)
#                     dooxb -= einsum('(ijab,iX,aY,bZ),(XI,YA,ZB,qJ,IJAB)->jq', t2ab, uxy_occ_a, uxy_vir_a, uxy_vir_b, vxy_occ_a, vxy_vir_a, vxy_vir_b, cy_occ_b, l2ab)

#                     # VV block
#                     dvvxa += 0.5 * einsum('(ijab,iX,jY,bZ),(XI,YJ,ZB,qA,IJAB)->aq', t2aa, uxy_occ_a, uxy_occ_a, uxy_vir_a, vxy_occ_a, vxy_occ_a, vxy_vir_a, cy_vir_a, l2aa)
#                     dvvxa += einsum('(ijab,iX,jY,bZ),(XI,YJ,ZB,qA,IJAB)->aq', t2ab, uxy_occ_a, uxy_occ_b, uxy_vir_b, vxy_occ_a, vxy_occ_b, vxy_vir_b, cy_vir_a, l2ab)
#                     dvvxb += 0.5 * einsum('(ijab,iX,jY,bZ),(XI,YJ,ZB,qA,IJAB)->aq', t2bb, uxy_occ_b, uxy_occ_b, uxy_vir_b, vxy_occ_b, vxy_occ_b, vxy_vir_b, cy_vir_b, l2bb)
#                     dvvxb += einsum('(ijab,iX,jY,aZ),(XI,YJ,ZA,qB,IJAB)->bq', t2ab, uxy_occ_a, uxy_occ_b, uxy_vir_a, vxy_occ_a, vxy_occ_b, vxy_vir_a, cy_vir_b, l2ab)

                    #--------------------------------

                    tlaa = einsum('(ijab,jX,bY->iXaY),(XJ,YB,IJAB)->iIaA', t2aa, uxy_occ_a, uxy_vir_a, vxy_occ_a, vxy_vir_a, l2aa)
                    tlbb = einsum('(ijab,jX,bY->iXaY),(XJ,YB,IJAB)->iIaA', t2bb, uxy_occ_b, uxy_vir_b, vxy_occ_b, vxy_vir_b, l2bb)

                    tlab1 = einsum('(ijab,jX,bY->iXaY),(XJ,YB,IJAB)->iIaA', t2ab, uxy_occ_b, uxy_vir_b, vxy_occ_b, vxy_vir_b, l2ab)
                    tlab2 = einsum('(ijab,iX,aY->XjYb),(XI,YA,IJAB)->jJbB', t2ab, uxy_occ_a, uxy_vir_a, vxy_occ_a, vxy_vir_a, l2ab)

                # OO block
                dooxa -= einsum('iIaA,aA,qI->iq', tlaa, rxy_vir_a, cy_occ_a) * 0.5
                dooxa -= einsum('iIaA,aA,qI->iq', tlab1, rxy_vir_a, cy_occ_a)
                dooxb -= einsum('iIaA,aA,qI->iq', tlbb, rxy_vir_b, cy_occ_b) * 0.5
                dooxb -= einsum('iIaA,aA,qI->iq', tlab2, rxy_vir_b, cy_occ_b)

                # VV block
                dvvxa += einsum('iIaA,iI,qA->aq', tlaa, rxy_occ_a, cy_vir_a) * 0.5
                dvvxa += einsum('iIaA,iI,qA->aq', tlab1, rxy_occ_a, cy_vir_a)
                dvvxb += einsum('iIaA,iI,qA->aq', tlbb, rxy_occ_b, cy_vir_b) * 0.5
                dvvxb += einsum('iIaA,iI,qA->aq', tlab2, rxy_occ_b, cy_vir_b)

            else:
                # Calculate some overlap matrices:
                cfxa, cfxb = x.get_overlap('cluster[occ]|frag')
                cfya, cfyb = y.get_overlap('cluster[occ]|frag')
                mfxa, mfxb = x.get_overlap('mo[occ]|frag')
                mfya, mfyb = y.get_overlap('mo[occ]|frag')
                if svd_tol is None:
                    cfxy_occ_a = dot(rxy_occ_a, cfya)
                    cfxy_occ_b = dot(rxy_occ_b, cfyb)
                    cfyx_occ_a = dot(rxy_occ_a.T, cfxa)
                    cfyx_occ_b = dot(rxy_occ_b.T, cfxb)
                    ffxya = dot(cfxa.T, rxy_occ_a, cfya)
                    ffxyb = dot(cfxb.T, rxy_occ_b, cfyb)
                else:
                    cfxy_occ_a = dot(uxy_occ_a, vxy_occ_a, cfya)
                    cfxy_occ_b = dot(uxy_occ_b, vxy_occ_b, cfyb)
                    cfyx_occ_a = dot(vxy_occ_a.T, uxy_occ_a.T, cfxa)
                    cfyx_occ_b = dot(vxy_occ_b.T, uxy_occ_b.T, cfxb)
                    ffxya = dot(cfxa.T, uxy_occ_a, vxy_occ_a, cfya)
                    ffxyb = dot(cfxb.T, uxy_occ_b, vxy_occ_b, cfyb)

                # --- Occupied
                # Deal with both virtual overlaps here:
                if svd_tol is None:
                    t2aatmp = einsum('xjab,aA,bB->xjAB', t2aa, rxy_vir_a, rxy_vir_a) # frag * cluster^4
                    t2abtmp = einsum('xjab,aA,bB->xjAB', t2ab, rxy_vir_a, rxy_vir_b)
                    t2batmp = einsum('xjab,aA,bB->xjAB', t2ba, rxy_vir_b, rxy_vir_a)
                    t2bbtmp = einsum('xjab,aA,bB->xjAB', t2bb, rxy_vir_b, rxy_vir_b)
                    l2aatmp, l2abtmp, l2batmp, l2bbtmp = l2aa, l2ab, l2ba, l2bb
                else:
                    t2aatmp = einsum('xjab,aS,bP->xjSP', t2aa, uxy_vir_a, uxy_vir_a)
                    t2abtmp = einsum('xjab,aS,bP->xjSP', t2ab, uxy_vir_a, uxy_vir_b)
                    t2batmp = einsum('xjab,aS,bP->xjSP', t2ba, uxy_vir_b, uxy_vir_a)
                    t2bbtmp = einsum('xjab,aS,bP->xjSP', t2bb, uxy_vir_b, uxy_vir_b)

                    l2aatmp = einsum('yjab,Sa,Pb->yjSP', l2aa, vxy_vir_a, vxy_vir_a)
                    l2abtmp = einsum('yjab,Sa,Pb->yjSP', l2ab, vxy_vir_a, vxy_vir_b)
                    l2batmp = einsum('yjab,Sa,Pb->yjSP', l2ba, vxy_vir_b, vxy_vir_a)
                    l2bbtmp = einsum('yjab,Sa,Pb->yjSP', l2bb, vxy_vir_b, vxy_vir_b)

                # T2 * L2
                if svd_tol is None:
                    tmpa  = -einsum('(xjAB,jJ->xJAB),YJAB->xY', t2aatmp, rxy_occ_a, l2aatmp)/8
                    tmpa += -einsum('(xjAB,jJ->xJAB),YJAB->xY', t2abtmp, rxy_occ_b, l2abtmp)/4
                    tmpb  = -einsum('(xjAB,jJ->xJAB),YJAB->xY', t2bbtmp, rxy_occ_b, l2bbtmp)/8
                    tmpb += -einsum('(xjAB,jJ->xJAB),YJAB->xY', t2batmp, rxy_occ_a, l2batmp)/4
                else:
                    tmpa  = -einsum('(xjAB,jS->xSAB),(SJ,YJAB->YSAB)->xY', t2aatmp, uxy_occ_a, vxy_occ_a, l2aatmp)/8
                    tmpa += -einsum('(xjAB,jS->xSAB),(SJ,YJAB->YSAB)->xY', t2abtmp, uxy_occ_b, vxy_occ_b, l2abtmp)/4
                    tmpb  = -einsum('(xjAB,jS->xSAB),(SJ,YJAB->YSAB)->xY', t2bbtmp, uxy_occ_b, vxy_occ_b, l2bbtmp)/8
                    tmpb += -einsum('(xjAB,jS->xSAB),(SJ,YJAB->YSAB)->xY', t2batmp, uxy_occ_a, vxy_occ_a, l2batmp)/4

                dooxa += dot(cfxa, tmpa, mfya.T)
                dooxb += dot(cfxb, tmpb, mfyb.T)
                # T2 * L2.T
                tmpa  = -einsum('(xjAB,jY->xYAB),YIBA->xI', t2aatmp, cfxy_occ_a, l2aatmp)/8
                tmpa += -einsum('(xjAB,jY->xYAB),YIBA->xI', t2abtmp, cfxy_occ_b, l2batmp)/4
                dooxa += dot(cfxa, tmpa, cy_occ_a.T)

                tmpb  = -einsum('(xjAB,jY->xYAB),YIBA->xI', t2bbtmp, cfxy_occ_b, l2bbtmp)/8
                tmpb += -einsum('(xjAB,jY->xYAB),YIBA->xI', t2batmp, cfxy_occ_a, l2abtmp)/4
                dooxb += dot(cfxb, tmpb, cy_occ_b.T)
                # T2.T * L2
                tmpa  = -einsum('xiBA,(Jx,YJAB->YxAB)->iY', t2aatmp, cfyx_occ_a, l2aatmp)/8
                tmpa += -einsum('xiBA,(Jx,YJAB->YxAB)->iY', t2batmp, cfyx_occ_b, l2abtmp)/4
                dooxa += np.dot(tmpa, mfya.T)

                tmpb  = -einsum('xiBA,(Jx,YJAB->YxAB)->iY', t2bbtmp, cfyx_occ_b, l2bbtmp)/8
                tmpb += -einsum('xiBA,(Jx,YJAB->YxAB)->iY', t2abtmp, cfyx_occ_a, l2batmp)/4
                dooxb += np.dot(tmpb, mfyb.T)

                # T2.T * L2.T
                tmpa  = -einsum('xiBA,xY,YIBA->iI', t2aatmp, ffxya, l2aatmp)/8
                tmpa += -einsum('xiBA,xY,YIBA->iI', t2batmp, ffxyb, l2batmp)/4
                dooxa += np.dot(tmpa, cy_occ_a.T)

                tmpb  = -einsum('xiBA,xY,YIBA->iI', t2bbtmp, ffxyb, l2bbtmp)/8
                tmpb += -einsum('xiBA,xY,YIBA->iI', t2abtmp, ffxya, l2abtmp)/4
                dooxb += np.dot(tmpb, cy_occ_b.T)

                # --- Virtual
                # T2 * L2 and T2.T * L2.T
                if svd_tol is None:
                    t2aatmp = einsum('xjab,xY,jJ->YJab', t2aa, ffxya, rxy_occ_a)
                    t2abtmp = einsum('xjab,xY,jJ->YJab', t2ab, ffxya, rxy_occ_b)
                    t2batmp = einsum('xjab,xY,jJ->YJab', t2ba, ffxyb, rxy_occ_a)
                    t2bbtmp = einsum('xjab,xY,jJ->YJab', t2bb, ffxyb, rxy_occ_b)

                    tmpa  = einsum('(YJab,aA->YJAb),YJAB->bB', t2aatmp, rxy_vir_a, l2aa)/8
                    tmpa += einsum('(YJab,aA->YJAb),YJAB->bB', t2batmp, rxy_vir_b, l2ba)/4
                    tmpb  = einsum('(YJab,aA->YJAb),YJAB->bB', t2bbtmp, rxy_vir_b, l2bb)/8
                    tmpb += einsum('(YJab,aA->YJAb),YJAB->bB', t2abtmp, rxy_vir_a, l2ab)/4


                    tmpa += einsum('(YIba,aA->YIbA),YIBA->bB', t2aatmp, rxy_vir_a, l2aa)/8
                    tmpa += einsum('(YIba,aA->YIbA),YIBA->bB', t2abtmp, rxy_vir_b, l2ab)/4
                    tmpb += einsum('(YIba,aA->YIbA),YIBA->bB', t2bbtmp, rxy_vir_b, l2bb)/8
                    tmpb += einsum('(YIba,aA->YIbA),YIBA->bB', t2batmp, rxy_vir_a, l2ba)/4
                else:
                    t2aatmp = einsum('xjab,jS->xSab', t2aa, uxy_occ_a)
                    t2abtmp = einsum('xjab,jS->xSab', t2ab, uxy_occ_b)
                    t2batmp = einsum('xjab,jS->xSab', t2ba, uxy_occ_a)
                    t2bbtmp = einsum('xjab,jS->xSab', t2bb, uxy_occ_b)

                    l2aatmp = einsum('YJAB,SJ,xY->xSAB', l2aa, vxy_occ_a, ffxya)
                    l2abtmp = einsum('YJAB,SJ,xY->xSAB', l2ab, vxy_occ_b, ffxya)
                    l2batmp = einsum('YJAB,SJ,xY->xSAB', l2ba, vxy_occ_a, ffxyb)
                    l2bbtmp = einsum('YJAB,SJ,xY->xSAB', l2bb, vxy_occ_b, ffxyb)

                    tmpa  = einsum('(xSab,aP->xSPb),(PA,xSAB->xSPB)->bB', t2aatmp, uxy_vir_a, vxy_vir_a, l2aatmp)/8
                    tmpa += einsum('(xSab,aP->xSPb),(PA,xSAB->xSPB)->bB', t2batmp, uxy_vir_b, vxy_vir_b, l2batmp)/4
                    tmpb  = einsum('(xSab,aP->xSPb),(PA,xSAB->xSPB)->bB', t2bbtmp, uxy_vir_b, vxy_vir_b, l2bbtmp)/8
                    tmpb += einsum('(xSab,aP->xSPb),(PA,xSAB->xSPB)->bB', t2abtmp, uxy_vir_a, vxy_vir_a, l2abtmp)/4


                    tmpa += einsum('(xSba,aP->xSbP),(PA,xSBA->xSBP)->bB', t2aatmp, uxy_vir_a, vxy_vir_a, l2aatmp)/8
                    tmpa += einsum('(xSba,aP->xSbP),(PA,xSBA->xSBP)->bB', t2abtmp, uxy_vir_b, vxy_vir_b, l2abtmp)/4
                    tmpb += einsum('(xSba,aP->xSbP),(PA,xSBA->xSBP)->bB', t2bbtmp, uxy_vir_b, vxy_vir_b, l2bbtmp)/8
                    tmpb += einsum('(xSba,aP->xSbP),(PA,xSBA->xSBP)->bB', t2batmp, uxy_vir_a, vxy_vir_a, l2batmp)/4

                # T2 * L2.T and T2.T * L2
                t2aatmp = einsum('xjab,jY->xYab', t2aa, cfxy_occ_a)
                t2abtmp = einsum('xjab,jY->xYab', t2ab, cfxy_occ_b)
                t2batmp = einsum('xjab,jY->xYab', t2ba, cfxy_occ_a)
                t2bbtmp = einsum('xjab,jY->xYab', t2bb, cfxy_occ_b)

                l2aatmp = einsum('Jx,YJAB->YxAB', cfyx_occ_a, l2aa)
                l2abtmp = einsum('Jx,YJAB->YxAB', cfyx_occ_b, l2ab)
                l2batmp = einsum('Jx,YJAB->YxAB', cfyx_occ_a, l2ba)
                l2bbtmp = einsum('Jx,YJAB->YxAB', cfyx_occ_b, l2bb)

                if svd_tol is None:
                    tmpa += einsum('(xYab,aA->xYAb),YxBA->bB', t2aatmp, rxy_vir_a, l2aatmp)/8
                    tmpa += einsum('(xYab,aA->xYAb),YxBA->bB', t2batmp, rxy_vir_b, l2abtmp)/4
                    tmpb += einsum('(xYab,aA->xYAb),YxBA->bB', t2bbtmp, rxy_vir_b, l2bbtmp)/8
                    tmpb += einsum('(xYab,aA->xYAb),YxBA->bB', t2abtmp, rxy_vir_a, l2batmp)/4

                    tmpa += einsum('(xYba,aA->xYbA),YxAB->bB', t2aatmp, rxy_vir_a, l2aatmp)/8
                    tmpa += einsum('(xYba,aA->xYbA),YxAB->bB', t2abtmp, rxy_vir_b, l2batmp)/4
                    tmpb += einsum('(xYba,aA->xYbA),YxAB->bB', t2bbtmp, rxy_vir_b, l2bbtmp)/8
                    tmpb += einsum('(xYba,aA->xYbA),YxAB->bB', t2batmp, rxy_vir_a, l2abtmp)/4
                else:
                    tmpa += einsum('(xYab,aS->xYSb),(SA,YxBA->YxBS)->bB', t2aatmp, uxy_vir_a, vxy_vir_a, l2aatmp)/8
                    tmpa += einsum('(xYab,aS->xYSb),(SA,YxBA->YxBS)->bB', t2batmp, uxy_vir_b, vxy_vir_b, l2abtmp)/4
                    tmpb += einsum('(xYab,aS->xYSb),(SA,YxBA->YxBS)->bB', t2bbtmp, uxy_vir_b, vxy_vir_b, l2bbtmp)/8
                    tmpb += einsum('(xYab,aS->xYSb),(SA,YxBA->YxBS)->bB', t2abtmp, uxy_vir_a, vxy_vir_a, l2batmp)/4


                    tmpa += einsum('(xYba,aS->xYbS),(SA,YxAB->YxSB)->bB', t2aatmp, uxy_vir_a, vxy_vir_a, l2aatmp)/8
                    tmpa += einsum('(xYba,aS->xYbS),(SA,YxAB->YxSB)->bB', t2abtmp, uxy_vir_b, vxy_vir_b, l2batmp)/4
                    tmpb += einsum('(xYba,aS->xYbS),(SA,YxAB->YxSB)->bB', t2bbtmp, uxy_vir_b, vxy_vir_b, l2bbtmp)/8
                    tmpb += einsum('(xYba,aS->xYbS),(SA,YxAB->YxSB)->bB', t2batmp, uxy_vir_a, vxy_vir_a, l2abtmp)/4

                dvvxa += np.dot(tmpa, cy_vir_a.T)
                dvvxb += np.dot(tmpb, cy_vir_b.T)
        dooa += np.dot(cx_occ_a, dooxa)
        doob += np.dot(cx_occ_b, dooxb)
        dvva += np.dot(cx_vir_a, dvvxa)
        dvvb += np.dot(cx_vir_b, dvvxb)

        # --- Use symmetry of fragments (rotations and translations)
        if use_sym:
            # Transform right index of intermediates to AO basis:
            dooxa = dot(dooxa, emb.mo_coeff_occ[0].T)
            dooxb = dot(dooxb, emb.mo_coeff_occ[1].T)
            dvvxa = dot(dvvxa, emb.mo_coeff_vir[0].T)
            dvvxb = dot(dvvxb, emb.mo_coeff_vir[1].T)
            #Loop over symmetry children of x:
            for x2, (cx2_occ_a, cx2_occ_b, cx2_vir_a, cx2_vir_b, dooxa2, dooxb2, dvvxa2, dvvxb2) in x.loop_symmetry_children(
                    (x.cluster.c_occ[0], x.cluster.c_occ[1], x.cluster.c_vir[0], x.cluster.c_vir[1], dooxa, dooxb, dvvxa, dvvxb), axes=[0,0,0,0,1,1,1,1]):
                dooa += dot(cs_occ_a, cx2_occ_a, dooxa2, cs_occ_a.T)
                doob += dot(cs_occ_b, cx2_occ_b, dooxb2, cs_occ_b.T)
                dvva += dot(cs_vir_a, cx2_vir_a, dvvxa2, cs_vir_a.T)
                dvvb += dot(cs_vir_b, cx2_vir_b, dvvxb2, cs_vir_b.T)

        if with_t1:
            l1xa = dot(cx_occ_a.T, l1a, cx_vir_a)
            l1xb = dot(cx_occ_b.T, l1b, cx_vir_b)
            if not late_t2_sym:
                # VO block
                dvoxa += einsum('ijab,jb->ai', t2aa, l1xa)
                dvoxa += einsum('ijab,jb->ai', t2ab, l1xb)
                dvoxb += einsum('ijab,jb->ai', t2bb, l1xb)
                dvoxb += einsum('ijab,ia->bj', t2ab, l1xa)

                dvoa += einsum('ai,Ii,Aa->AI', dvoxa, cx_occ_a, cx_vir_a)
                dvob += einsum('ai,Ii,Aa->AI', dvoxb, cx_occ_b, cx_vir_b)

            else:
                cfxa, cfxb = x.get_overlap('cluster[occ]|frag')
                dvoxa1  = einsum('xjab,jb->ax', t2aa, l1xa)/2
                dvoxa1 += einsum('xjab,jb->ax', t2ab, l1xb)/2
                dvoxa2  = einsum('xiba,(jx,jb->xb)->ai', t2aa, cfxa, l1xa)/2
                dvoxa2 += einsum('xiba,(jx,jb->xb)->ai', t2ba, cfxb, l1xb)/2
                dvoa += dot(cx_vir_a, dvoxa1, cfxa.T, cx_occ_a.T)
                dvoa += dot(cx_vir_a, dvoxa2, cx_occ_a.T)

                dvoxb1  = einsum('xjab,jb->ax', t2bb, l1xb)/2
                dvoxb1 += einsum('xjab,jb->ax', t2ba, l1xa)/2
                dvoxb2  = einsum('xiba,(jx,jb->xb)->ai', t2bb, cfxb, l1xb)/2
                dvoxb2 += einsum('xiba,(jx,jb->xb)->ai', t2ab, cfxa, l1xa)/2
                dvob += dot(cx_vir_b, dvoxb1, cfxb.T, cx_occ_b.T)
                dvob += dot(cx_vir_b, dvoxb2, cx_occ_b.T)

            if use_sym:
                # TODO: Use only one loop
                for x2, (cx2_frag_a, cx2_occ_a, cx2_vir_a, cx2_frag_b, cx2_occ_b, cx2_vir_b) in x.loop_symmetry_children(
                    (x.c_frag[0], x.cluster.c_occ[0], x.cluster.c_vir[0], x.c_frag[1], x.cluster.c_occ[1], x.cluster.c_vir[1]), include_self=False, maxgen=None):
                    cx2_occ_a = np.dot(cs_occ_a, cx2_occ_a)
                    cx2_vir_a = np.dot(cs_vir_a, cx2_vir_a)
                    cx2_occ_b = np.dot(cs_occ_b, cx2_occ_b)
                    cx2_vir_b = np.dot(cs_vir_b, cx2_vir_b)
                    if not late_t2_sym:
                        dvoa += einsum('ai,Ii,Aa->AI', dvoxa, cx2_occ_a, cx2_vir_a)
                        dvob += einsum('ai,Ii,Aa->AI', dvoxb, cx2_occ_b, cx2_vir_b)
                    else:
                        dvoa += dot(cx2_vir_a, dvoxa1, cx2_frag_a.T, cs_occ_a.T)
                        dvoa += dot(cx2_vir_a, dvoxa2, cx2_occ_a.T)
                        dvob += dot(cx2_vir_b, dvoxb1, cx2_frag_b.T, cs_occ_b.T)
                        dvob += dot(cx2_vir_b, dvoxb2, cx2_occ_b.T)

    if mpi:
        rma.clear()
        dooa, doob, dvoa, dvob, dvva, dvvb = mpi.nreduce(dooa, doob, dvoa, dvob, dvva, dvvb, target=mpi_target, logfunc=emb.log.timingv)
        # Make sure no more MPI calls are made after returning some ranks early!
        if mpi_target not in (None, mpi.rank):
            return None

    if with_t1:
        xt1a = -dooa.T.copy()
        xt1b = -doob.T.copy()
        xt2a = dvva.copy()
        xt2b = dvvb.copy()

        dooa -= einsum('ie,je->ij', l1a, t1a)
        doob -= einsum('ie,je->ij', l1b, t1b)

        dvva += einsum('ma,mb->ab', t1a, l1a)
        dvvb += einsum('ma,mb->ab', t1b, l1b)

        xt2a += einsum('ma,me->ae', t1a, l1a)
        dvoa -= einsum('mi,ma->ai', xt1a, t1a)
        dvoa -= einsum('ie,ae->ai', t1a, xt2a)
        dvoa += t1a.T + l1a.T

        xt2b += einsum('ma,me->ae', t1b, l1b)
        dvob -= einsum('mi,ma->ai', xt1b, t1b)
        dvob -= einsum('ie,ae->ai', t1b, xt2b)
        dvob += t1b.T + l1b.T

    nmoa = nocca + nvira
    dm1a = np.zeros((nmoa,nmoa))
    dm1a[:nocca,:nocca] = dooa + dooa.conj().T
    dm1a[nocca:,nocca:] = dvva + dvva.conj().T
    if with_t1:
        dm1a[:nocca,nocca:] = dvoa.conj().T
        dm1a[nocca:,:nocca] = (dm1a[:nocca,nocca:].conj().T)
    dm1a = (dm1a + dm1a.T)/4

    nmob = noccb + nvirb
    dm1b = np.zeros((nmob, nmob))
    dm1b[:noccb,:noccb] = doob + doob.conj().T
    dm1b[noccb:,noccb:] = dvvb + dvvb.conj().T
    if with_t1:
        dm1b[:noccb,noccb:] =  dvob.conj().T
        dm1b[noccb:,:noccb] = (dm1b[:noccb,noccb:].conj().T)
    dm1b = (dm1b + dm1b.T)/4

    if with_mf:
        dm1a[np.diag_indices(nocca)] += 1
        dm1b[np.diag_indices(noccb)] += 1

    if ao_basis:
        dm1a = dot(emb.mo_coeff[0], dm1a, emb.mo_coeff[0].T)
        dm1b = dot(emb.mo_coeff[1], dm1b, emb.mo_coeff[1].T)

    # --- Some information:
    emb.log.debug("Cluster-pairs: total= %d  kept= %d (%.1f%%)", total_xy, kept_xy, 100*kept_xy/total_xy)
    emb.log.debug("Singular values (α) total= %d  kept= %d (%.1f%%)", total_sv_a, kept_sv_a, 100*kept_sv_a/total_sv_a)
    emb.log.debug("Singular values (β) total= %d  kept= %d (%.1f%%)", total_sv_b, kept_sv_b, 100*kept_sv_b/total_sv_b)

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
