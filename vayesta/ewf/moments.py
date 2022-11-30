"""Routines to generate CCSD Green's function  moments () from spin-restricted embedded CCSD calculations."""

import numpy as np

import pyscf
import pyscf.cc

from vayesta.core.util import *
from vayesta.ewf.rdm import _get_mockcc
from vayesta.mpi import mpi


def get_global_ip_bra(emb, slow=True, t_as_lambda=False):

    if t_as_lambda is None:
        t_as_lambda = emb.opts.t_as_lambda

    if slow:
        t1 = emb.get_global_t1()
        t2 = emb.get_global_t2()
        l1 = t1 if t_as_lambda else emb.get_global_l1()
        l2 = t2 if t_as_lambda else emb.get_global_l2()

        nocc, nvir = t1.shape
        nmo = nocc + nvir

        ei_o = np.eye(nocc) - np.dot(t1, l1.T) - einsum('imab,pmab->pi', l2, (2*t2 - t2.transpose(0,1,3,2))) #problem term
        ei_v = l1.T

        eija_o = einsum('ijea,pe->pija', (l2.transpose(1,0,2,3) - 2*l2), t1) + 2*einsum('ja,pi->pija', l1, np.eye(nocc)) - einsum('ia,pj->pija', l1, np.eye(nocc))
        eija_v = 2*l2.transpose(2,0,1,3) - l2.transpose(3,0,1,2)

        ei = np.concatenate((ei_o, ei_v))
        eija = np.concatenate((eija_o, eija_v))

        return ei, eija

    ovlp = emb.get_ovlp()
    cs_occ = np.dot(emb.mo_coeff_occ.T, ovlp)
    cs_vir = np.dot(emb.mo_coeff_vir.T, ovlp)

    t1g = emb.get_global_t1()
    l1g = t1 if t_as_lambda else emb.get_global_l1()

    nocc, nvir = cs_occ.shape[0], cs_vir.shape[0]
    nmo = nocc + nvir

    #ei = np.zeros((nmo, nocc))
    ei_o = np.eye(nocc) -  einsum('ia,ja->ij', t1g, l1g)
    ei_v = l1g.T#np.zeros((nvir, nocc))
    eija_o = np.zeros((nocc , nocc, nocc, nvir))
    eija_v = np.zeros((nvir , nocc, nocc, nvir))

    symfilter = {}#dict(sym_parent=None) if use_sym else {}
    for fx in emb.get_fragments(active=True, mpi_rank=mpi.rank, **symfilter):

        cx_occ = fx.get_overlap('mo[occ]|cluster[occ]')
        cx_vir = fx.get_overlap('mo[vir]|cluster[vir]')
        cx = fx.get_overlap('mo|cluster')
        cfx = fx.get_overlap('cluster[occ]|frag')
        #mfx = fx.get_overlap('mo[occ]|frag')
        cfc = cfx.dot(cfx.T)
        # No late symmetrisation

        #t1 = einsum('iI,aA,ia->IA', cx_occ, cx_vir, t1g)
        #l1 = einsum('iI,aA,ia->IA', cx_occ, cx_vir, l1g)

        wfx = fx.results.pwf.as_ccsd().restore()
        t2 = wfx.t2
        l2 = t2 if t_as_lambda else wfx.l2

        t1 = wfx.t1
        l1 = t1 if t_as_lambda else wfx.l1

        nocc, nvir = t1.shape
        nmo = nocc + nvir

        #ei_o += cfc #- einsum('ia,ja->ij', t1, l1)#0.5*einsum('xi,ia,ja->xj', cfc, t1, l1) - 0.5*einsum('xj,ia,ja->ix', cfc, t1, l1) #- einsum('imab,pmab->pi', l2, (2*t2 - t2.transpose(0,1,3,2))) #problem term
        ei_o_x = np.zeros((nocc, ei_o.shape[0]))
        eija_o_x = np.zeros(( ei_o.shape[0], nocc, nocc, nvir))

        for fy in emb.get_fragments(active=True, mpi_rank=mpi.rank, **symfilter):
            #cy_occ = np.dot(cs_occ, cy_occ_ao)
            #cy_vir = np.dot(cs_vir, cy_vir_ao)
            cy_occ = fy.get_overlap('mo[occ]|cluster[occ]')
            cy_vir = fy.get_overlap('mo[vir]|cluster[vir]')
            cfy = fy.get_overlap('cluster[occ]|frag')
            cy = fy.get_overlap('mo|cluster')
            # Overlap between cluster x and cluster y:
            rxy_occ = np.dot(cx_occ.T, cy_occ)
            rxy_vir = np.dot(cx_vir.T, cy_vir)
            #mfy = np.dot(cs_occ, cy_frag)



            wfy = fy.results.pwf.as_ccsd().restore()
            l2y = wfy.t2 if t_as_lambda else wfy.l2
            t1y = wfy.t1
            #ei_o_x -= einsum('iI,Ia,aA,jJ,JA->ij', cx_occ, wfx.t1, rxy_vir, cy_occ, wfy.l1)

            theta = 2*t2 - t2.transpose(0,1,3,2)
            ei_o_x -= einsum('IMAB,iI,mM,aA,bB,pmab->pi', l2y, cy_occ, rxy_occ, rxy_vir, rxy_vir, theta)

            cfc = cfy.dot(cfy.T)

            #eija_o_xy = einsum('IJEA,pe->pija', (l2.transpose(1,0,2,3) - 2*l2), t1y)

            #tmp += 2*einsum('ja,pi->pija', l1, np.eye(nocc)) - einsum('ia,pj->pija', l1, np.eye(nocc))

            eija_o_x += einsum('ijea,eE,pP,PE->pija', (l2.transpose(1,0,2,3) - 2*l2), rxy_vir, cy_occ, t1y)

            eija_o += 2*einsum('jJ,aA,JA,pP,iI,PI->pija', cx_occ, cx_vir, l1, cy_occ, cy_occ, cfc)
            eija_o -= einsum('iI,aA,IA,pP,jJ,PJ->pija', cx_occ, cx_vir, l1, cy_occ, cy_occ, cfc)

        #ei_v = wfx.l1.T#.dot(cfc)
        #t1gx = einsum('iI,aA,ia->IA', cx_occ, cx_vir, t1g)
        #eija_o_x  = einsum('ijea,pe->pija', (l2.transpose(1,0,2,3) - 2*l2), t1gx)
        #eija_o_x += 2*einsum('ja,pi->pija', l1, cfc)
        #eija_o_x -= einsum('ia,pj->pija', l1, cfc)

        eija_v_x = 2*l2.transpose(2,0,1,3) - l2.transpose(3,0,1,2)

        #ei_x = np.concatenate((ei_o, ei_v))
        #eija_x = np.concatenate((eija_o, eija_v))

        ei_o += einsum('pP,Pi->pi', cx_occ, ei_o_x)
        eija_o += einsum('iI,jJ,aA,pIJA->pija', cx_occ, cx_occ, cx_vir, eija_o_x)
        eija_v += einsum('pP,iI,jJ,aA,PIJA->pija', cx_vir, cx_occ, cx_occ, cx_vir, eija_v_x)

    ei = np.concatenate((ei_o, ei_v))
    eija = np.concatenate((eija_o, eija_v))
    return ei, eija

def _build_bra(t1, t2, l1, l2):
    nocc, nvir = t1.shape
    nmo = nocc+nvir
    ei_o = np.eye(nocc) - np.dot(t1, l1.T) - einsum('imab,pmab->pi', l2, (2*t2 - t2.transpose(0,1,3,2))) #problem term
    ei_v = l1.T
    eija_o = einsum('ijea,pe->pija', (l2.transpose(1,0,2,3) - 2*l2), t1) + 2*einsum('ja,pi->pija', l1, np.eye(nocc)) - einsum('ia,pj->pija', l1, np.eye(nocc))
    eija_v = 2*l2.transpose(2,0,1,3) - l2.transpose(3,0,1,2)

    ei = np.concatenate((ei_o, ei_v))
    eija = np.concatenate((eija_o, eija_v))

    return ei, eija

def _build_ket(t1, t2, l1, l2):
    nocc, nvir = t1.shape
    nmo = nocc+nvir
    bi_o = np.eye(nocc)
    bi_v = t1.T
    bija_o = np.zeros((nocc,nocc,nocc,nvir))
    bija_v = t2.transpose(2,0,1,3)

    bi = np.concatenate((bi_o, bi_v))
    bija = np.concatenate((bija_o, bija_v))

    return bi, bija

def get_global_ip_ket(emb, slow=True, t_as_lambda=False):

    if t_as_lambda is None:
        t_as_lambda = emb.opts.t_as_lambda
    if slow:
        t1 = emb.get_global_t1()
        t2 = emb.get_global_t2()

        nocc, nvir = t1.shape
        #nmo = nocc+nvir

        bi_o = np.eye(nocc)
        bi_v = t1.T
        bija_o = np.zeros((nocc,nocc,nocc,nvir))
        bija_v = t2.transpose(2,0,1,3)

        bi = np.concatenate((bi_o, bi_v))
        bija = np.concatenate((bija_o, bija_v))

        return bi, bija

    ovlp = emb.get_ovlp()
    cs_occ = np.dot(emb.mo_coeff_occ.T, ovlp)
    cs_vir = np.dot(emb.mo_coeff_vir.T, ovlp)

    nocc, nvir = cs_occ.shape[0], cs_vir.shape[0]
    nmo = nocc + nvir

    bi = np.zeros((nmo, nocc))
    bija = np.zeros((nmo, nocc, nocc, nvir))


    symfilter = {}#dict(sym_parent=None) if use_sym else {}
    for fx in emb.get_fragments(active=True, mpi_rank=mpi.rank, **symfilter):

        cx_occ = fx.get_overlap('mo[occ]|cluster[occ]')
        cx_vir = fx.get_overlap('mo[vir]|cluster[vir]')
        cx = fx.get_overlap('mo|cluster')
        cfx = fx.get_overlap('cluster[occ]|frag')
        #mfx = fx.get_overlap('mo[occ]|frag')

        # No late symmetrisation
        wfx = fx.results.pwf.as_ccsd().restore()
        t1, t2 = wfx.t1, wfx.t2

        nocc, nvir = t1.shape
        nmo = nocc + nvir

        bi_o = cfx.dot(cfx.T) #np.eye(nocc) Identity matrix projected onto fragment
        bi_v = t1.T
        bija_o = np.zeros((nocc,nocc,nocc,nvir))
        bija_v = t2.transpose(2,0,1,3)

        bi_x = np.concatenate((bi_o, bi_v))
        bija_x = np.concatenate((bija_o, bija_v))

        bi += einsum('pP,iI,PI->pi', cx, cx_occ, bi_x)
        bija += einsum('pP,iI,jJ,aA,PIJA->pija', cx, cx_occ, cx_occ, cx_vir, bija_x)


    return bi, bija

def get_global_ea_bra(emb, slow=True, t_as_lambda=False):

    if t_as_lambda is None:
        t_as_lambda = emb.opts.t_as_lambda
    if slow:
        raise NotImplementedError()

def get_global_ea_ket(emb, slow=True, t_as_lambda=False):

    if t_as_lambda is None:
        t_as_lambda = emb.opts.t_as_lambda
    if slow:
        raise NotImplementedError()

def make_ccsdgf_moms(emb, ao_basis=False, t_as_lambda=False, ovlp_tol=None, svd_tol=None, symmetrize=False, use_sym=False, mpi_target=None, slow=True):
    nmom = emb.opts.nmoments
    if t_as_lambda is None:
        t_as_lambda = emb.opts.t_as_lambda

    if slow:
        t1 = emb.get_global_t1()
        t2 = emb.get_global_t2()
        if t_as_lambda:
            l1, l2 = t1, t2
        else:
            l1 = emb.get_global_l1()
            l2 = emb.get_global_l2()

        cc = pyscf.cc.CCSD(emb.mf)
        cc.t1 = t1
        cc.t2 = t2

        ip = make_ip_moms(cc, t1, t2, l1, l2, nmom=nmom)
        ea = make_ea_moms(cc, t1, t2, l1, l2, nmom=nmom)

        if symmetrize:
            ip = 0.5 * (ip + ip.swapaxes(1, 2).conj())
            ea = 0.5 * (ea + ea.swapaxes(1, 2).conj())

        if ao_basis:
            mo_coeff = emb.mo_coeff
            ip = dot('pP,qQ,mPQ->mpq', mo_coeff, mo_coeff, ip)
            ea = dot('pP,qQ,mPQ->mpq', mo_coeff, mo_coeff, ea)

        return ip, ea

    if ovlp_tol is None:
        ovlp_tol = svd_tol

    # if with_t1:
    t1 = emb.get_global_t1()
    #     l1 = (t1 if t_as_lambda else emb.get_global_l1())

    ovlp = emb.get_ovlp()
    cs_occ = np.dot(emb.mo_coeff_occ.T, ovlp)
    cs_vir = np.dot(emb.mo_coeff_vir.T, ovlp)

    # TODO: MPI

    nocc, nvir = t1.shape
    nmo = nocc + nvir

    ip_mom = np.zeros((nmom, nmo, nmo))
    ea_mom = np.zeros((nmom, nmo, nmo))

    symfilter = dict(sym_parent=None) if use_sym else {}
    for fx in emb.get_fragments(active=True, mpi_rank=mpi.rank, **symfilter):

        if fx.ip_moms is None:
            raise  RuntimeError("No IP moments found for %s!" % fx)

        cx_occ = fx.get_overlap('mo[occ]|cluster[occ]')
        cx_vir = fx.get_overlap('mo[vir]|cluster[vir]')
        cx = fx.get_overlap('mo|cluster')
        cfx = fx.get_overlap('cluster[occ]|frag')
        mfx = fx.get_overlap('mo[occ]|frag')

        bi, bija = fx.ip_moms[0], fx.ip_moms[1]
        # Project occupied indices to frag x and symmetrize
        cfcx = dot(cfx, cfx.T)
        bi = np.einsum('iI,pI->pi', cfcx, bi)
        bija = 0.5 * (einsum('iI,pIja->pija', cfcx, bija) + einsum('jJ,piJa->pija', cfcx, bija))

        ba, biab = fx.ea_moms[0], fx.ea_moms[1]
        # Project occupied indices to frag x and symmetrize
        cfcx = dot(cfx, cfx.T)
        #ba = np.einsum('iI,pI->pi', cfcx, ba)
        biab = einsum('iI,pIab->piab', cfcx, biab)

        nmox = fx.cluster.nocc_active + fx.cluster.nvir_active
        ip_mom_x = np.zeros((nmom,nmox,nmo))
        ea_mom_x = np.zeros((nmom,nmox,nmo))

        for fy in emb.get_fragments(active=True):

            if fy.ip_moms is None:
                raise  RuntimeError("No IP moments found for %s!" % fy)

            #cy_occ = np.dot(cs_occ, cy_occ_ao)
            #cy_vir = np.dot(cs_vir, cy_vir_ao)
            cy_occ = fy.get_overlap('mo[occ]|cluster[occ]')
            cy_vir = fy.get_overlap('mo[vir]|cluster[vir]')
            cfy = fy.get_overlap('cluster[occ]|frag')
            cy = fy.get_overlap('mo|cluster')
            # Overlap between cluster x and cluster y:
            rxy_occ = np.dot(cx_occ.T, cy_occ)
            rxy_vir = np.dot(cx_vir.T, cy_vir)
            #mfy = np.dot(cs_occ, cy_frag)



            # TODO SVD / ovlptol

            # IP
            hei, heija = fy.ip_moms[2], fy.ip_moms[3]
            cfcy = dot(cfy, cfy.T)
            hei = np.einsum('iI,npI->npi', cfcy, hei)
            heija = 0.5 * (einsum('iI,npIja->npija', cfcy, heija) + einsum('jJ,npiJa->npija', cfcy, heija))

            ip_mom_xy = einsum('pi,iI,nQI->npQ', bi, rxy_occ, hei)
            ip_mom_xy += einsum('pija,iI,jJ,aA,nQIJA->npQ', bija, rxy_occ, rxy_occ, rxy_vir, heija)

            # EA
            hea, heiab = fy.ea_moms[2], fy.ea_moms[3]
            cfcy = dot(cfy, cfy.T)
            #hea = np.einsum('aA,npA->npa', cfcy, hea)
            heiab = einsum('iI,npIab->npiab', cfcy, heija)

            ea_mom_xy = einsum('pa,aA,nQA->npQ', ba, rxy_vir, hea)
            ea_mom_xy += einsum('piab,iI,aA,bB,nQIAB->npQ', biab, rxy_occ, rxy_vir, rxy_vir, heiab)

            print(ip_mom_x.shape)
            print(cy.shape)
            print(ip_mom_xy.shape)
            ip_mom_x += einsum('qQ,npQ->npq', cy, ip_mom_xy)
            ea_mom_x += einsum('qQ,npQ->npq', cy, ea_mom_xy)

        # TODO: Symmetry
        ip_mom += einsum('pP,nPq', cx, ip_mom_x)
        ea_mom += einsum('pP,nPq', cx, ea_mom_x)

    return ip_mom


def make_ip_moms(cc, t1, t2, l1, l2, nmom=3, contract_be=True):
        nocc, nvir = t1.shape
        nmo = nocc + nvir

        eom = pyscf.cc.eom_rccsd.EOMIP(cc)
        matvec, diag = eom.gen_matvec()

        bi = np.zeros((nmo, nocc))
        bija = np.zeros((nmo, nocc, nocc, nvir))

        ei = np.zeros((nmo, nocc))
        eija = np.zeros((nmo, nocc, nocc, nvir))

        for p in range(nmo):
            # Build bra vector for orbital p
            b = pyscf.cc.gfccsd.build_bra_hole(cc, eom, t1, t2, l1, l2, p)
            # Unpack bra vector into 1h and 2h1p amplitudes
            b1, b2 = eom.vector_to_amplitudes(b)
            bi[p] = b1
            bija[p] = b2

            # Build ket vector for orbital p
            e = pyscf.cc.gfccsd.build_ket_hole(cc, eom, t1, t2, p)
            # Unpack ket vector into 1h and 2h1p amplitudes
            e1, e2 = eom.vector_to_amplitudes(e)
            ei[p] = e1
            eija[p] = e2

            t = np.zeros((nmom, nmo, nmo))


        if contract_be:
            # Loop over q orbitals
            for q in range(nmo):
                # Loop over moments
                for n in range(nmom):
                    # Loop over p orbitals
                    for p in range(nmo):
                        # Contract 1h and 2h1p parts of b and e to moment
                        t[n, p, q] += np.einsum("i,i->", bi[p], ei[q])
                        t[n, p, q] += np.einsum("ija,ija->", bija[p], eija[q])
                    if (n+1) != nmom:
                        # Scale e = H e for the next moment
                        e_vec = eom.amplitudes_to_vector(ei[q], eija[q])
                        e_vec = -matvec([e_vec])[0]
                        ei[q], eija[q] = eom.vector_to_amplitudes(e_vec)


            return t

        else:
            hei = np.zeros((nmom, nmo, nocc))
            heija = np.zeros((nmom, nmo, nocc, nocc, nvir))
            for n in range(nmom):
                # Loop over p orbitals
                for q in range(nmo):
                    hei[n,q] += ei[q]
                    heija[n,q] += eija[q]
                    if (n+1) != nmom:
                        # Scale e = H e for the next moment
                        e_vec = eom.amplitudes_to_vector(ei[q], eija[q])
                        e_vec = -matvec([e_vec])[0]
                        ei[q], eija[q] = eom.vector_to_amplitudes(e_vec)

            return bi, bija, hei, heija

def make_ea_moms(cc, t1, t2, l1, l2, nmom=3, contract_be=True):
        nocc, nvir = t1.shape
        nmo = nocc + nvir

        eom = pyscf.cc.eom_rccsd.EOMEA(cc)
        matvec, diag = eom.gen_matvec()

        ba = np.zeros((nmo, nvir))
        biab = np.zeros((nmo, nocc, nvir, nvir))

        ea = np.zeros((nmo, nvir))
        eiab = np.zeros((nmo, nocc, nvir, nvir))

        for p in range(nmo):
            # Build bra vector for orbital p
            b = pyscf.cc.gfccsd.build_bra_part(cc, eom, t1, t2, l1, l2, p)
            # Unpack bra vector into 1h and 2h1p amplitudes
            b1, b2 = eom.vector_to_amplitudes(b)

            ba[p] = b1
            biab[p] = b2

            # Build ket vector for orbital p
            e = pyscf.cc.gfccsd.build_ket_part(cc, eom, t1, t2, p)
            # Unpack ket vector into 1h and 2h1p amplitudes
            e1, e2 = eom.vector_to_amplitudes(e)
            ea[p] = e1
            eiab[p] = e2

            t = np.zeros((nmom, nmo, nmo))

        if contract_be:
            # Loop over q orbitals
            for q in range(nmo):
                # Loop over moments
                for n in range(nmom):
                    # Loop over p orbitals
                    for p in range(nmo):
                        # Contract 1h and 2h1p parts of b and e to moment
                        t[n, p, q] -= np.einsum("i,i->", ba[p], ea[q])
                        t[n, p, q] -= np.einsum("ija,ija->", biab[p], eiab[q])
                    if (n+1) != nmom:
                        # Scale e = H e for the next moment
                        e_vec = eom.amplitudes_to_vector(ea[q], eiab[q])
                        e_vec = matvec([e_vec])[0]
                        ea[q], eiab[q] = eom.vector_to_amplitudes(e_vec)

            return t
        else:
            hea = np.zeros((nmom, nmo, nvir))
            heiab = np.zeros((nmom, nmo, nocc, nvir, nvir))
            for n in range(nmom):
                # Loop over p orbitals
                for q in range(nmo):
                    hea[n,q] += ea[q]
                    heiab[n,q] += eiab[q]
                    if (n+1) != nmom:
                        # Scale e = H e for the next moment
                        e_vec = eom.amplitudes_to_vector(ea[q], eiab[q])
                        e_vec = matvec([e_vec])[0]
                        ea[q], eiab[q] = eom.vector_to_amplitudes(e_vec)

            return ba, biab, hea, heiab
