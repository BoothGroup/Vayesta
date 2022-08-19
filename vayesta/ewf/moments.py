"""Routines to generate CCSD Green's function  moments () from spin-restricted embedded CCSD calculations."""

import numpy as np

import pyscf
import pyscf.cc

from vayesta.core.util import *
from vayesta.ewf.rdm import _get_mockcc
from vayesta.mpi import mpi

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
            heiab = einsum('iI,npIab->npija', cfcy, heija)

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
