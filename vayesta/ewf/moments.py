"""Routines to generate CCSD Green's function  moments () from spin-restricted embedded CCSD calculations."""

import pyscf
import pyscf.cc

from vayesta.core.util import *
from vayesta.ewf.rdm import _get_mockcc
from vayesta.mpi import mpi

def make_ccsdgf_moms(emb, niter=0, ao_basis=False, t_as_lambda=False, symmetrize=True, mpi_target=None, slow=True):

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

        ip = make_ip_moms(cc, t1, t2, l1, l2, niter=niter)
        ea = make_ea_moms(cc, t1, t2, l1, l2, niter=niter)

        if symmetrize:
            ip = 0.5 * (ip + ip.swapaxes(1, 2).conj())
            ea = 0.5 * (ea + ea.swapaxes(1, 2).conj())

        if ao_basis:
            mo_coeff = emb.mo_coeff
            ip = dot('pP,qQ,mPQ->mpq', mo_coeff, mo_coeff, ip)
            ea = dot('pP,qQ,mPQ->mpq', mo_coeff, mo_coeff, ea)

        return ip, ea

    else:
        raise NotImplementedError()

def make_ip_moms(cc, t1, t2, l1, l2, niter=0):
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

            nmom = 2 * niter + 2
            t = np.zeros((nmom, nmo, nmo))

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
                    e_vec = -matvec([e_vec])[0]  # <--- minus sign only needed for IP
                    ei[q], eija[q] = eom.vector_to_amplitudes(e_vec)

        return t

def make_ea_moms(cc, t1, t2, l1, l2, niter=0):
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
            #print('nocc: %d     nvir: %d    nmo: %d'%(nocc,nvir,nmo))
            #print(bi.shape)
            ba[p] = b1
            biab[p] = b2

            # Build ket vector for orbital p
            e = pyscf.cc.gfccsd.build_ket_part(cc, eom, t1, t2, p)
            # Unpack ket vector into 1h and 2h1p amplitudes
            e1, e2 = eom.vector_to_amplitudes(e)
            ea[p] = e1
            eiab[p] = e2

            nmom = 2 * niter + 2
            t = np.zeros((nmom, nmo, nmo))

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
                    e_vec = matvec([e_vec])[0]  # <--- minus sign only needed for IP
                    ea[q], eiab[q] = eom.vector_to_amplitudes(e_vec)

        return t
