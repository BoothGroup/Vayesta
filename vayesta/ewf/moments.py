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
