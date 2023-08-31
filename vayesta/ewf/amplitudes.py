import numpy as np

from vayesta.core.util import NotCalculatedError, dot, einsum
from vayesta.mpi import mpi


def get_global_t1_rhf(emb, get_lambda=False, mpi_target=None, ao_basis=False, for_dm2=False, include_bosons=True):
    """Get global CCSD T1 amplitudes from fragment calculations.

    Runtime: N(frag)/N(MPI) * N^2

    Parameters
    ----------
    get_lambda: bool, optional
        If True, return L1 amplitudes. Default: False.
    mpi_target: int or None, optional
        If set to an integer, the result will only be available at the specified MPI rank.
        If set to None, an MPI allreduce will be performed and the result will be available
        at all MPI ranks. Default: None.
    include_bosons: bool, optional
        If True, include bosonic contributions where present. Default: True.

    Returns
    -------
    t1: (n(occ), n(vir)) array
        Global T1 amplitudes.
    """
    t1 = np.zeros((emb.nocc, emb.nvir))
    ovlp = emb.get_ovlp()
    cs_occ = np.dot(emb.mo_coeff_occ.T, ovlp)
    cs_vir = np.dot(emb.mo_coeff_vir.T, ovlp)
    for x in emb.get_fragments(contributes=True, mpi_rank=mpi.rank, sym_parent=None):
        pwf = x.results.pwf.restore().as_ccsd()
        if for_dm2 and x.solver == "MP2":
            continue
        t1x = pwf.l1 if (get_lambda and not x.opts.t_as_lambda) else pwf.t1
        if t1x is None:
            raise NotCalculatedError("Fragment %s" % x)
        for x2, (cx2_occ, cx2_vir) in x.loop_symmetry_children((x.cluster.c_occ, x.cluster.c_vir), include_self=True):
            ro = np.dot(cs_occ, cx2_occ)
            rv = np.dot(cs_vir, cx2_vir)
            t1 += dot(ro, t1x, rv.T)
            # print("Fermionic T1 contrib:", np.linalg.norm(dot(ro, t1x, rv.T)))
        if not (pwf.inc_bosons and include_bosons):
            continue
        # Add bosonic contributions
        s1x = pwf.ls1 if (get_lambda and not x.opts.t_as_lambda) else pwf.s1
        if s1x is None:
            raise NotCalculatedError("Fragment %s" % x)
        s1x = einsum("n,nia->ia", s1x, pwf.mbos.coeff_ex_3d[0])
        forb = pwf.mbos.forbitals
        for x2, (cx2_occ, cx2_vir) in x.loop_symmetry_children((forb.coeff_occ, forb.coeff_vir), include_self=True):
            ro = np.dot(cs_occ, cx2_occ)
            rv = np.dot(cs_vir, cx2_vir)
            # print("S1 T1 contrib:", np.linalg.norm(dot(ro, s1x, rv.T)))
            t1 += dot(ro, s1x, rv.T)

    # --- MPI
    if mpi:
        t1 = mpi.nreduce(t1, target=mpi_target, logfunc=emb.log.timingv)
    if ao_basis:
        t1 = dot(emb.mo_coeff_occ, t1, emb.mo_coeff_vir.T)
    return t1


def get_global_t2_rhf(
    emb, get_lambda=False, symmetrize=True, mpi_target=None, ao_basis=False, for_dm2=False, include_bosons=True
):
    """Get global CCSD T2 amplitudes from fragment calculations.

    Runtime: N(frag)/N(MPI) * N^4

    Parameters
    ----------
    get_lambda: bool, optional
        If True, return L1 amplitudes. Default: False.
    mpi_target: int or None, optional
        If set to an integer, the result will only be available at the specified MPI rank.
        If set to None, an MPI allreduce will be performed and the result will be available
        at all MPI ranks. Default: None.
    include_bosons: bool, optional
        If True, include bosonic contributions where present. Default: True.

    Returns
    -------
    t2: (n(occ), n(occ), n(vir), n(vir)) array
        Global T2 amplitudes.
    """
    t2 = np.zeros((emb.nocc, emb.nocc, emb.nvir, emb.nvir))
    # Add fragment WFs in intermediate normalization
    for x in emb.get_fragments(contributes=True, mpi_rank=mpi.rank):
        emb.log.debugv("Now adding projected %s-amplitudes of fragment %s", ("L" if get_lambda else "T"), x)
        ro = x.get_overlap("mo[occ]|cluster[occ]")
        rv = x.get_overlap("mo[vir]|cluster[vir]")
        pwf = x.results.pwf.restore().as_ccsd()
        if for_dm2 and x.solver == "MP2":
            # Lambda=0 for DM2(MP2)
            if get_lambda:
                continue
            t2x = 2 * pwf.t2
        else:
            t2x = pwf.l2 if (get_lambda and not x.opts.t_as_lambda) else pwf.t2
        if t2x is None:
            raise NotCalculatedError("Fragment %s" % x)
        val = einsum("ijab,Ii,Jj,Aa,Bb->IJAB", t2x, ro, ro, rv, rv)
        # print("Fermionic T2 term:", np.linalg.norm(val))
        t2 += val
        if not (pwf.inc_bosons and include_bosons):
            continue

        # Add bosonic contributions
        u11 = pwf.lu11 if (get_lambda and not x.opts.t_as_lambda) else pwf.u11
        if u11 is None:
            raise NotCalculatedError("Fragment %s" % x)

        u11 = einsum("nia,Ii,Aa->nIA", u11, ro, rv)
        # TODO add check that bosons are correctly in MOs. This will always be the case though...
        u11 = einsum("nia,njb->ijab", u11, pwf.mbos.coeff_ex_3d[0])
        # Note that this is a separate contribution, so we don't need to average.
        t2 += u11 + u11.transpose(1, 0, 3, 2)
        # print("Coupled T2 term:", np.linalg.norm(u11), np.linalg.norm(u11+u11.transpose(1, 0, 3, 2)))
        # This may not be present in CCSD-S-1-1
        s2 = pwf.ls2 if (get_lambda and not x.opts.t_as_lambda) else pwf.s2
        if s2 is None:
            continue

        t2 += einsum("nm,nia,mjb->ijab", s2, pwf.mbos.coeff_ex_3d[0], pwf.mbos.coeff_ex_3d[0])
        # print("S2 T2 term:", np.linalg.norm(einsum("nm,nia,mjb->ijab", s2, pwf.mbos.coeff_ex_3d[0], pwf.mbos.coeff_ex_3d[0])))

    # --- MPI
    if mpi:
        t2 = mpi.nreduce(t2, target=mpi_target, logfunc=emb.log.timingv)
    if ao_basis:
        t2 = einsum(
            "Ii,Jj,ijab,Aa,Bb->IJAB", emb.mo_coeff_occ, emb.mo_coeff_occ, t2, emb.mo_coeff_vir, emb.mo_coeff_vir
        )
    return t2


def get_global_t1_uhf(emb, get_lambda=False, mpi_target=None, ao_basis=False):
    """Get global CCSD T1 from fragment calculations.

    Parameters
    ----------
    get_lambda: bool, optional
        If True, return L1 amplitudes. Default: False.
    mpi_target: int or None, optional
        If set to an integer, the result will only be available at the specified MPI rank.
        If set to None, an MPI allreduce will be performed and the result will be available
        at all MPI ranks. Default: None.

    Returns
    -------
    t1: tuple(2) of (n(occ), n(vir)) array
        Global T1 amplitudes.
    """
    t1a = np.zeros((emb.nocc[0], emb.nvir[0]))
    t1b = np.zeros((emb.nocc[1], emb.nvir[1]))
    # Add fragment WFs in intermediate normalization
    for x in emb.get_fragments(contributes=True, mpi_rank=mpi.rank):
        emb.log.debugv("Now adding projected %s-amplitudes of fragment %s", ("L" if get_lambda else "T"), x)
        roa, rob = x.get_overlap("mo[occ]|cluster[occ]")
        rva, rvb = x.get_overlap("mo[vir]|cluster[vir]")
        pwf = x.results.pwf.restore().as_ccsd()
        t1xa, t1xb = pwf.l1 if (get_lambda and not x.opts.t_as_lambda) else pwf.t1
        if t1xa is None:
            raise NotCalculatedError("Fragment %s" % x)
        t1a += einsum("ia,Ii,Aa->IA", t1xa, roa, rva)
        t1b += einsum("ia,Ii,Aa->IA", t1xb, rob, rvb)
    # --- MPI
    if mpi:
        t1a, t1b = mpi.nreduce(t1a, t1b, target=mpi_target, logfunc=emb.log.timingv)
    if ao_basis:
        t1a = dot(emb.mo_coeff_occ[0], t1a, emb.mo_coeff_vir[0].T)
        t1b = dot(emb.mo_coeff_occ[1], t1b, emb.mo_coeff_vir[1].T)
    return (t1a, t1b)


def get_global_t2_uhf(emb, get_lambda=False, symmetrize=True, mpi_target=None, ao_basis=False):
    """Get global CCSD T2 amplitudes from fragment calculations.

    Parameters
    ----------
    get_lambda: bool, optional
        If True, return L1 amplitudes. Default: False.
    mpi_target: int or None, optional
        If set to an integer, the result will only be available at the specified MPI rank.
        If set to None, an MPI allreduce will be performed and the result will be available
        at all MPI ranks. Default: None.

    Returns
    -------
    t2: tuple(3) of (n(occ), n(occ), n(vir), n(vir)) array
        Global T2 amplitudes.
    """
    t2aa = np.zeros((emb.nocc[0], emb.nocc[0], emb.nvir[0], emb.nvir[0]))
    t2ab = np.zeros((emb.nocc[0], emb.nocc[1], emb.nvir[0], emb.nvir[1]))
    t2bb = np.zeros((emb.nocc[1], emb.nocc[1], emb.nvir[1], emb.nvir[1]))
    # Add fragment WFs in intermediate normalization
    for x in emb.get_fragments(contributes=True, mpi_rank=mpi.rank):
        emb.log.debugv("Now adding projected %s-amplitudes of fragment %s", ("L" if get_lambda else "T"), x)
        roa, rob = x.get_overlap("mo[occ]|cluster[occ]")
        rva, rvb = x.get_overlap("mo[vir]|cluster[vir]")
        pwf = x.results.pwf.restore().as_ccsd()
        t2xaa, t2xab, t2xbb = pwf.l2 if (get_lambda and not x.opts.t_as_lambda) else pwf.t2
        if t2xaa is None:
            raise NotCalculatedError("Fragment %s" % x)
        t2aa += einsum("ijab,Ii,Jj,Aa,Bb->IJAB", t2xaa, roa, roa, rva, rva)
        t2ab += einsum("ijab,Ii,Jj,Aa,Bb->IJAB", t2xab, roa, rob, rva, rvb)
        t2bb += einsum("ijab,Ii,Jj,Aa,Bb->IJAB", t2xbb, rob, rob, rvb, rvb)
    # --- MPI
    if mpi:
        t2aa, t2ab, t2bb = mpi.nreduce(t2aa, t2ab, t2bb, target=mpi_target, logfunc=emb.log.timingv)
    if ao_basis:
        coa, cob = emb.mo_coeff_occ
        cva, cvb = emb.mo_coeff_vir
        t2aa = einsum("Ii,Jj,ijab,Aa,Bb->IJAB", coa, coa, t2aa, cva, cva)
        t2ab = einsum("Ii,Jj,ijab,Aa,Bb->IJAB", coa, cob, t2ab, cva, cvb)
        t2bb = einsum("Ii,Jj,ijab,Aa,Bb->IJAB", cob, cob, t2bb, cvb, cvb)
    return (t2aa, t2ab, t2bb)
