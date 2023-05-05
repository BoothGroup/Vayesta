import numpy as np
import pyscf
from vayesta.core.ao2mo import helper as ao2mo_helper
from vayesta.core.util import *
from vayesta.core import spinalg
from vayesta.mpi import mpi, RMA_Dict
from vayesta.solver.simple import CCSD as SimpleCCSD
from vayesta.solver_rewrite import ccsdtq


def transform_amplitude(t, u_occ, u_vir, u_occ2=None, u_vir2=None, inverse=False):
    """u: (old basis|new basis)"""
    if u_occ2 is None:
        u_occ2 = u_occ
    if u_vir2 is None:
        u_vir2 = u_vir
    if inverse:
        u_occ = spinalg.T(u_occ)
        u_occ2 = spinalg.T(u_occ2)
        u_vir = spinalg.T(u_vir)
        u_vir2 = spinalg.T(u_vir2)

    ndim = t[0].ndim + 1
    # Restricted T1:
    if ndim == 2:
        return einsum('ia,ix,ay->xy', t, u_occ, u_vir)
    # Restricted T2:
    if ndim == 4:
        return einsum('ijab,ix,jy,az,bw->xyzw', t, u_occ, u_occ2, u_vir, u_vir2)
    # Unrestricted T1:
    if ndim == 3:
        ta = transform_amplitude(t[0], u_occ[0], u_vir[0])
        tb = transform_amplitude(t[1], u_occ[1], u_vir[1])
        return (ta, tb)
    # Unrestricted T2:
    if ndim == 5:
        taa = transform_amplitude(t[0], u_occ[0], u_vir[0])
        tab = transform_amplitude(t[1], u_occ[0], u_vir[0], u_occ[1], u_vir[1])
        tbb = transform_amplitude(t[2], u_occ[1], u_vir[1])
        return (taa, tab, tbb)
    raise NotImplementedError("Transformation of %s amplitudes with ndim=%d" % (spinsym, np.ndim(t[0])+1))


def get_amplitude_norm(t1, t2):
    # Restricted:
    if np.ndim(t1[0]) == 1:
        t1norm = np.linalg.norm(t1)
        t2norm = np.linalg.norm(t2)
    # Unrestricted
    elif np.ndim(t1[0]) == 2:
        t1norm = (np.linalg.norm(t1[0])+np.linalg.norm(t1[1]))/2
        t2norm = (np.linalg.norm(t2[0])+2*np.linalg.norm(t2[1])+np.linalg.norm(t2[2]))/2
    return t1norm, t2norm


def project_t2(t2, proj, projectors):
    ndim = t2[0].ndim + 1
    if ndim == 4:
        return project_t2_rspin(t2, proj, projectors)
    if ndim == 5:
        return project_t2_uspin(t2, proj, projectors)
    raise ValueError


def project_t2_rspin(t2, proj, projectors):
    if projectors == 0:
        return t2
    if projectors == 1:
        t2 = einsum('xi,i...->x...', proj, t2)
        return (t2 + t2.transpose(1,0,3,2))/2
    if projectors == 2:
        return einsum('xi,yj,ij...->xy...', proj, proj, t2)
    raise ValueError


def project_t2_uspin(t2, proj, projectors):
    if projectors == 0:
        return t2
    t2aa = project_t2_rspin(t2[0], proj[0], projectors=projectors)
    t2bb = project_t2_rspin(t2[2], proj[1], projectors=projectors)
    if projectors == 1:
        # Average between projecting alpha and beta:
        t2ab = (einsum('xi,ij...->xj...', proj[0], t2[1])
              + einsum('xj,ij...->ix...', proj[1], t2[1]))/2
    elif projectors == 2:
        t2ab = einsum('xi,yj,ij...->xy...', proj[0], proj[1], t2[1])
    else:
        raise ValueError
    #assert np.allclose(t2ab, -t2ab.transpose(0,1,3,2))
    return (t2aa, t2ab, t2bb)


def couple_ccsd_iterations(solver, fragments):
    """

    Requires MPI.
    """
    # Make projector P(y):
    # P(y) = C(x).T S F(y) F(y).T S C(y)
    # where
    # C(x): Cluster orbitals of fragment x
    # S: AO-overlap
    # F(x): Fragment orbitals of fragment x

    ovlp = solver.base.get_ovlp()
    c_occ_x = np.asarray(solver.cluster.c_active_occ, order='C')
    c_vir_x = np.asarray(solver.cluster.c_active_vir, order='C')
    p_occ = {}
    r_occ = {}
    r_vir = {}
    rma = RMA_Dict.from_dict(mpi, {
        (solver.fragment.id, 'c_active_occ'): c_occ_x,
        (solver.fragment.id, 'c_active_vir'): c_vir_x})
    for y in fragments:
        fy = y.c_proj
        c_occ_y = rma[(y.id, 'c_active_occ')]
        c_vir_y = rma[(y.id, 'c_active_vir')]
        p_occ[y.id] = einsum('ai,ab,by,cy,cd,dj->ij', c_occ_x, ovlp, fy, fy, ovlp, c_occ_y)
        r_occ[y.id] = einsum('ai,ab,bj->ij', c_occ_x, ovlp, c_occ_y)
        r_vir[y.id] = einsum('ai,ab,bj->ij', c_vir_x, ovlp, c_vir_y)
    rma.clear()

    def tailorfunc(kwargs):
        cc = kwargs['mycc']
        t1, t2 = kwargs['t1new'], kwargs['t2new']
        cc.force_iter = True
        cc.force_exit = bool(mpi.world.allreduce(int(cc.conv_flag), op=mpi.MPI.PROD))
        conv = mpi.world.gather(int(cc.conv_flag), root=0)

        rma = RMA_Dict.from_dict(mpi, {(mpi.rank, 't1'): t1, (mpi.rank, 't2'): t2})

        t1_out = np.zeros_like(t1)
        t2_out = np.zeros_like(t2)
        for y in fragments:
            t1y, t2y = rma[(y.id, 't1')], rma[(y.id, 't2')]
            po = p_occ[y.id]
            ro = r_occ[y.id]
            rv = r_vir[y.id]
            #print(solver.fragment.id, y.id, py.shape, t1_out.shape, t1y.shape)
            t1_out += einsum('Ii,ia,Aa->IA', po, t1y, rv)
            t2_out += einsum('Ii,Jj,ijab,Aa,Bb->IJAB', po, ro, t2y, rv, rv)
        solver.log.info("Tailoring: |dT1|= %.3e  |dT2|= %.3e", np.linalg.norm(t1_out-t1), np.linalg.norm(t2_out-t2))
        rma.clear()
        t1[:] = t1_out
        t2[:] = t2_out

    return tailorfunc


def tailor_with_fragments(solver, fragments, project=False, tailor_t1=True, tailor_t2=True, ovlp_tol=1e-6):
    """Tailor current CCSD calculation with amplitudes of other fragments.

    This assumes orthogonal fragment spaces.

    Parameters
    ----------
    project: int, optional
        Level of external correction of T2 amplitudes:
        1: Both occupied indices are projected to each other fragment X.
        2: Both occupied indices are projected to each other fragment X
           and combinations of other fragments X,Y.
        3: Only the first occupied indices is projected to each other fragment X.
    coupled_fragments: list, optional
        List of fragments, which are used for the external correction.
        Each fragment x must have the following attributes defined:
        `c_active_occ` : Active occupied MO orbitals of fragment x
        `c_active_vir` : Active virtual MO orbitals of fragment x
        `results.t1` :   T1 amplitudes of fragment x
        `results.t2` :   T2 amplitudes of fragment x

    Returns
    -------
    tailor_func : function(cc, t1, t2) -> t1, t2
        Tailoring function for CCSD.
    """
    fragment = solver.fragment
    cluster = solver.cluster
    ovlp = solver.base.get_ovlp()       # AO overlap matrix
    cx_occ = cluster.c_active_occ       # Occupied active orbitals of current cluster
    cx_vir = cluster.c_active_vir       # Virtual  active orbitals of current cluster
    cxs_occ = spinalg.dot(spinalg.T(cx_occ), ovlp)
    cxs_vir = spinalg.dot(spinalg.T(cx_vir), ovlp)
    project = int(project)
    nxy_occ = solver.base.get_fragment_overlap_norm(fragments=([fragment], fragments), virtual=False, norm=None)[0]
    nxy_vir = solver.base.get_fragment_overlap_norm(fragments=([fragment], fragments), occupied=False, norm=None)[0]

    def tailor_func(kwargs):
        """Add external correction to T1 and T2 amplitudes."""
        t1, t2 = kwargs['t1new'], kwargs['t2new']
        # Collect all changes to the amplitudes in dt1 and dt2:
        if tailor_t1:
            dt1 = spinalg.zeros_like(t1)
        if tailor_t2:
            dt2 = spinalg.zeros_like(t2)

        # Loop over all *other* fragments/cluster X
        for y, fy in enumerate(fragments):
            assert (fy is not fragment)

            # Rotation & projections from cluster X active space to current fragment active space
            rxy_occ = spinalg.dot(cxs_occ, fy.cluster.c_active_occ)
            rxy_vir = spinalg.dot(cxs_vir, fy.cluster.c_active_vir)
            # Skip fragment if there is no overlap
            if solver.spinsym == 'restricted':
                maxovlp = min(abs(rxy_occ).max(), abs(rxy_vir).max())
            elif solver.spinsym == 'unrestricted':
                maxovlp = min(max(abs(rxy_occ[0]).max(), abs(rxy_occ[1]).max()),
                              max(abs(rxy_vir[0]).max(), abs(rxy_vir[1]).max()))
            if maxovlp < ovlp_tol:
                self.log.debug("Skipping tailoring fragment %s due to small overlap= %.1e", fy, maxovlp)
                continue

            wfy = fy.results.wf.as_ccsd()
            # Transform to x-amplitudes to y-space, instead of y-amplitudes to x-space:
            # x may be CCSD and y FCI, such that x-space >> y-space
            if tailor_t1:
                t1x = transform_amplitude(t1, rxy_occ, rxy_vir)
                dt1y = spinalg.subtract(wfy.t1, t1x)
            if tailor_t2:
                t2x = transform_amplitude(t2, rxy_occ, rxy_vir)
                dt2y = spinalg.subtract(wfy.t2, t2x)

            # Project first one/two occupied index/indices onto fragment(y) space:
            if project:
                proj = fy.get_overlap('frag|cluster-occ')
                proj = spinalg.dot(spinalg.T(proj), proj)
                if tailor_t1:
                    dt1y = spinalg.dot(proj, dt1y)
                if tailor_t2:
                    dt2y = project_t2(dt2y, proj, projectors=project)

            # Transform back to x-space and add:
            if tailor_t1:
                dt1 = spinalg.add(dt1, transform_amplitude(dt1y, rxy_occ, rxy_vir, inverse=True))
            if tailor_t2:
                dt2 = spinalg.add(dt2, transform_amplitude(dt2y, rxy_occ, rxy_vir, inverse=True))

            solver.log.debug("Tailoring with fragment %3d (%s):  S(occ)= %.3e  S(vir)= %.3e  dT1= %.3e  dT2= %.3e",
                             fy.id, fy.solver, nxy_occ[y], nxy_vir[y], *get_amplitude_norm(dt1y, dt2y))

        # Add correction:
        if tailor_t1:
            if solver.spinsym == 'restricted':
                t1[:] += dt1
            elif solver.spinsym == 'unrestricted':
                t1[0][:] += dt1[0]
                t1[1][:] += dt1[1]
        if tailor_t2:
            if solver.spinsym == 'restricted':
                t2[:] += dt2
            elif solver.spinsym == 'unrestricted':
                t2[0][:] += dt2[0]
                t2[1][:] += dt2[1]
                t2[2][:] += dt2[2]
        solver.log.debug("Tailoring total:  dT1= %.3e  dT2= %.3e", *get_amplitude_norm(dt1, dt2))

    return tailor_func


def _integrals_for_extcorr(fragment, fock):
    eris = fragment._eris
    cluster = fragment.cluster
    emb = fragment.base
    if eris is None:
        if emb.spinsym == 'restricted':
            eris = emb.get_eris_array(cluster.c_active)
        else:
            eris = emb.get_eris_array_uhf(cluster.c_active)

    if emb.spinsym == 'restricted':
        occ = np.s_[:cluster.nocc_active]
        vir = np.s_[cluster.nocc_active:]
        govov = eris[occ,vir,occ,vir] # chemical notation 
        gvvov = eris[vir,vir,occ,vir]
        gooov = eris[occ,occ,occ,vir]
        govoo = eris[occ,vir,occ,occ]
        fov = dot(cluster.c_active_occ.T, fock, cluster.c_active_vir)
        return fov, (govov, gvvov, gooov, govoo)

    elif emb.spinsym == 'unrestricted':
        oa = np.s_[:cluster.nocc_active[0]]
        ob = np.s_[:cluster.nocc_active[1]]
        va = np.s_[cluster.nocc_active[0]:]
        vb = np.s_[cluster.nocc_active[1]:]
        fova = dot(cluster.c_active_occ[0].T, fock[0], cluster.c_active_vir[0])
        fovb = dot(cluster.c_active_occ[1].T, fock[1], cluster.c_active_vir[1])
        fov = (fova, fovb)
        gooov = (eris[0][oa,oa,oa,va], eris[1][oa,oa,ob,vb], eris[2][ob,ob,ob,vb])
        govov = (eris[0][oa,va,oa,va], eris[1][oa,va,ob,vb], eris[2][ob,vb,ob,vb])
        govvv = (eris[0][oa,va,va,va], eris[1][oa,va,vb,vb], eris[2][ob,vb,vb,vb])
        gvvov = (None, eris[1][va,va,ob,vb], None)
        govoo = (None, eris[1][oa,va,ob,ob], None)
        return fov, (gooov, govov, govvv, gvvov, govoo)

    else:
        raise NotImplementedError(emb.spinsym)

def _get_delta_t_for_extcorr(fragment, fock, solver, include_t3v=True):
    """Make T3 and T4 residual correction to CCSD wave function for given fragment.
    Expressions consistent with J. Chem. Phys. 86, 2881 (1987): G. E. Scuseria et al.
    and verified same behaviour as implementation in git@github.com:gustavojra/Methods.git
    TODO
    ----
    o Add Option: Contract T4's down at original solver point to save memory.
    o UHF implementation

    Parameters
    ----------
    fragment : Fragment
        FCI fragment with FCI, CISDTQ or CCSDTQ wave function object in results
    fock : numpy.ndarray
        Full system for matrix used for CCSD residuals
    solver : Solver
        Used for logging options
    include_t3v : bool
        If include_t3v, then these terms are included. If not, they are left out
        (to be contracted later with cluster y integrals). 

    Returns
    -------
    dt1, dt2 : numpy.ndarray
        T1 and T2 expressions. 

    NOTE: These expressions still need to be contracted with energy denominators 
    for full amplitude updates.
    """

    wf = fragment.results.wf.as_ccsdtq()

    # --- Make correction to T1 and T2 amplitudes
    # J. Chem. Theory Comput. 2021, 17, 182−190
    # also with reference to git@github.com:gustavojra/Methods.git
    
    if fragment.base.spinsym == 'restricted':
        # Get ERIs and Fock matrix for the given fragment
        f, v = _integrals_for_extcorr(fragment, fock)
        t1, t2, t3 = wf.t1, wf.t2, wf.t3
        t4_abaa, t4_abab = wf.t4

        dt1, dt2 = ccsdtq.t_residual_rhf(solver, fragment, wf.t1, wf.t2, wf.t3, wf.t4, f, v, include_t3v=include_t3v)

    elif fragment.base.spinsym == 'unrestricted':
        # Get ERIs and Fock matrix for the given fragment
        f, v = _integrals_for_extcorr(fragment, fock)

        dt1, dt2 = ccsdtq.t_residual_uhf(solver, fragment, wf.t1, wf.t2, wf.t3, wf.t4, f, v, include_t3v=include_t3v)

    else:
        raise ValueError

    return dt1, dt2

def _get_delta_t2_from_t3v(govvv_x, gvvov_x, gooov_x, govoo_x, frag_child, rxy_occ, rxy_vir, cxs_occ, projectors):
    """Perform the (T3 * V) contraction for the external correction, with the V integrals
    in the parent basis (x). This will change the results as the V retains one open index
    in the resulting T2 contributions.

    Parameters
    ----------
    govvv_x, gvvov_x, gooov_x, govoo_x : ndarray
        Integrals over various occ, vir slices in the parent (x) cluster.
        govvv_x not required for unrestricted calculations.
    frag_child : Fragment
        Fragment of the child cluster (y)
    rxy_occ, rxy_vir : numpy.ndarray
        Projection operator from x cluster to y cluster in the occ (vir) space
    cxs_occ : numpy.ndarray
        Cluster orbitals of cluster x contracted with overlap
    projectors : int
        Number of projectors onto the fragment space of y

    Returns
    -------
    dt2 : numpy.ndarray
        Update to T2 amplitudes in the parent (x) basis
    """
    
    wf = frag_child.results.wf.as_ccsdtq()
    t3 = wf.t3

    if frag_child.base.spinsym == 'restricted':

        # First term, using (vv|ov): 1/2 P_ab [t_ijmaef v_efbm]
        # Three-quarter transform of passed-in integrals from parent (x) to child (y) basis
        # Keep first index of vvov integrals in x basis. Transform rest to y basis.
        gvvov_ = einsum('abic,bB,iI,cC -> aBIC', gvvov_x, rxy_vir, rxy_occ, rxy_vir)

        # Contract with T3 amplitudes in the y basis
        t3v_ = 0.5*einsum('bemf, jimeaf -> ijab', gvvov_ - gvvov_.transpose(0,3,2,1), t3)
        t3v_ += einsum('bemf, ijmaef -> ijab', gvvov_, t3)
        # Final is in a mixed basis form, with last index in t3v here in the x basis
        # Rotate remaining indices into x basis: another three-quarter transform
        t3v_x = einsum('IJAb,iI,jJ,aA -> ijab', t3v_, rxy_occ, rxy_occ, rxy_vir)

        # Second term: -1/2 P_ij [t_imnabe v_jemn]
        # ooov three-quarter transform, to IjKA (capital is y (child) basis)
        gooov_ = einsum('ijka,iI,kK,aA -> IjKA', gooov_x, rxy_occ, rxy_occ, rxy_vir)
        # ovoo three-quarter transform, to IAJk (capital is y (child) basis)
        govoo_ = einsum('iajk,iI,aA,jJ -> IAJk', govoo_x, rxy_occ, rxy_vir, rxy_occ)

        # Second index of t3v_ in the parent (x) basis
        t3v_ = -0.5*einsum('mjne, minbae -> ijab', gooov_ - govoo_.transpose(0,3,2,1), t3)
        t3v_ -= einsum('mjne, imnabe -> ijab', gooov_, t3)
        # Rotate remaining indices into x basis: another three-quarter transform
        t3v_x += einsum('IjAB,iI,aA,bB -> ijab', t3v_, rxy_occ, rxy_vir, rxy_vir)

        # Include permutation
        dt2 = t3v_x + t3v_x.transpose(1,0,3,2)

    elif frag_child.base.spinsym == 'unrestricted':
        gooov_aaaa, gooov_aabb, gooov_bbbb = gooov_x
        govoo_aaaa, govoo_aabb, govoo_bbbb = govoo_x
        govvv_aaaa, govvv_aabb, govvv_bbbb = govvv_x
        gvvov_aaaa, gvvov_aabb, gvvov_bbbb = gvvov_x

        t3_aaa, t3_aba, t3_bab, t3_bbb = t3

        govvv_aaaa_ = einsum('iabc,iI,aA,cC -> IAbC', govvv_aaaa, rxy_occ[0], rxy_vir[0], rxy_vir[0])
        govvv_aabb_ = einsum('iabc,iI,aA,cC -> IAbC', govvv_aabb, rxy_occ[0], rxy_vir[0], rxy_vir[1])
        govvv_bbbb_ = einsum('iabc,iI,aA,cC -> IAbC', govvv_bbbb, rxy_occ[1], rxy_vir[1], rxy_vir[1])

        gvvov_aaaa_ = einsum('abic,bB,iI,cC -> aBIC', gvvov_aaaa, rxy_vir[0], rxy_occ[0], rxy_vir[0])
        gvvov_aabb_ = einsum('abic,bB,iI,cC -> aBIC', gvvov_aabb, rxy_vir[0], rxy_occ[1], rxy_vir[1])
        gvvov_bbbb_ = einsum('abic,bB,iI,cC -> aBIC', gvvov_bbbb, rxy_vir[1], rxy_occ[1], rxy_vir[1])

        gooov_aaaa_ = einsum('ijka,jJ,kK,aA -> iJKA', gooov_aaaa, rxy_occ[0], rxy_occ[0], rxy_vir[0])
        gooov_aabb_ = einsum('ijka,jJ,kK,aA -> iJKA', gooov_aabb, rxy_occ[0], rxy_occ[1], rxy_vir[1])
        gooov_bbbb_ = einsum('ijka,jJ,kK,aA -> iJKA', gooov_bbbb, rxy_occ[1], rxy_occ[1], rxy_vir[1])

        govoo_aaaa_ = einsum('iajk,iI,aA,kK -> IAjK', govoo_aaaa, rxy_occ[0], rxy_vir[0], rxy_occ[0])
        govoo_aabb_ = einsum('iajk,iI,aA,kK -> IAjK', govoo_aabb, rxy_occ[0], rxy_vir[0], rxy_occ[1])
        govoo_bbbb_ = einsum('iajk,iI,aA,kK -> IAjK', govoo_bbbb, rxy_occ[1], rxy_vir[1], rxy_occ[1])

        x0 = einsum("Jlkc,iklacb->iJab", gooov_aabb_, t3_aba)
        x1 = einsum("Jlkc,iklabc->iJab", gooov_aaaa_, t3_aaa) * -1.0
        x2 =  einsum("iJba->iJab", x0) * -1.0
        x2 += einsum("iJba->iJab", x1) * -1.0
        x2 =  einsum("iJab,Ii,Aa,Bb->IJAB", x2, rxy_occ[0], rxy_vir[0], rxy_vir[0])
        dt2_aaaa =  einsum("ijab->ijab", x2) * -1.0
        dt2_aaaa += einsum("jiab->ijab", x2)

        x0 = einsum("Bdkc,ikjacd->ijaB", gvvov_aabb_, t3_aba)
        x1 = einsum("kcBd,ijkacd->ijaB", govvv_aaaa_, t3_aaa)
        x2 =  einsum("ijaB->ijaB", x0)
        x2 += einsum("ijaB->ijaB", x1) * -1.0
        x2 =  einsum("ijaB,Ii,Jj,Aa->IJAB", x2, rxy_occ[0], rxy_occ[0], rxy_vir[0])
        dt2_aaaa += einsum("ijab->ijab", x2)
        dt2_aaaa += einsum("ijba->ijab", x2) * -1.0

        x0 = einsum("Jklc,iklabc->iJab", gooov_bbbb_, t3_bbb)
        x1 = einsum("lcJk,ilkacb->iJab", govoo_aabb_, t3_bab)
        x2 =  einsum("iJba->iJab", x0) * -1.0
        x2 += einsum("iJba->iJab", x1) * -1.0
        x2 =  einsum("iJab,Ii,Aa,Bb->IJAB", x2, rxy_occ[1], rxy_vir[1], rxy_vir[1])
        dt2_bbbb =  einsum("ijab->ijab", x2) * -1.0
        dt2_bbbb += einsum("jiab->ijab", x2)

        x0 = einsum("kdBc,ijkacd->ijaB", govvv_bbbb_, t3_bbb) * -1.0
        x1 = einsum("kdBc,ikjadc->ijaB", govvv_aabb_, t3_bab)
        x2 =  einsum("ijaB->ijaB", x0)
        x2 += einsum("ijaB->ijaB", x1) * -1.0
        x2 =  einsum("ijaB,Ii,Jj,Aa->IJAB", x2, rxy_occ[1], rxy_occ[1], rxy_vir[1])
        dt2_bbbb += einsum("ijab->ijab", x2) * -1.0
        dt2_bbbb += einsum("ijba->ijab", x2)

        x0 =  einsum("Ilkc,jlkbac->Ijab", gooov_aabb_, t3_bab) * -1.0
        x0 += einsum("Ilmd,ljmabd->Ijab", gooov_aaaa_, t3_aba) * -1.0
        dt2_abab =  einsum("Ijab,Jj,Aa,Bb->IJAB", x0, rxy_occ[1], rxy_vir[0], rxy_vir[1])

        x0 =  einsum("Jnkc,kinbac->iJab", gooov_bbbb_, t3_bab)
        x0 += einsum("ldJk,iklabd->iJab", govoo_aabb_, t3_aba) * -1.0
        dt2_abab += einsum("iJab,Ii,Aa,Bb->IJAB", x0, rxy_occ[0], rxy_vir[0], rxy_vir[1])

        x0 =  einsum("ldAe,ijldbe->ijAb", govvv_aaaa_, t3_aba) * -1.0
        x0 += einsum("Adkc,jikbdc->ijAb", gvvov_aabb_, t3_bab)
        dt2_abab += einsum("ijAb,Ii,Jj,Bb->IJAB", x0, rxy_occ[0], rxy_occ[1], rxy_vir[1])

        x0 =  einsum("ldBc,ijlacd->ijaB", govvv_aabb_, t3_aba)
        x0 += einsum("kcBf,jikcaf->ijaB", govvv_bbbb_, t3_bab) * -1.0
        dt2_abab += einsum("ijaB,Ii,Jj,Aa->IJAB", x0, rxy_occ[0], rxy_occ[1], rxy_vir[0])

        dt2 = (dt2_aaaa, dt2_abab, dt2_bbbb)

    else:
        raise ValueError

    # Find the fragment projector of cluster y (child) in the basis of cluster x (parent)
    c_frag_xocc = spinalg.dot(spinalg.T(frag_child.c_frag), spinalg.T(cxs_occ))
    proj_y_in_x = spinalg.dot(spinalg.T(c_frag_xocc), c_frag_xocc)
    
    # Project (t3 v) contribution onto fragment of cluster y
    dt2 = project_t2(dt2, proj_y_in_x, projectors=projectors)

    return dt2

def _get_delta_t_for_delta_tailor(fragment, fock):
    wf = fragment.results.wf.as_ccsd()
    t1, t2 = wf.t1, wf.t2
    # CCSD
    cluster = fragment.cluster
    fock = spinalg.dot(spinalg.T(cluster.c_active), fock, cluster.c_active)
    nocc = wf.mo.nocc
    if fragment.base.spinsym == 'restricted':
        mo_energy = np.diag(fock).copy()
        if fragment.base.has_exxdiv:
            mo_energy[:nocc] -= fragment.base.madelung
    else:
        mo_energy = (np.diag(fock[0]).copy(), np.diag(fock[1]).copy())
        if fragment.base.has_exxdiv:
            moa, mob = mo_energy
            moa[:nocc[0]] -= fragment.base.madelung
            mob[:nocc[1]] -= fragment.base.madelung
            mo_energy = (moa, mob)

    eris = fragment._eris
    if eris is None:
        # Symmetry-derived fragments may not have eris stored
        # TODO: This may not scale well, since we are performing
        # integral transformations for all fragments. Can be improved.
        eris = fragment.base.get_eris_array(cluster.c_active)
    ccsd = SimpleCCSD(fock, eris, nocc, mo_energy=mo_energy)
    wf = ccsd.kernel(t1=t1, t2=t2)
    assert ccsd.converged
    dt1 = spinalg.subtract(t1, wf.t1)
    dt2 = spinalg.subtract(t2, wf.t2)
    return dt1, dt2


def externally_correct(solver, external_corrections, eris=None):
    """Build callback function for CCSD, to add external correction from other fragments.

    TODO: combine with `tailor_with_fragments`?

    Parameters
    ----------
    solver : CCSD_Solver
        Vayesta CCSD solver.
    external_corrections : list of tuple of (int, str, int, bool)
        List of external corrections. Each tuple contains the fragment ID, type of correction,
        and number of projectors for the given external correction. Final element is boolean giving
        the low_level_coul optional argument.
    eris : _ChemistsERIs
        ERIs for parent CCSD fragment. Used for MO energies in residual contraction, and for
        the case of low_level_coul, where the parent Coulomb integral is contracted.
        If not passed in, MO energy if needed will be constructed from the diagonal of 
        get_fock() of embedding base class, and the eris will be also be obtained from the
        embedding base class. Optional.

    Returns
    -------
    callback : callable
        Callback function for PySCF's CCSD solver.
    """

    fx = solver.fragment
    cluster = solver.cluster
    emb = solver.base
    nocc = cluster.nocc
    nvir = cluster.nvir
    ovlp = emb.get_ovlp()               # AO overlap matrix
    cx_occ = cluster.c_active_occ       # Occupied active orbitals of current cluster
    cx_vir = cluster.c_active_vir       # Virtual  active orbitals of current cluster
    cxs_occ = spinalg.dot(spinalg.T(cx_occ), ovlp)
    cxs_vir = spinalg.dot(spinalg.T(cx_vir), ovlp)
    if eris is None:
        # Note that if no MO energies are passed in, we construct them from the 
        # get_fock function without with_exxdiv=False. For PBC CCSD, this may be different
        # behaviour.
        mo_energy = einsum('ai,ab,bi->i', cluster.c_active, emb.get_fock(), cluster.c_active)
    else:
        mo_energy = eris.mo_energy

    if (len(external_corrections) > 1) and any([corr[2] == 0 for corr in external_corrections]):
        # We are externally correcting from multiple fragments, but not projecting them
        # into their fragment spaces. This means we are at risk of double-counting the external
        # corrections.
        solver.log.warn("Multiple external correcting fragments, but not fragment-projecting the resulting correction!")
        solver.log.warn("This will likely lead to double-counting of external correction.")
        solver.log.warn("Are you sure you want to do this?!")

    # CCSD uses exxdiv-uncorrected Fock matrix for residuals
    fock = emb.get_fock(with_exxdiv=False)

    if any([corr[3] and corr[1] == 'external' for corr in external_corrections]):
        # At least one fragment is externally corrected, *and* contracted with
        # integrals in the parent (i.e. CCSD) cluster. We can take the integrals from eris
        # if passed in. Otherwise, form the required integrals
        # for this parent cluster. Note that not all of these are needed.
        if eris is None:
            _, govvv_x, gvvov_x, gooov_x, govoo_x = _integrals_for_extcorr(fx, fock)
        else:
            if emb.spinsym == 'restricted':
                govvv_x = None
                gvvov_x = np.array(ao2mo_helper.get_ovvv(eris)).transpose(2,3,0,1)
                gooov_x = np.array(eris.ovoo).transpose(2,3,0,1)
                govoo_x = np.array(eris.ovoo)
            elif emb.spinsym == 'unrestricted':
                govvv_x = (np.array(ao2mo_helper.get_ovvv(eris, block="ovvv")),
                           np.array(ao2mo_helper.get_ovvv(eris, block="ovVV")),
                           np.array(ao2mo_helper.get_ovvv(eris, block="OVVV")))
                gvvov_x = (np.array(ao2mo_helper.get_ovvv(eris, block="ovvv")).transpose(2,3,0,1),
                           np.array(ao2mo_helper.get_ovvv(eris, block="OVvv")).transpose(2,3,0,1),
                           np.array(ao2mo_helper.get_ovvv(eris, block="OVVV")).transpose(2,3,0,1))
                gooov_x = (np.array(eris.ovoo).transpose(2,3,0,1),
                           np.array(eris.OVoo).transpose(2,3,0,1),
                           np.array(eris.OVOO).transpose(2,3,0,1))
                govoo_x = (np.array(eris.ovoo),
                           np.array(eris.ovOO),
                           np.array(eris.OVOO))
    
    # delta-T1 and delta-T2 amplitudes, to be added to the CCSD amplitudes
    if solver.spinsym == 'restricted':
        dt1 = np.zeros((nocc, nvir))
        dt2 = np.zeros((nocc, nocc, nvir, nvir))
    elif solver.spinsym == 'unrestricted':
        dt1 = (np.zeros((nocc[0], nvir[0])),
               np.zeros((nocc[1], nvir[1])))
        dt2 = (np.zeros((nocc[0], nocc[0], nvir[0], nvir[0])),
               np.zeros((nocc[0], nocc[1], nvir[0], nvir[1])),
               np.zeros((nocc[1], nocc[1], nvir[1], nvir[1])))

    frag_dir = {f.id: f for f in emb.fragments}
    for y, corrtype, projectors, low_level_coul in external_corrections:

        fy = frag_dir[y] # Get fragment y object from its index
        assert (y != fx.id)

        if corrtype == 'external':
            if low_level_coul:
                dt1y, dt2y = _get_delta_t_for_extcorr(fy, fock, solver, include_t3v=False)
            else:
                dt1y, dt2y = _get_delta_t_for_extcorr(fy, fock, solver, include_t3v=True)
        elif corrtype == 'delta-tailor':
            dt1y, dt2y = _get_delta_t_for_delta_tailor(fy, fock)
        else:
            raise ValueError

        # Project T1 and T2 corrections:
        if projectors:
            # projectors is an integer giving the number of projectors onto
            # occupied fragment
            proj = fy.get_overlap('frag|cluster-occ')
            proj = spinalg.dot(spinalg.T(proj), proj)
            dt1y = spinalg.dot(proj, dt1y)
            dt2y = project_t2(dt2y, proj, projectors=projectors)

        # Transform back to fragment x space
        rxy_occ = spinalg.dot(cxs_occ, fy.cluster.c_active_occ)
        rxy_vir = spinalg.dot(cxs_vir, fy.cluster.c_active_vir)
        dt1y = transform_amplitude(dt1y, rxy_occ, rxy_vir, inverse=True)
        dt2y = transform_amplitude(dt2y, rxy_occ, rxy_vir, inverse=True)
        dt1 = spinalg.add(dt1, dt1y)
        dt2 = spinalg.add(dt2, dt2y)

        if low_level_coul and corrtype == 'external': 
            # Include the t3v term, contracting with the integrals from the x cluster
            # These have already been fragment projected, and rotated into the x cluster
            # in this function.
            dt2y_t3v = _get_delta_t2_from_t3v(govvv_x, gvvov_x, gooov_x, govoo_x, fy,
                                              rxy_occ, rxy_vir, cxs_occ, projectors)
            dt2 = spinalg.add(dt2, dt2y_t3v)

        solver.log.info("External correction residuals from fragment %3d (%s via %s):  dT1= %.3e  dT2= %.3e",
                        fy.id, fy.solver, corrtype, *get_amplitude_norm(dt1y, dt2y))
    
    if solver.spinsym == 'restricted':
        
        if corrtype == 'external':
            # Contract with fragment x (CCSD) energy denominators
            # Note that this will not work correctly if a level shift used
            eia = mo_energy[:nocc, None] - mo_energy[None, nocc:]
            eijab = pyscf.lib.direct_sum('ia,jb->ijab', eia, eia)
            dt1 /= eia
            dt2 /= eijab

        solver.log.info("Total external correction amplitudes from all fragments:  dT1= %.3e  dT2= %.3e",
                *get_amplitude_norm(dt1, dt2))

        def callback(kwargs):
            """Add external correction to T1 and T2 amplitudes."""
            t1, t2 = kwargs['t1new'], kwargs['t2new']
            t1[:] += dt1
            t2[:] += dt2

    elif solver.spinsym == 'unrestricted':

        if corrtype == "external":
            eia_a = mo_energy[0][:nocc[0], None] - mo_energy[0][None, nocc[0]:]
            eia_b = mo_energy[1][:nocc[1], None] - mo_energy[1][None, nocc[1]:]
            eijab_aa = pyscf.lib.direct_sum('ia,jb->ijab', eia_a, eia_a)
            eijab_ab = pyscf.lib.direct_sum('ia,jb->ijab', eia_a, eia_b)
            eijab_bb = pyscf.lib.direct_sum('ia,jb->ijab', eia_b, eia_b)

            dt1 = (dt1[0] / eia_a, dt1[1] / eia_b)
            dt2 = (dt2[0] / eijab_aa, dt2[1] / eijab_ab, dt2[2] / eijab_bb)

        solver.log.info("Total external correction amplitudes from all fragments:  dT1= %.3e  dT2= %.3e", \
                *get_amplitude_norm(dt1, dt2))

        def callback(kwargs):
            """Add external correction to T1 and T2 amplitudes."""
            t1, t2 = kwargs['t1new'], kwargs['t2new']

            t1[0][:] += dt1[0]
            t1[1][:] += dt1[1]
            t2[0][:] += dt2[0]
            t2[1][:] += dt2[1]
            t2[2][:] += dt2[2]

    return callback
