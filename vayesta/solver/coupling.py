import numpy as np
from vayesta.core.util import *
from vayesta.core import spinalg
from vayesta.mpi import mpi, RMA_Dict


def transform_amplitude(t, u_occ, u_vir, u_occ2=None, u_vir2=None, spinsym='restricted', inverse=False):
    """(Old basis|New basis)"""
    if u_occ2 is None:
        u_occ2 = u_occ
    if u_vir2 is None:
        u_vir2 = u_vir
    if spinsym == 'restricted':
        if np.ndim(t) == 2:
            if inverse:
                return einsum('ia,xi,ya->xy', t, u_occ, u_vir)
            else:
                return einsum('ia,xi,ya->xy', t, u_occ, u_vir)
        if np.ndim(t) == 4:
            if inverse:
                return einsum('ijab,xi,yj,za,wb->xyzw', t, u_occ, u_occ2, u_vir, u_vir2)
            else:
                return einsum('ijab,ix,jy,az,bw->xyzw', t, u_occ, u_occ2, u_vir, u_vir2)
    if spinsym == 'unrestricted':
        if np.ndim(t[0]) == 2:
            ta = transform_amplitude(t[0], u_occ[0], u_vir[0], inverse=inverse)
            tb = transform_amplitude(t[1], u_occ[1], u_vir[1], inverse=inverse)
            return (ta, tb)
        if np.ndim(t[0]) == 4:
            taa = transform_amplitude(t[0], u_occ[0], u_vir[0], inverse=inverse)
            tab = transform_amplitude(t[1], u_occ[0], u_vir[0], u_occ[1], u_vir[1], inverse=inverse)
            tbb = transform_amplitude(t[2], u_occ[1], u_vir[1], inverse=inverse)
            return (taa, tab, tbb)
    raise NotImplementedError("Transformation of %s amplitudes with ndim=%d" % (spinsym, np.ndim(t[0])+1))


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
    mode : int, optional
        Level of external correction of T2 amplitudes:
        1: Both occupied indices are projected to each other fragment X.
        2: Both occupied indices are projected to each other fragment X
           and combinations of other fragments X,Y.
        3: Only the first occupied indices is projected to each other fragment X.
    coupled_fragments : list, optional
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
    fx = solver.fragment
    cx = solver.cluster
    ovlp = solver.base.get_ovlp()   # AO overlap matrix
    c_occ = cluster.c_active_occ    # Occupied active orbitals of current cluster
    c_vir = cluster.c_active_vir    # Virtual  active orbitals of current cluster
    cs_occ = spinalg.dot(spinalg.tranpose(c_occ), ovlp)
    cs_vir = spinalg.dot(spinalg.tranpose(c_vir), ovlp)

    def _tailor_func(kwargs):
        """Add external correction to T1 and T2 amplitudes."""
        t1, t2 = kwargs['t1new'], kwargs['t2new']
        # Add the correction to dt1 and dt2:
        if tailor_t1: dt1 = np.zeros_like(t1)
        if tailor_t2: dt2 = np.zeros_like(t2)

        # Loop over all *other* fragments/cluster X
        for fy in fragments:
            assert (fy is not solver.fragment)

            # Rotation & projections from cluster X active space to current fragment active space
            rxy_occ = spinalg.dot(cx_occ, fy.c_active_occ)
            rxy_vir = spinalg.dot(cx_vir, fy.c_active_vir)
            # Skip fragment if there is no overlap
            if min(abs(rxy_occ).max(), abs(rxy_vir).max()) < ovlp_tol:
                continue

            wfy = fy.results.wf.as_ccsd()
            if tailor_t1:
                #t1y = transform_amplitude(wfy.t1, rxy_occ, rxy_vir, spinsym=solver.spinsym, inverse=True)
                t1x = transform_amplitude(t1, rxy_occ, rxy_vir, spinsym=solver.spinsym)
                dt1 = (wfy.t1 - t1x)
            if tailor_t2:
                t2x = transform_amplitude(t2, rxy_occ, rxy_vir, spinsym=solver.spinsym)
                dt2 = (wfy.t2 - t2x)






            px = x.get_fragment_projector(c_occ)    # this is C_occ^T . S . C_frag . C_frag^T . S . C_occ
            assert np.allclose(px, px.T)
            #pxv = x.get_fragment_projector(c_vir)   # this is C_occ^T . S . C_frag . C_frag^T . S . C_occ
            #assert np.allclose(pxv, pxv.T)
            if x.results.t1 is None and x.results.c1 is not None:
                solver.log.debugv("Converting C-amplitudes of %s to T-amplitudes", x)
                x.results.convert_amp_c_to_t()
            # Transform fragment X T-amplitudes to current active space and form difference
            if correct_t1:
                tx1 = transform_amplitude(x.results.t1, p_occ, p_vir)   # ia,ix,ap->xp
                #tx1[:] = 0
                dtx1 = (tx1 - t1)
                dtx1 = np.dot(px, dtx1)
                #dtx1o = np.dot(px, dtx1)
                #dtx1v = np.dot(dtx1, pxv)
                #dtx1 = dtx1o + dtx1v - np.linalg.multi_dot((px, dtx1, pxv))
                assert dtx1.shape == dt1.shape
                dt1 += dtx1
            else:
                dtx1 = 0
            if correct_t2:
                tx2 = transform_amplitude(x.results.t2, p_occ, p_vir)   # ijab,ix,jy,ap,bq->xypq
                #tx2[:] = 0
                dtx2 = (tx2 - t2)
                if mode == 1:
                    dtx2 = einsum('xi,yj,ijab->xyab', px, px, dtx2)
                    #dtx2 = einsum('xi,yj,ijab,pa,qb->xypq', px, px, dtx2, pxv, pxv)
                    #dtx2o = einsum('xi,yj,ijab->xyab', px, px, dtx2)
                    #dtx2v = einsum('xa,yb,ijab->ijxy', pxv, pxv, dtx2)
                    #dtx2dc = einsum('xi,yj,ijab,pa,qb->xypq', px, px, dtx2, pxv, pxv)
                    #dtx2 = dtx2o + dtx2v - dtx2dc
                elif mode == 2:
                    py = solver.fragment.get_fragment_projector(c_occ, inverse=True)
                    dtx2 = einsum('xi,yj,ijab->xyab', px, py, dtx2)
                elif mode == 3:
                    dtx2 = einsum('xi,ijab->xjab', px, dtx2)
                assert dtx2.shape == dt2.shape
                dt2 += dtx2
            else:
                dtx2 = 0
            solver.log.debugv("Tailoring %12s <- %12s: |dT1|= %.2e  |dT2|= %.2e", solver.fragment, x, np.linalg.norm(dtx1), np.linalg.norm(dtx2))
            #print(t2loc(dt2, 0))
            #print(t2loc(dt2, 1))


        # Store these norms in cc, to log their final value:
        cc._norm_dt1 = np.linalg.norm(dt1) if correct_t1 else 0.0
        cc._norm_dt2 = np.linalg.norm(dt2) if correct_t2 else 0.0
        # Add correction:
        if correct_t1:
            t1 = (t1 + dt1)
        if correct_t2:
            if symmetrize_t2:
                solver.log.debugv("T2 symmetry error: %e", np.linalg.norm(dt2 - dt2.transpose(1,0,3,2))/2)
                dt2 = (dt2 + dt2.transpose(1,0,3,2))/2
            t2 = (t2 + dt2)

        #print(np.linalg.norm(t1))
        #print(np.linalg.norm(t2))
        #1/0

        return t1, t2


    def tailor_func(cc, t1, t2):
        """Add external correction to T1 and T2 amplitudes."""
        # Add the correction to dt1 and dt2:
        if correct_t1: dt1 = np.zeros_like(t1)
        if correct_t2: dt2 = np.zeros_like(t2)

        t1[:] = 0
        t2[:] = 0

        mo_coeff = np.hstack((c_occ, c_vir))
        nsite = mo_coeff.shape[0]
        t1 = np.zeros((nsite, nsite))
        t2 = np.zeros((nsite, nsite, nsite, nsite))
        diag = list(range(nsite))
        val = 1000.0
        t1[diag,diag] = val
        t2[diag,diag,diag,diag] = val

        p_occ = np.dot(c_occ, c_occ.T)
        p_vir = np.dot(c_vir, c_vir.T)

        e, v = np.linalg.eigh(p_occ)
        print(e)
        e, v = np.linalg.eigh(p_vir)
        print(e)

        print(t2[0,0,0,0])
        print(t2[0,0,0,1])

        # Make "physical"
        t1 = einsum('xy,xi,ya->ia', t1, p_occ, p_vir)
        t2 = einsum('xyzw,xi,yj,za,wb->ijab', t2, p_occ, p_occ, p_vir, p_vir)

        print(t2[0,0,0,0])
        print(t2[0,0,0,1])
        1/0


        t1 = einsum('xy,xi,ya->ia', t1, c_occ, c_vir)
        t2 = einsum('xyzw,xi,yj,za,wb->ijab', t2, c_occ, c_occ, c_vir, c_vir)

        def t1loc(t1, site=0):
            t1 = einsum('xz,xi,za->ia', t1, c_occ.T, c_vir.T)
            return t1[site,site]

        def t2loc(t2, site=0):
            t2 = einsum('xyzw,xi,yj,za,wb->ijab', t2, c_occ.T, c_occ.T, c_vir.T, c_vir.T)
            return t2
            #return t2[site,site,site,site+1]

        t2l = t2loc(t2)
        print(t2l[0,0,0,0])
        print(t2l[0,0,0,1])
        1/0

        print(t2loc(t2, 0))
        print(t2loc(t2, 1))
        1/0

        for x in coupled_fragments:
            p_occ = np.linalg.multi_dot((x.c_active_occ.T, ovlp, c_occ))
            #p = np.dot(p_occ.T, p_occ)
            dt = np.einsum('xi,yj,ijab->xyab', p_occ, p_occ, t2)
            px = x.get_fragment_projector(x.c_active_occ)    # this is C_occ^T . S . C_frag . C_frag^T . S . C_occ
            #dt = np.einsum('xi,yj,ijab->xyab', px, px, dt)
            dt = np.einsum('xi,ijab->xjab', px, dt)
            dt = -np.einsum('xi,yj,ijab->xyab', p_occ.T, p_occ.T, dt)
            dt2 += dt
            print(t2loc(dt2, 0))
            print(t2loc(dt2, 1))

        1/0

        # Loop over all *other* fragments/cluster X
        for x in coupled_fragments:
            assert (x is not solver.fragment)

            # Rotation & projections from cluster X active space to current fragment active space
            p_occ = np.linalg.multi_dot((x.c_active_occ.T, ovlp, c_occ))
            p_vir = np.linalg.multi_dot((x.c_active_vir.T, ovlp, c_vir))
            px = x.get_fragment_projector(x.c_active_occ)    # this is C_occ^T . S . C_frag . C_frag^T . S . C_occ
            assert np.allclose(px, px.T)
            #assert np.allclose(pxv, pxv.T)
            if x.results.t1 is None and x.results.c1 is not None:
                solver.log.debugv("Converting C-amplitudes of %s to T-amplitudes", x)
                tx1, tx2 = x.results.convert_amp_c_to_t()
            else:
                tx1, tx2 = x.results.t1, x.results.t2
            tx1 = tx1.copy()
            tx2 = tx2.copy()
            tx1[:] = 0
            tx2[:] = 0

            # Project CC T amplitudes to cluster space of fragment X
            if correct_t1:
                tx1_cc = transform_amplitude(t1, p_occ.T, p_vir.T)
                dtx1 = (tx1 - tx1_cc)
                dtx1 = np.dot(px, dtx1)
                dtx1 = transform_amplitude(dtx1, p_occ, p_vir)   # Transform back
                assert dtx1.shape == dt1.shape
                dt1 += dtx1
            else:
                dtx1 = 0
            if correct_t2:
                tx2_cc = transform_amplitude(t2, p_occ.T, p_vir.T)   # ijab,ix,jy,ap,bq->xypq
                dtx2 = (tx2 - tx2_cc)
                if mode == 1:
                    dtx2 = einsum('xi,yj,ijab->xyab', px, px, dtx2)
                elif mode == 2:
                    env = solver.fragment.get_fragment_projector(x.c_active_occ, inverse=True)
                    dtx2 = einsum('xi,yj,ijab->xyab', px, env, dtx2)
                elif mode == 3:
                    dtx2 = einsum('xi,ijab->xjab', px, dtx2)
                dtx2 = transform_amplitude(dtx2, p_occ, p_vir)   # Transform back
                assert dtx2.shape == dt2.shape
                dt2 += dtx2
            else:
                dtx2 = 0
            solver.log.debugv("Tailoring %12s <- %12s: |dT1|= %.2e  |dT2|= %.2e", solver.fragment, x, np.linalg.norm(dtx1), np.linalg.norm(dtx2))
            print(t2loc(dt2, 0))
            print(t2loc(dt2, 1))


        # Store these norms in cc, to log their final value:
        cc._norm_dt1 = np.linalg.norm(dt1) if correct_t1 else 0.0
        cc._norm_dt2 = np.linalg.norm(dt2) if correct_t2 else 0.0
        # Add correction:
        if correct_t1:
            t1 = (t1 + dt1)
        if correct_t2:
            if symmetrize_t2:
                solver.log.debugv("T2 symmetry error: %e", np.linalg.norm(dt2 - dt2.transpose(1,0,3,2))/2)
                dt2 = (dt2 + dt2.transpose(1,0,3,2))/2
            t2 = (t2 + dt2)

        print(np.linalg.norm(t1))
        print(np.linalg.norm(t2))
        1/0

        return t1, t2

    return tailor_func_old
