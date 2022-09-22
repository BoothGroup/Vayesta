import numpy as np
from vayesta.core.util import *
from vayesta.core import spinalg
from vayesta.mpi import mpi, RMA_Dict


def transform_amplitude(t, u_occ, u_vir, u_occ2=None, u_vir2=None, spinsym='restricted', inverse=False):
    """u: (old basis|new basis)"""
    if u_occ2 is None:
        u_occ2 = u_occ
    if u_vir2 is None:
        u_vir2 = u_vir
    if spinsym == 'restricted':
        if inverse:
            u_occ = u_occ.T
            u_occ2 = u_occ2.T
            u_vir = u_vir.T
            u_vir2 = u_vir2.T
        if np.ndim(t) == 2:
            return einsum('ia,ix,ay->xy', t, u_occ, u_vir)
        if np.ndim(t) == 4:
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
    fragment = solver.fragment
    cluster = solver.cluster
    ovlp = solver.base.get_ovlp()   # AO overlap matrix
    cx_occ = cluster.c_active_occ    # Occupied active orbitals of current cluster
    cx_vir = cluster.c_active_vir    # Virtual  active orbitals of current cluster
    cxs_occ = spinalg.dot(spinalg.T(cx_occ), ovlp)
    cxs_vir = spinalg.dot(spinalg.T(cx_vir), ovlp)

    def tailor_func(kwargs):
        """Add external correction to T1 and T2 amplitudes."""
        t1, t2 = kwargs['t1new'], kwargs['t2new']
        # Collect all changes to the amplitudes in dt1 and dt2:
        if tailor_t1:
            dt1 = spinalg.zeros_like(t1)
        if tailor_t2:
            dt2 = spinalg.zeros_like(t2)

        # Loop over all *other* fragments/cluster X
        for fy in fragments:
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
                continue

            wfy = fy.results.wf.as_ccsd()
            # Transform to x-amplitudes to y-space, instead of y-amplitudes to x-space:
            # x may be CCSD and y FCI, such that x-space >> y-space
            if tailor_t1:
                t1x = transform_amplitude(t1, rxy_occ, rxy_vir, spinsym=solver.spinsym)
                dt1y = spinalg.subtract(wfy.t1, t1x)
            if tailor_t2:
                t2x = transform_amplitude(t2, rxy_occ, rxy_vir, spinsym=solver.spinsym)
                dt2y = spinalg.subtract(wfy.t2, t2x)

            # Project
            if project:
                proj = fy.get_overlap('frag|cluster-occ')
                proj = spinalg.dot(spinalg.T(proj), proj)
                # Project first occupied index onto fragment(y) space:
                if int(project) == 1:
                    if tailor_t1:
                        dt1y = spinalg.dot(proj, dt1y)
                    if tailor_t2:
                        if solver.spinsym == 'restricted':
                            dt2y = einsum('xi,i...->x...', proj, dt2y)
                            dt2y = (dt2y + dt2y.transpose(1,0,3,2))/2
                        elif solver.spinsym == 'unrestricted':
                            dt2y_aa = einsum('xi,i...->x...', proj[0], dt2y[0])/2
                            dt2y_bb = einsum('xi,i...->x...', proj[1], dt2y[2])/2
                            dt2y_aa = (dt2y_aa + dt2y_aa.transpose(1,0,3,2))/2
                            dt2y_bb = (dt2y_bb + dt2y_bb.transpose(1,0,3,2))/2
                            dt2y_ab = (einsum('xi,i...->x...', proj[0], dt2y[1])
                                     + einsum('xj,ij...->x...', proj[1], dt2y[1]))
                            dt2y = (dt2y_aa, dt2y_ab, dt2y_bb)

                # Project first and second occupied index onto fragment(y) space:
                elif project == 2:
                    raise NotImplementedError
                else:
                    raise ValueError("project= %s" % project)

            # Transform back to x-space and add:
            if tailor_t1:
                dt1 += transform_amplitude(dt1y, rxy_occ, rxy_vir, spinsym=solver.spinsym, inverse=True)
            if tailor_t2:
                dt2 += transform_amplitude(dt2y, rxy_occ, rxy_vir, spinsym=solver.spinsym, inverse=True)
            if solver.spinsym == 'restricted':
                solver.log.debug("Tailoring %12s with %12s:  |dT1|= %.2e  |dT2|= %.2e", fragment, fy, np.linalg.norm(dt1), np.linalg.norm(dt2))
            elif solver.spinsym == 'unrestricted':
                solver.log.debug("Tailoring %12s with %12s:  |dT1|= %.2e  |dT2|= %.2e", fragment, fy,
                        (np.linalg.norm(dt1[0])+np.linalg.norm(dt1[1]))/2,
                        (np.linalg.norm(dt2[0])+2*np.linalg.norm(dt2[1])+np.linalg.norm(dt2[2]))/4)

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

    return tailor_func
