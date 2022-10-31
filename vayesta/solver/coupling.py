import numpy as np
from vayesta.core.util import *
from vayesta.core import spinalg
from vayesta.mpi import mpi, RMA_Dict


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
        govov = eris[occ,vir,occ,vir]
        fov = dot(cluster.c_active_occ.T, fock, cluster.c_active_vir)
    if emb.spinsym == 'unrestricted':
        oa = np.s_[:cluster.nocc_active[0]]
        ob = np.s_[:cluster.nocc_active[1]]
        va = np.s_[cluster.nocc_active[0]:]
        vb = np.s_[cluster.nocc_active[1]:]
        fova = dot(cluster.c_active_occ[0].T, fock[0], cluster.c_active_vir[0])
        fovb = dot(cluster.c_active_occ[1].T, fock[1], cluster.c_active_vir[1])
        govovaa = eris[oa,va,oa,va]
        govovab = eris[oa,va,ob,vb]
        govovbb = eris[ob,vb,ob,vb]
        fov = (fova, fovb)
        govov = (govovaa, govovab, govovbb)
    return fov, govov


def _get_delta_t_for_extcorr(fragment, fock):
    emb = fragment.base

    # Make CCSDTQ wave function from cluster y
    wf = fragment.results.wf.as_ccsdtq()
    t1, t2, t3, t4 = wf.t1, wf.t2, wf.t3, wf.t4

    # Get ERIs and Fock matrix
    fov, govov = _integrals_for_extcorr(fragment, fock)
    # --- Make correction to T1 and T2 amplitudes
    # J. Chem. Theory Comput. 2021, 17, 182−190
    # [what is v_ef^mn? (em|fn) ?]
    dt1 = spinalg.zeros_like(t1)
    dt2 = spinalg.zeros_like(t2)

    raise NotImplementedError
    if emb.spinsym == 'restricted':
        # --- T1
        # T3 * V
        #dt1 += einsum('imnaef,menf->ia', t3, govov)
        # --- T2
        # F * T3
        #dt2 += einsum('me,ijmabe->ijab', fov, t3)
        # T1 * T3 * V
        #dt2 += einsum('me,ijnabf,menf->ijab', t1, t3, govov)
        # TODO:
        # P(ab) T3 * V
        # P(ij) T3 * V
        # P(ij) T1 * T3 * V
        # P(ab) T1 * T3 * V
        # T4 * V
        pass
    elif emb.spinsym == 'unrestricted':
        # TODO
        pass
    else:
        raise ValueError

    return dt1, dt2


def externally_correct(solver, external_corrections):
    """Build callback function for CCSD, to add external correction from other fragments.

    TODO: combine with `tailor_with_fragments`?

    Parameters
    ----------
    solver: CCSD_Solver
        Vayesta CCSD solver.
    external_corrections: list[tuple(int, str, int)]
        List of external corrections. Each tuple contains the fragment ID, type of correction,
        and number of projectors for the given external correction.

    Returns
    -------
    callback: callable
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
    # CCSD uses exxdiv-uncorrected Fock matrix:
    fock = emb.get_fock(with_exxdiv=False)

    for y, corrtype, projectors in external_corrections:

        assert corrtype == 'external'
        fy = frag_dir[y]
        assert (y != fx.id)

        dt1y, dt2y = _get_delta_t_for_extcorr(fy, fock)

        # Project T1 and T2 corrections:
        if projectors:
            proj = fy.get_overlap('frag|cluster-occ')
            proj = spinalg.dot(spinalg.T(proj), proj)
            dt1y = spinalg.dot(proj, dt1y)
            dt2y = project_t2(dt2y, proj, projectors=projectors)

        # Transform back to fragment x space and add:
        rxy_occ = spinalg.dot(cxs_occ, fy.cluster.c_active_occ)
        rxy_vir = spinalg.dot(cxs_vir, fy.cluster.c_active_vir)
        dt1y = transform_amplitude(dt1y, rxy_occ, rxy_vir, inverse=True)
        dt2y = transform_amplitude(dt2y, rxy_occ, rxy_vir, inverse=True)
        dt1 = spinalg.add(dt1, dt1y)
        dt2 = spinalg.add(dt2, dt2y)
        solver.log.info("External correction from fragment %3d (%s):  dT1= %.3e  dT2= %.3e",
                        fy.id, fy.solver, *get_amplitude_norm(dt1y, dt2y))

    if solver.spinsym == 'restricted':

        def callback(kwargs):
            """Add external correction to T1 and T2 amplitudes."""
            t1, t2 = kwargs['t1new'], kwargs['t2new']
            t1[:] += dt1
            t2[:] += dt2

    elif solver.spinsym == 'unrestricted':

        def callback(kwargs):
            """Add external correction to T1 and T2 amplitudes."""
            t1, t2 = kwargs['t1new'], kwargs['t2new']
            t1[0][:] += dt1[0]
            t1[1][:] += dt1[1]
            t2[0][:] += dt2[0]
            t2[1][:] += dt2[1]
            t2[2][:] += dt2[2]

    return callback
