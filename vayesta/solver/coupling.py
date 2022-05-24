import numpy as np

import pyscf
import pyscf.lib
import pyscf.ao2mo
import pyscf.ci
import pyscf.fci

from vayesta.core.util import *
from vayesta.core.mpi import mpi, RMA_Dict

def transform_amplitude(t, u_occ, u_vir):
    """(Old basis|new basis)"""
    if np.ndim(t) == 2:
        return einsum("ia,ix,ay->xy", t, u_occ, u_vir)
    if np.ndim(t) == 4:
        return einsum("ijab,ix,jy,az,bw->xyzw", t, u_occ, u_occ, u_vir, u_vir)
    raise NotImplementedError('Transformation of amplitudes with ndim=%d' % np.ndim(t))


def make_cas_tcc_function(solver, c_cas_occ, c_cas_vir, eris):
    """Make tailor function for Tailored CC."""

    cluster = solver.cluster

    ncasocc = c_cas_occ.shape[-1]
    ncasvir = c_cas_vir.shape[-1]
    ncas = ncasocc + ncasvir
    nelec = 2*ncasocc

    solver.log.info("Running FCI in (%d, %d) CAS", nelec, ncas)

    c_cas = np.hstack((c_cas_occ, c_cas_vir))
    ovlp = solver.base.get_ovlp()

    # Rotation & projection into CAS
    ro = dot(cluster.c_active_occ.T, ovlp, c_cas_occ)
    rv = dot(cluster.c_active_vir.T, ovlp, c_cas_vir)
    r = np.block([[ro, np.zeros((ro.shape[0], rv.shape[1]))],
                  [np.zeros((rv.shape[0], ro.shape[1])), rv]])

    o = np.s_[:ncasocc]
    v = np.s_[ncasocc:]

    def make_cas_eris(eris):
        """Make 4c ERIs in CAS."""
        t0 = timer()
        g_cas = np.zeros(4*[ncas])
        # 0 v
        g_cas[o,o,o,o] = einsum('IJKL,Ii,Jj,Kk,Ll->ijkl', eris.oooo[:], ro, ro, ro, ro)
        # 1 v
        g_cas[o,v,o,o] = einsum('IJKL,Ii,Jj,Kk,Ll->ijkl', eris.ovoo[:], ro, rv, ro, ro)
        g_cas[v,o,o,o] = g_cas[o,v,o,o].transpose(1,0,3,2)
        g_cas[o,o,o,v] = g_cas[o,v,o,o].transpose(2,3,0,1)
        g_cas[o,o,v,o] = g_cas[o,o,o,v].transpose(1,0,3,2)
        # 2 v
        g_cas[o,o,v,v] = einsum('IJKL,Ii,Jj,Kk,Ll->ijkl', eris.oovv[:], ro, ro, rv, rv)
        g_cas[v,v,o,o] = g_cas[o,o,v,v].transpose(2,3,0,1)
        g_cas[o,v,o,v] = einsum('IJKL,Ii,Jj,Kk,Ll->ijkl', eris.ovov[:], ro, rv, ro, rv)
        g_cas[v,o,v,o] = g_cas[o,v,o,v].transpose(1,0,3,2)
        g_cas[o,v,v,o] = einsum('IJKL,Ii,Jj,Kk,Ll->ijkl', eris.ovvo[:], ro, rv, rv, ro)
        g_cas[v,o,o,v] = g_cas[o,v,v,o].transpose(2,3,0,1)
        # 3 v
        nocc = cluster.nocc_active
        nvir = cluster.nvir_active
        if eris.ovvv.ndim == 3:
            nvir_pair = nvir*(nvir+1)//2
            g_ovvv = pyscf.lib.unpack_tril(eris.ovvv.reshape(nocc*nvir, nvir_pair)).reshape(nocc,nvir,nvir,nvir)
        else:
            g_ovvv = eris.ovvv[:]
        g_cas[o,v,v,v] = einsum('IJKL,Ii,Jj,Kk,Ll->ijkl', g_ovvv, ro, rv, rv, rv)
        g_cas[v,o,v,v] = g_cas[o,v,v,v].transpose(1,0,3,2)
        g_cas[v,v,o,v] = g_cas[o,v,v,v].transpose(2,3,0,1)
        g_cas[v,v,v,o] = g_cas[v,v,o,v].transpose(1,0,3,2)
        # 4 v
        if hasattr(eris, 'vvvv') and eris.vvvv is not None:
            if eris.vvvv.ndim == 2:
                g_vvvv = pyscf.ao2mo.restore(1, np.asarray(eris.vvvv), nvir)
            else:
                g_vvvv = eris.vvvv[:]
            g_cas[v,v,v,v] = einsum('IJKL,Ii,Jj,Kk,Ll->ijkl', g_vvvv, rv, rv, rv, rv)
        # Note that this will not work for 2D systems!:
        elif hasattr(eris, 'vvL') and eris.vvL is not None:
            if eris.vvL.ndim == 2:
                naux = eris.vvL.shape[-1]
                vvl = pyscf.lib.unpack_tril(eris.vvL, axis=0).reshape(nvir,nvir,naux)
            else:
                vvl = eris.vvL
            vvl = einsum('IJQ,Ii,Jj->ijQ', vvl, rv, rv)
            g_cas[v,v,v,v] = einsum('ijQ,klQ->ijkl', vvl.conj(), vvl)
        else:
            raise RuntimeError("ERIs has not attribute 'vvvv' or 'vvL'.")
        solver.log.timingv("Time to make CAS ERIs: %s", time_string(timer()-t0))
        return g_cas

    g_cas = make_cas_eris(eris)
    # For the FCI, we need an effective one-electron Hamiltonian,
    # which contains Coulomb and exchange interaction to all frozen occupied orbitals
    # To calculate this, we would in principle need the whole-system 4c-integrals
    # Instead, we can start from the full system Fock matrix, which we already know
    # and subtract the parts NOT due to the frozen core density:
    # This Fock matrix does NOT contain exxdiv correction!
    #f_act = np.linalg.multi_dot((c_cas.T, eris.fock, c_cas))
    f_act = dot(r.T, eris.fock, r)
    v_act = 2*einsum('iipq->pq', g_cas[o,o]) - einsum('iqpi->pq', g_cas[o,:,:,o])
    h_eff = f_act - v_act

    #fcisolver = pyscf.fci.direct_spin0.FCISolver(solver.mol)
    fcisolver = pyscf.fci.direct_spin1.FCISolver(solver.mol)
    solver.opts.tcc_fci_opts['max_cycle'] = solver.opts.tcc_fci_opts.get('max_cycle', 1000)
    fix_spin = solver.opts.tcc_fci_opts.pop('fix_spin', 0)
    if fix_spin not in (None, False):
        solver.log.debugv("Fixing spin of FCIsolver to S^2= %r", fix_spin)
        fcisolver = pyscf.fci.addons.fix_spin_(fcisolver, ss=fix_spin)
    for key, val in solver.opts.tcc_fci_opts.items():
        solver.log.debugv("Setting FCIsolver attribute %s to %r", key, val)
        setattr(fcisolver, key, val)

    t0 = timer()
    e_fci, wf0 = fcisolver.kernel(h_eff, g_cas, ncas, nelec)
    solver.log.timing("Time for FCI: %s", time_string(timer()-t0))
    if not fcisolver.converged:
        solver.log.error("FCI not converged!")
    # Get C0,C1,and C2 from WF
    cisdvec = pyscf.ci.cisd.from_fcivec(wf0, ncas, nelec)
    c0, c1, c2 = pyscf.ci.cisd.cisdvec_to_amplitudes(cisdvec, ncas, ncasocc)
    solver.log.info("FCI weight on reference determinant: %.8g", abs(c0))
    if abs(c0) < 1e-4:
        solver.log.warning("Weight on reference determinant small!")
    if (c0 == 0):
        msg = "FCI wave function has no overlap with HF determinant."
        solver.log.critical(msg)
        raise RuntimeError(msg)
    # Intermediate normalization
    c1 /= c0
    c2 /= c0
    t1_fci = c1
    t2_fci = c2 - einsum('ia,jb->ijab', c1, c1)

    def tailor_func(kwargs):
        t1, t2 = kwargs['t1new'], kwargs['t2new']
        # Rotate & project CC amplitudes to CAS
        t1_cc = einsum('IA,Ii,Aa->ia', t1, ro, rv)
        t2_cc = einsum('IJAB,Ii,Jj,Aa,Bb->ijab', t2, ro, ro, rv, rv)
        # Take difference wrt to FCI
        dt1 = (t1_fci - t1_cc)
        dt2 = (t2_fci - t2_cc)
        # Rotate back to CC space
        dt1 = einsum('ia,Ii,Aa->IA', dt1, ro, rv)
        dt2 = einsum('ijab,Ii,Jj,Aa,Bb->IJAB', dt2, ro, ro, rv, rv)
        # Add correction
        t1 += dt1
        t2 += dt2
        cc._norm_dt1 = np.linalg.norm(dt1)
        cc._norm_dt2 = np.linalg.norm(dt2)

    return tailor_func


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
    c_occ_x = solver.cluster.c_active_occ
    c_vir_x = solver.cluster.c_active_vir
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
        cc.force_exit = bool(mpi.world.allreduce(int(cc.conv_flag), op=mpi.op.prod))
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


def make_cross_fragment_tcc_function(solver, mode, coupled_fragments=None, correct_t1=True, correct_t2=True, symmetrize_t2=True):
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
    if mode not in (1, 2, 3):
        raise ValueError()
    solver.log.debugv("TCC mode= %d", mode)
    cluster = solver.cluster
    ovlp = solver.base.get_ovlp()     # AO overlap matrix
    c_occ = cluster.c_active_occ       # Occupied active orbitals of current cluster
    c_vir = cluster.c_active_vir       # Virtual  active orbitals of current cluster

    if coupled_fragments is None:
        coupled_fragments = solver.fragment.opts.coupled_fragments

    #mode = 1

    def tailor_func_old(cc, t1, t2):
        """Add external correction to T1 and T2 amplitudes."""
        # Add the correction to dt1 and dt2:
        if correct_t1: dt1 = np.zeros_like(t1)
        if correct_t2: dt2 = np.zeros_like(t2)

        ##t1[:] = 0
        ##t2[:] = 0

        ##mo_coeff = np.hstack((c_occ, c_vir))
        ##nsite = mo_coeff.shape[0]
        ##t1 = np.zeros((nsite, nsite))
        ##t2 = np.zeros((nsite, nsite, nsite, nsite))
        ##diag = list(range(nsite))
        ##val = 1000.0
        ##t1[diag,diag] = val
        ##t2[diag,diag,diag,diag] = val
        ##t1 = einsum('xy,xi,ya->ia', t1, c_occ, c_vir)
        ##t2 = einsum('xyzw,xi,yj,za,wb->ijab', t2, c_occ, c_occ, c_vir, c_vir)

        ##def t2loc(t2, site=0):
        ##    t2 = einsum('xyzw,xi,yj,za,wb->ijab', t2, c_occ.T, c_occ.T, c_vir.T, c_vir.T)
        ##    return t2[site,site,site,site]

        ##print(t2loc(t2, 0))
        ##print(t2loc(t2, 1))

        # Loop over all *other* fragments/cluster X
        for x in coupled_fragments:
            assert (x is not solver.fragment)

            # Rotation & projections from cluster X active space to current fragment active space
            p_occ = np.linalg.multi_dot((x.c_active_occ.T, ovlp, c_occ))
            p_vir = np.linalg.multi_dot((x.c_active_vir.T, ovlp, c_vir))
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
