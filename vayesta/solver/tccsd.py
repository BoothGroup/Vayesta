import numpy as np
import pyscf
import pyscf.lib
import pyscf.ao2mo
import pyscf.ci
import pyscf.fci
from vayesta.core.util import dot, einsum, time_string, timer


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
        cc = kwargs['mycc']
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
