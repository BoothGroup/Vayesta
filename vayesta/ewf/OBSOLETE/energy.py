"""Local energy for EmbWF calculations.

These require a projection of the some indices of the C1 and C2
amplitudes.
"""

import numpy as np

from quanterm.core.util import *

__all__ = [
        "get_local_amplitudes",
        "get_local_amplitudes_general",
        "get_local_energy",
        ]


def project_amplitudes_to_fragment(self, cm, c1, c2, **kwargs):
    """Wrapper for project_amplitude_to_fragment, where the mo coefficients are extracted from a MP2 or CC object."""

    act = cm.get_frozen_mask()
    occ = cm.mo_occ[act] > 0
    vir = cm.mo_occ[act] == 0
    c = cm.mo_coeff[:,act]
    c_occ = c[:,occ]
    c_vir = c[:,vir]

    p1 = p2 = None
    if c1 is not None:
        p1 = self.project_amplitude_to_fragment(c1, c_occ, c_vir, **kwargs)
    if c2 is not None:
        p2 = self.project_amplitude_to_fragment(c2, c_occ, c_vir, **kwargs)
    return p1, p2


def project_amplitude_to_fragment(self, c, c_occ=None, c_vir=None, partitioning=None, symmetrize=False):
    """Get local contribution of amplitudes."""

    if np.ndim(c) not in (2, 4):
        raise NotImplementedError()
    if partitioning not in ('first-occ', 'first-vir', 'democratic'):
        raise ValueError("Unknown partitioning of amplitudes: %s", partitioning)

    # Projectors into fragment occupied and virtual space
    if part in ("first-occ", "democratic"):
        assert c_occ is not None
        fo = self.get_fragment_projector(c_occ)
    if part in ("first-vir", "democratic"):
        assert c_vir is not None
        fv = self.get_fragment_projector(c_vir)
    # Inverse projectors needed
    if part == "democratic":
        ro = np.eye(fo.shape[-1]) - fo
        rv = np.eye(fv.shape[-1]) - fv

    if np.ndim(c) == 2:
        if part == "first-occ":
            p = einsum("xi,ia->xa", fo, c)
        elif part == "first-vir":
            p = einsum("ia,xa->ix", c, fv)
        elif part == "democratic":
            p = einsum("xi,ia,ya->xy", fo, c, fv)
            p += einsum("xi,ia,ya->xy", fo, c, rv) / 2.0
            p += einsum("xi,ia,ya->xy", ro, c, fv) / 2.0
        return p

    # ndim == 4:

    if partitioning == "first-occ":
        p = einsum("xi,ijab->xjab", fo, c)
    elif part == "first-vir":
        p = einsum("ijab,xa->ijxb", c, fv)
    elif part == "democratic":

        def project(p1, p2, p3, p4):
            p = einsum("xi,yj,ijab,za,wb->xyzw", p1, p2, c, p3, p4)
            return p

        # Factors of 2 due to ij,ab <-> ji,ba symmetry
        # Denominators 1/N due to element being shared between N clusters

        # Quadruple F
        # ===========
        # This is fully included
        p = project(fo, fo, fv, fv)
        # Triple F
        # ========
        # This is fully included
        p += 2*project(fo, fo, fv, rv)
        p += 2*project(fo, ro, fv, fv)
        # Double F
        # ========
        # P(FFrr) [This wrongly includes: 1x P(FFaa), instead of 0.5x - correction below]
        p +=   project(fo, fo, rv, rv)
        p += 2*project(fo, ro, fv, rv)
        p += 2*project(fo, ro, rv, fv)
        p +=   project(ro, ro, fv, fv)
        # Single F
        # ========
        # P(Frrr) [This wrongly includes: P(Faar) (where r could be a) - correction below]
        p += 2*project(fo, ro, rv, rv) / 4.0
        p += 2*project(ro, ro, fv, rv) / 4.0

        # Corrections
        # ===========
        # Loop over all other clusters x
        for x in self.loop_fragments(exclude_self=True):

            xo = x.get_fragment_projector(c_occ)
            xv = x.get_fragment_projector(c_vir)

            # Double correction
            # -----------------
            # Correct for wrong inclusion of P(FFaa)
            # The case P(FFaa) was included with prefactor of 1 instead of 1/2
            # We thus need to only correct by "-1/2"
            p -=   project(fo, fo, xv, xv) / 2.0
            p -= 2*project(fo, xo, fv, xv) / 2.0
            p -= 2*project(fo, xo, xv, fv) / 2.0
            p -=   project(xo, xo, fv, fv) / 2.0

            # Single correction
            # -----------------
            # Correct for wrong inclusion of P(Faar)
            # This corrects the case P(Faab) but overcorrects P(Faaa)!
            p -= 2*project(fo, xo, xv, rv) / 4.0
            p -= 2*project(fo, xo, rv, xv) / 4.0 # If r == x this is the same as above -> overcorrection
            p -= 2*project(fo, ro, xv, xv) / 4.0 # overcorrection
            p -= 2*project(xo, xo, fv, rv) / 4.0
            p -= 2*project(xo, ro, fv, xv) / 4.0 # overcorrection
            p -= 2*project(ro, xo, fv, xv) / 4.0 # overcorrection

            # Correct overcorrection
            # The additional factor of 2 comes from how often the term was wrongly included above
            p += 2*2*project(fo, xo, xv, xv) / 4.0
            p += 2*2*project(xo, xo, fv, xv) / 4.0

    # Note that the energy should be invariant to symmetrization
    if symmetrize:
        p = (p + p.transpose(1,0,3,2)) / 2

    return p


def get_local_energy(self, cm, p1, p2, eris):
    """
    Parameters
    ----------
    cm : pyscf[.pbc].cc.CCSD or pyscf[.pbc].mp.MP2
        PySCF coupled cluster or MP2 object. This function accesses:
            cc.get_frozen_mask()
            cc.mo_occ
    p1 : ndarray
        Locally projected C1 amplitudes.
    p2 : ndarray
        Locally projected C2 amplitudes.
    eris :
        PySCF eris object as returned by cc.ao2mo()

    Returns
    -------
    e_loc : float
        Local energy contribution.
    """

    # MP2
    if p1 is None:
        e1 = 0
    # CC
    else:
        act = cc.get_frozen_mask()
        occ = cc.mo_occ[act] > 0
        vir = cc.mo_occ[act] == 0
        f = eris.fock[occ][:,vir]
        e1 = 2*np.sum(f * p1)

    if hasattr(eris, "ovvo"):
        eris_ovvo = eris.ovvo
    # MP2 only has eris.ovov - are these the same integrals?
    else:
        no, nv = p2.shape[1:3]
        eris_ovvo = eris.ovov[:].reshape(no,nv,no,nv).transpose(0, 1, 3, 2).conj()
    e2 = 2*einsum('ijab,iabj', p2, eris_ovvo)
    e2 -=  einsum('ijab,jabi', p2, eris_ovvo)

    self.log.info("Energy components: E1= % 16.8f Ha, E2=% 16.8f Ha", e1, e2)
    if e1 > 1e-4 and 10*e1 > e2:
        self.log.warning("WARNING: Large E1 component!")

    # Symmetry factor if fragment is repeated in molecule, (e.g. in hydrogen rings: only calculate one fragment)
    e_frag = self.sym_factor * (e1 + e2)

    return e_frag


#def get_local_energy_parts(self, cc, C1, C2):

#    a = cc.get_frozen_mask()
#    # Projector to local, occupied region
#    S = self.mf.get_ovlp()
#    C = cc.mo_coeff[:,a]
#    CTS = np.dot(C.T, S)

#    # Project one index of T amplitudes
#    l= self.indices
#    r = self.not_indices
#    o = cc.mo_occ[a] > 0
#    v = cc.mo_occ[a] == 0

#    eris = cc.ao2mo()

#    def get_projectors(aos):
#        Po = np.dot(CTS[o][:,aos], C[aos][:,o])
#        Pv = np.dot(CTS[v][:,aos], C[aos][:,v])
#        return Po, Pv

#    Lo, Lv = get_projectors(l)
#    Ro, Rv = get_projectors(r)

#    # Nomenclature:
#    # old occupied: i,j
#    # old virtual: a,b
#    # new occupied: p,q
#    # new virtual: s,t
#    T1_ll = einsum("pi,ia,sa->ps", Lo, C1, Lv)
#    T1_lr = einsum("pi,ia,sa->ps", Lo, C1, Rv)
#    T1_rl = einsum("pi,ia,sa->ps", Ro, C1, Lv)
#    T1 = T1_ll + (T1_lr + T1_rl)/2

#    F = eris.fock[o][:,v]
#    e1 = 2*np.sum(F * T1)
#    if not np.isclose(e1, 0):
#        self.log.warning("Warning: large E1 component: %.8e" % e1)

#    #tau = cc.t2 + einsum('ia,jb->ijab', cc.t1, cc.t1)
#    def project_T2(P1, P2, P3, P4):
#        T2p = einsum("pi,qj,ijab,sa,tb->pqst", P1, P2, C2, P3, P4)
#        return T2p


#    def epart(P1, P2, P3, P4):
#        T2_part = project_T2(P1, P2, P3, P4)
#        e_part = (2*einsum('ijab,iabj', T2_part, eris.ovvo)
#              - einsum('ijab,jabi', T2_part, eris.ovvo))
#        return e_part

#    energies = []
#    # 4
#    energies.append(epart(Lo, Lo, Lv, Lv))
#    # 3
#    energies.append(2*epart(Lo, Lo, Lv, Rv))
#    energies.append(2*epart(Lo, Ro, Lv, Lv))
#    assert np.isclose(epart(Lo, Lo, Rv, Lv), epart(Lo, Lo, Lv, Rv))
#    assert np.isclose(epart(Ro, Lo, Lv, Lv), epart(Lo, Ro, Lv, Lv))

#    energies.append(  epart(Lo, Lo, Rv, Rv))
#    energies.append(2*epart(Lo, Ro, Lv, Rv))
#    energies.append(2*epart(Lo, Ro, Rv, Lv))
#    energies.append(  epart(Ro, Ro, Lv, Lv))

#    energies.append(2*epart(Lo, Ro, Rv, Rv))
#    energies.append(2*epart(Ro, Ro, Lv, Rv))
#    assert np.isclose(epart(Ro, Lo, Rv, Rv), epart(Lo, Ro, Rv, Rv))
#    assert np.isclose(epart(Ro, Ro, Rv, Lv), epart(Ro, Ro, Lv, Rv))

#    energies.append(  epart(Ro, Ro, Rv, Rv))

#    #e4 = e_aaaa
#    #e3 = e_aaab + e_aaba + e_abaa + e_baaa
#    #e2 = 0.5*(e_aabb + e_abab + e_abba + e_bbaa)

#    with open("energy-parts.txt", "a") as f:
#        f.write((10*"  %16.8e" + "\n") % tuple(energies))

##def get_local_energy_most_indices_2C(self, cc, C1, C2, eris=None, symmetry_factor=None):
##
##    if symmetry_factor is None:
##        symmetry_factor = self.symmetry_factor
##
##    a = cc.get_frozen_mask()
##    # Projector to local, occupied region
##    S = self.mf.get_ovlp()
##    C = cc.mo_coeff[:,a]
##    CTS = np.dot(C.T, S)
##
##    # Project one index of T amplitudes
##    l= self.indices
##    r = self.not_indices
##    o = cc.mo_occ[a] > 0
##    v = cc.mo_occ[a] == 0
##
##    if eris is None:
##        self.log.warning("Warning: recomputing AO->MO integral transformation")
##        eris = cc.ao2mo()
##
##    def get_projectors(aos):
##        Po = np.dot(CTS[o][:,aos], C[aos][:,o])
##        Pv = np.dot(CTS[v][:,aos], C[aos][:,v])
##        return Po, Pv
##
##    Lo, Lv = get_projectors(l)
##    Ro, Rv = get_projectors(r)
##
##    # Nomenclature:
##    # old occupied: i,j
##    # old virtual: a,b
##    # new occupied: p,q
##    # new virtual: s,t
##    T1_ll = einsum("pi,ia,sa->ps", Lo, C1, Lv)
##    T1_lr = einsum("pi,ia,sa->ps", Lo, C1, Rv)
##    T1_rl = einsum("pi,ia,sa->ps", Ro, C1, Lv)
##    T1 = T1_ll + (T1_lr + T1_rl)/2
##
##    F = eris.fock[o][:,v]
##    e1 = 2*np.sum(F * T1)
##    if not np.isclose(e1, 0):
##        self.log.warning("Warning: large E1 component: %.8e" % e1)
##
##    #tau = cc.t2 + einsum('ia,jb->ijab', cc.t1, cc.t1)
##    def project_T2(P1, P2, P3, P4):
##        T2p = einsum("pi,qj,ijab,sa,tb->pqst", P1, P2, C2, P3, P4)
##        return T2p
##
##    f3 = 1.0
##    f2 = 0.5
##    # 4
##    T2 = 1*project_T2(Lo, Lo, Lv, Lv)
##    # 3
##    T2 += f3*(2*project_T2(Lo, Lo, Lv, Rv)      # factor 2 for LLRL
##            + 2*project_T2(Ro, Lo, Lv, Lv))     # factor 2 for RLLL
##    ## 2
##    #T2 += f2*(  project_T2(Lo, Lo, Rv, Rv)
##    #        + 2*project_T2(Lo, Ro, Lv, Rv)      # factor 2 for RLRL
##    #        + 2*project_T2(Lo, Ro, Rv, Lv)      # factor 2 for RLLR
##    #        +   project_T2(Ro, Ro, Lv, Lv))
##
##    # 2
##    T2 +=   project_T2(Lo, Lo, Rv, Rv)
##    T2 += 2*project_T2(Lo, Ro, Lv, Rv)      # factor 2 for RLRL
##    #T2 += 1*project_T2(Lo, Ro, Rv, Lv)      # factor 2 for RLLR
##    #T2 +=   project_T2(Ro, Ro, Lv, Lv)
##
##    e2 = (2*einsum('ijab,iabj', T2, eris.ovvo)
##           -einsum('ijab,jabi', T2, eris.ovvo))
##
##    e_loc = symmetry_factor * (e1 + e2)
##
##    return e_loc
##
##def get_local_energy_most_indices(self, cc, C1, C2, variant=1):
##
##    a = cc.get_frozen_mask()
##    # Projector to local, occupied region
##    S = self.mf.get_ovlp()
##    C = cc.mo_coeff[:,a]
##    CTS = np.dot(C.T, S)
##
##    # Project one index of T amplitudes
##    l= self.indices
##    r = self.not_indices
##    o = cc.mo_occ[a] > 0
##    v = cc.mo_occ[a] == 0
##
##    eris = cc.ao2mo()
##
##    def get_projectors(aos):
##        Po = np.dot(CTS[o][:,aos], C[aos][:,o])
##        Pv = np.dot(CTS[v][:,aos], C[aos][:,v])
##        return Po, Pv
##
##    Lo, Lv = get_projectors(l)
##    Ro, Rv = get_projectors(r)
##
##    # ONE-ELECTRON
##    # ============
##    pC1 = einsum("pi,ia,sa->ps", Lo, C1, Lv)
##    pC1 += 0.5*einsum("pi,ia,sa->ps", Lo, C1, Rv)
##    pC1 += 0.5*einsum("pi,ia,sa->ps", Ro, C1, Lv)
##
##    F = eris.fock[o][:,v]
##    e1 = 2*np.sum(F * pC1)
##    if not np.isclose(e1, 0):
##        self.log.warning("Warning: large E1 component: %.8e" % e1)
##
##    # TWO-ELECTRON
##    # ============
##
##    def project_C2_P1(P1):
##        pC2 = einsum("pi,ijab->pjab", P1, C2)
##        return pC2
##
##    def project_C2(P1, P2, P3, P4):
##        pC2 = einsum("pi,qj,ijab,sa,tb->pqst", P1, P2, C2, P3, P4)
##        return pC2
##
##    if variant == 1:
##
##        # QUADRUPLE L
##        # ===========
##        pC2 = project_C2(Lo, Lo, Lv, Lv)
##
##        # TRIPEL L
##        # ========
##        pC2 += 2*project_C2(Lo, Lo, Lv, Rv)
##        pC2 += 2*project_C2(Lo, Ro, Lv, Lv)
##
##        # DOUBLE L
##        # ========
##        # P(LLRR) [This wrongly includes: P(LLAA) - correction below]
##        pC2 +=   project_C2(Lo, Lo, Rv, Rv)
##        pC2 += 2*project_C2(Lo, Ro, Lv, Rv)
##        pC2 += 2*project_C2(Lo, Ro, Rv, Lv)
##        pC2 +=   project_C2(Ro, Ro, Lv, Lv)
##
##        # SINGLE L
##        # ========
##        # P(LRRR) [This wrongly includes: P(LAAR) - correction below]
##        four_idx_from_occ = False
##
##        if not four_idx_from_occ:
##            pC2 += 0.25*2*project_C2(Lo, Ro, Rv, Rv)
##            pC2 += 0.25*2*project_C2(Ro, Ro, Lv, Rv)
##        else:
##            pC2 += 0.5*2*project_C2(Lo, Ro, Rv, Rv)
##
##        # CORRECTIONS
##        # ===========
##        for x in self.loop_clusters(exclude_self=True):
##            Xo, Xv = get_projectors(x.indices)
##
##            # DOUBLE CORRECTION
##            # -----------------
##            # Correct for wrong inclusion of P(LLAA)
##            # The case P(LLAA) was included with prefactor of 1 instead of 1/2
##            # We thus need to only correct by "-1/2"
##            pC2 -= 0.5*  project_C2(Lo, Lo, Xv, Xv)
##            pC2 -= 0.5*2*project_C2(Lo, Xo, Lv, Xv)
##            pC2 -= 0.5*2*project_C2(Lo, Xo, Xv, Lv)
##            pC2 -= 0.5*  project_C2(Xo, Xo, Lv, Lv)
##
##            # SINGLE CORRECTION
##            # -----------------
##            # Correct for wrong inclusion of P(LAAR)
##            # This corrects the case P(LAAB) but overcorrects P(LAAA)!
##            if not four_idx_from_occ:
##                pC2 -= 0.25*2*project_C2(Lo, Xo, Xv, Rv)
##                pC2 -= 0.25*2*project_C2(Lo, Xo, Rv, Xv) # If R == X this is the same as above -> overcorrection
##                pC2 -= 0.25*2*project_C2(Lo, Ro, Xv, Xv) # overcorrection
##                pC2 -= 0.25*2*project_C2(Xo, Xo, Lv, Rv)
##                pC2 -= 0.25*2*project_C2(Xo, Ro, Lv, Xv) # overcorrection
##                pC2 -= 0.25*2*project_C2(Ro, Xo, Lv, Xv) # overcorrection
##
##                # Correct overcorrection
##                pC2 += 0.25*2*2*project_C2(Lo, Xo, Xv, Xv)
##                pC2 += 0.25*2*2*project_C2(Xo, Xo, Lv, Xv)
##
##            else:
##                pC2 -= 0.5*2*project_C2(Lo, Xo, Xv, Rv)
##                pC2 -= 0.5*2*project_C2(Lo, Xo, Rv, Xv) # If R == X this is the same as above -> overcorrection
##                pC2 -= 0.5*2*project_C2(Lo, Ro, Xv, Xv) # overcorrection
##
##                # Correct overcorrection
##                pC2 += 0.5*2*2*project_C2(Lo, Xo, Xv, Xv)
##
##        e2 = (2*einsum('ijab,iabj', pC2, eris.ovvo)
##               -einsum('ijab,jabi', pC2, eris.ovvo))
##
##    elif variant == 2:
##        # QUADRUPLE L
##        # ===========
##        pC2 = project_C2(Lo, Lo, Lv, Lv)
##
##        # TRIPEL L
##        # ========
##        pC2 += 2*project_C2(Lo, Lo, Lv, Rv)
##        pC2 += 2*project_C2(Lo, Ro, Lv, Lv)
##
##        # DOUBLE L
##        # ========
##        pC2 +=   project_C2(Lo, Lo, Rv, Rv)
##        pC2 +=   2*project_C2(Lo, Ro, Lv, Rv)
##        pC2 +=   2*project_C2(Lo, Ro, Rv, Lv)
##        for x in self.loop_clusters(exclude_self=True):
##            Xo, Xv = get_projectors(x.indices)
##            pC2 -= project_C2(Lo, Xo, Lv, Xv)
##            pC2 -= project_C2(Lo, Xo, Xv, Lv)
##
##        # SINGLE L
##        # ========
##
##        # This wrongly includes LXXX
##        pC2 += 0.5*2*project_C2(Lo, Ro, Rv, Rv)
##        for x in self.loop_clusters(exclude_self=True):
##            Xo, Xv = get_projectors(x.indices)
##
##            pC2 -= 0.5*2*project_C2(Lo, Xo, Rv, Xv)
##            pC2 -= 0.5*2*project_C2(Lo, Xo, Xv, Rv)
##
##            pC2 += 0.5*2*project_C2(Lo, Xo, Xv, Xv)
##
##        e2 = (2*einsum('ijab,iabj', pC2, eris.ovvo)
##               -einsum('ijab,jabi', pC2, eris.ovvo))
##
##    elif variant == 3:
##        # QUADRUPLE + TRIPLE L
##        # ====================
##        pC2 = project_C2_P1(Lo)
##        pC2 += project_C2(Ro, Lo, Lv, Lv)
##        for x in self.loop_clusters(exclude_self=True):
##            Xo, Xv = get_projectors(x.indices)
##            pC2 -= project_C2(Lo, Xo, Xv, Xv)
##
##        e2 = (2*einsum('ijab,iabj', pC2, eris.ovvo)
##               -einsum('ijab,jabi', pC2, eris.ovvo))
##
##    e_loc = e1 + e2
##
##    return e_loc
