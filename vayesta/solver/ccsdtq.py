import numpy as np
from vayesta.core.util import einsum


def t2_residual_rhf_t3v(solver, fragment, t3, v):
    govov, gvvov, gooov, govoo = v
    nocc, nvir = govov.shape[:2]
    dt2 = np.zeros((nocc, nocc, nvir, nvir))

    # First term: 1/2 P_ab [t_ijmaef v_efbm]
    dt2 += einsum('bemf, jimeaf -> ijab', gvvov - gvvov.transpose(0,3,2,1), t3) / 2
    dt2 += einsum('bemf, ijmaef -> ijab', gvvov, t3)
    # Second term: -1/2 P_ij [t_imnabe v_jemn]
    dt2 -= einsum('mjne, minbae -> ijab', gooov - govoo.transpose(0,3,2,1), t3) / 2
    dt2 -= einsum('mjne, imnabe -> ijab', gooov, t3)
    # Permutation
    dt2 += dt2.transpose(1,0,3,2)

    return dt2


def t_residual_rhf(solver, fragment, t1, t2, t3, t4, f, v, include_t3v=False):
    t4_abaa, t4_abab = t4
    fov = f
    govov, gvvov, gooov, govoo = v
    nocc, nvir = t1.shape

    dt1 = np.zeros_like(t1)
    dt2 = np.zeros_like(t2)

    # Construct physical antisymmetrized integrals for some contractions
    # Note that some contractions are with physical and some chemical integrals (govov)
    antiphys_g = (govov - govov.transpose(0,3,2,1)).transpose(0,2,1,3)
    spinned_antiphys_g = (2.0*govov - govov.transpose(0,3,2,1)).transpose(0,2,1,3)

    # --- T1 update
    # --- T3 * V
    dt1 -= einsum('ijab, jiupab -> up', spinned_antiphys_g, t3)

    # --- T2 update
    # --- T3 * F
    if np.allclose(fov, np.zeros_like(fov)):
        solver.log.info("fov block zero: No T3 * f contribution.")
    # (Fa) (Taba) contraction
    dt2 += einsum('me, ijmabe -> ijab', fov, t3)
    # (Fb) (Tabb) contraction
    dt2 += einsum('me, jimbae -> ijab', fov, t3)
    solver.log.info("(T3 * F) -> T2 update norm from fragment {}: {}".format(fragment.id, np.linalg.norm(dt2)))

    # --- T4 * V
    # (Vaa) (Tabaa) contraction
    t4v = einsum('mnef, ijmnabef -> ijab', antiphys_g, t4_abaa) / 4
    t4v += t4v.transpose(1,0,3,2)
    # (Vab) (Tabab) contraction
    t4v += einsum('menf, ijmnabef -> ijab', govov, t4_abab)
    dt2 += t4v

    # --- (T1 T3) * V
    # Note: Approximate T1 by the CCSDTQ T1 amplitudes of this fragment.
    # TODO: Relax this approximation via the callback?
    t1t3v = np.zeros_like(dt2)
    X_ = einsum('mnef, me -> nf', spinned_antiphys_g, t1)
    t1t3v += einsum('nf, nijfab -> ijab', X_, t3)

    X_ =  einsum('mnef, njiebf -> ijmb', antiphys_g, t3) / 2
    X_ += einsum('menf, jinfeb -> ijmb', govov, t3)
    t1t3v += einsum('ijmb, ma -> ijab', X_, t1)

    X_ = einsum('mnef, mjnfba -> ejab', antiphys_g, t3) / 2
    X_ += einsum('menf, nmjbaf -> ejab', govov, t3)
    t1t3v += einsum('ejab, ie -> ijab', X_, t1)
    # apply permutation
    t1t3v += t1t3v.transpose(1,0,3,2)
    dt2 += t1t3v
    solver.log.info("T1 norm in ext corr from fragment {}: {}".format(fragment.id, np.linalg.norm(t1)))

    # --- T3 * V 
    if include_t3v:
        # Option to leave out this term, and instead perform T3 * V with the
        # integrals in the parent cluster later.
        # This will give a different result since the V operators
        # will span a different space. Instead, here we just contract T3 with integrals 
        # in cluster y (FCI), rather than cluster x (CCSD)
        dt2 += t2_residual_rhf_t3v(solver, fragment, t3, v)

    return dt1, dt2


def t2_residual_uhf_t3v(solver, fragment, t3, v):
    t3_aaaaaa, t3_abaaba, t3_babbab, t3_bbbbbb = t3
    v_ooov, v_ovov, v_ovvv, v_vvov, v_ovoo = v
    v_aaaa_ooov, v_aabb_ooov, v_bbbb_ooov = v_ooov
    v_aaaa_ovov, v_aabb_ovov, v_bbbb_ovov = v_ovov
    v_aaaa_ovvv, v_aabb_ovvv, v_bbbb_ovvv = v_ovvv
    v_aaaa_vvov, v_aabb_vvov, v_bbbb_vvov = v_vvov
    v_aaaa_ovoo, v_aabb_ovoo, v_bbbb_ovoo = v_ovoo
    nocc = (t3_abaaba.shape[0], t3_abaaba.shape[1])
    nvir = (t3_abaaba.shape[3], t3_abaaba.shape[4])

    x0 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x0 += einsum("jlkc,iklacb->ijab", v_aabb_ooov, t3_abaaba)
    x1 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x1 += einsum("jlkc,iklabc->ijab", v_aaaa_ooov, t3_aaaaaa) * -1.0
    x2 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x2 += einsum("ijba->ijab", x0) * -1.0
    x2 += einsum("ijba->ijab", x1) * -1.0
    dt2_aaaa = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    dt2_aaaa += einsum("ijab->ijab", x2) * -1.0
    dt2_aaaa += einsum("jiab->ijab", x2)
    x3 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x3 += einsum("bdkc,ikjacd->ijab", v_aabb_vvov, t3_abaaba)
    x4 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x4 += einsum("kcbd,ijkacd->ijab", v_aaaa_ovvv, t3_aaaaaa)
    x5 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x5 += einsum("ijab->ijab", x3)
    x5 += einsum("ijab->ijab", x4) * -1.0
    dt2_aaaa += einsum("ijab->ijab", x5)
    dt2_aaaa += einsum("ijba->ijab", x5) * -1.0
    x0 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x0 += einsum("jklc,iklabc->ijab", v_bbbb_ooov, t3_bbbbbb)
    x1 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1 += einsum("lcjk,ilkacb->ijab", v_aabb_ovoo, t3_babbab)
    x2 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x2 += einsum("ijba->ijab", x0) * -1.0
    x2 += einsum("ijba->ijab", x1) * -1.0
    dt2_bbbb = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    dt2_bbbb += einsum("ijab->ijab", x2) * -1.0
    dt2_bbbb += einsum("jiab->ijab", x2)
    x3 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x3 += einsum("kdbc,ijkacd->ijab", v_bbbb_ovvv, t3_bbbbbb) * -1.0
    x4 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x4 += einsum("kdbc,ikjadc->ijab", v_aabb_ovvv, t3_babbab)
    x5 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x5 += einsum("ijab->ijab", x3)
    x5 += einsum("ijab->ijab", x4) * -1.0
    dt2_bbbb += einsum("ijab->ijab", x5) * -1.0
    dt2_bbbb += einsum("ijba->ijab", x5)
    dt2_abab = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    dt2_abab += einsum("ilkc,jlkbac->ijab", v_aabb_ooov, t3_babbab) * -1.0
    dt2_abab += einsum("ldjk,iklabd->ijab", v_aabb_ovoo, t3_abaaba) * -1.0
    dt2_abab += einsum("ilmd,ljmabd->ijab", v_aaaa_ooov, t3_abaaba) * -1.0
    dt2_abab += einsum("ldbc,ijlacd->ijab", v_aabb_ovvv, t3_abaaba)
    dt2_abab += einsum("adkc,jikbdc->ijab", v_aabb_vvov, t3_babbab)
    dt2_abab += einsum("jnkc,kinbac->ijab", v_bbbb_ooov, t3_babbab)
    dt2_abab += einsum("ldae,ijldbe->ijab", v_aaaa_ovvv, t3_abaaba) * -1.0
    dt2_abab += einsum("kcbf,jikcaf->ijab", v_bbbb_ovvv, t3_babbab) * -1.0

    dt2 = (dt2_aaaa, dt2_abab, dt2_bbbb)

    return dt2


def t_residual_uhf(solver, fragment, t1, t2, t3, t4, f, v, include_t3v=False):
    t1_aa, t1_bb = t1
    t2_aaaa, t2_abab, t2_bbbb = t2
    t3_aaaaaa, t3_abaaba, t3_babbab, t3_bbbbbb = t3
    t4_aaaaaaaa, t4_aaabaaab, t4_abababab, t4_abbbabbb, t4_bbbbbbbb = t4
    f_aa_ov, f_bb_ov = f
    v_ooov, v_ovov, v_ovvv, v_vvov, v_ovoo = v
    v_aaaa_ooov, v_aabb_ooov, v_bbbb_ooov = v_ooov
    v_aaaa_ovov, v_aabb_ovov, v_bbbb_ovov = v_ovov
    v_aaaa_ovvv, v_aabb_ovvv, v_bbbb_ovvv = v_ovvv
    v_aaaa_vvov, v_aabb_vvov, v_bbbb_vvov = v_vvov
    v_aaaa_ovoo, v_aabb_ovoo, v_bbbb_ovoo = v_ovoo
    nocc = (t1_aa.shape[0], t1_bb.shape[0])
    nvir = (t1_aa.shape[1], t1_bb.shape[1])

    dt1_aa = np.zeros((nocc[0], nvir[0]), dtype=np.float64)
    dt1_aa += einsum("jbkc,ijkabc->ia", v_aaaa_ovov, t3_aaaaaa) * 0.5
    dt1_aa += einsum("ldme,limdae->ia", v_bbbb_ovov, t3_babbab) * 0.5
    dt1_aa += einsum("jbmd,imjadb->ia", v_aabb_ovov, t3_abaaba)
    dt1_bb = np.zeros((nocc[1], nvir[1]), dtype=np.float64)
    dt1_bb += einsum("jbkc,jikbac->ia", v_aaaa_ovov, t3_abaaba) * 0.5
    dt1_bb += einsum("ldme,ilmade->ia", v_bbbb_ovov, t3_bbbbbb) * 0.5
    dt1_bb += einsum("kcmd,ikmacd->ia", v_aabb_ovov, t3_babbab)

    dt2_aaaa = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    dt2_aaaa += einsum("ldkc,ijlkabdc->ijab", v_aabb_ovov, t4_aaabaaab)
    dt2_aaaa += einsum("kcme,ikjmacbe->ijab", v_bbbb_ovov, t4_abababab) * 0.5
    dt2_aaaa += einsum("lfnd,ijlnabdf->ijab", v_aaaa_ovov, t4_aaaaaaaa) * -0.5
    x0 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=np.float64)
    x0 += einsum("jb,kbia->iajk", t1_aa, v_aabb_ovov)
    x1 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x1 += einsum("kcil,jklacb->ijab", x0, t3_abaaba)
    x2 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x2 += einsum("ib,jakb->ijka", t1_aa, v_aaaa_ovov)
    x3 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x3 += einsum("iklc,jklabc->ijab", x2, t3_aaaaaa) * -1.0
    x4 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x4 += einsum("ijba->ijab", x1) * -1.0
    x4 += einsum("ijba->ijab", x3) * -1.0
    dt2_aaaa += einsum("ijab->ijab", x4)
    dt2_aaaa += einsum("jiab->ijab", x4) * -1.0
    x5 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x5 += einsum("kclb,iljabc->ijka", v_aabb_ovov, t3_abaaba)
    x6 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x6 += einsum("kclb,ijlabc->ijka", v_aaaa_ovov, t3_aaaaaa) * -1.0
    x7 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x7 += einsum("ijka->ijka", x5)
    x7 += einsum("ijka->ijka", x6)
    x8 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x8 += einsum("ka,ijkb->ijab", t1_aa, x7)
    dt2_aaaa += einsum("ijab->ijab", x8)
    dt2_aaaa += einsum("ijba->ijab", x8) * -1.0
    x9 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x9 += einsum("ibja->ijab", v_aaaa_ovov)
    x9 += einsum("iajb->ijab", v_aaaa_ovov) * -1.0
    x10 = np.zeros((nocc[0], nvir[0]), dtype=np.float64)
    x10 += einsum("ia->ia", f_aa_ov)
    x10 += einsum("jb,iajb->ia", t1_bb, v_aabb_ovov)
    x10 += einsum("kc,kica->ia", t1_aa, x9) * -1.0
    dt2_aaaa += einsum("ld,ijlabd->ijab", x10, t3_aaaaaa)
    x11 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x11 += einsum("ibja->ijab", v_bbbb_ovov)
    x11 += einsum("iajb->ijab", v_bbbb_ovov) * -1.0
    x12 = np.zeros((nocc[1], nvir[1]), dtype=np.float64)
    x12 += einsum("ia->ia", f_bb_ov)
    x12 += einsum("jb,jbia->ia", t1_aa, v_aabb_ovov)
    x12 += einsum("kc,kica->ia", t1_bb, x11) * -1.0
    dt2_aaaa += einsum("kc,ikjacb->ijab", x12, t3_abaaba)
    dt2_bbbb = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    dt2_bbbb += einsum("kcld,ijklabcd->ijab", v_bbbb_ovov, t4_bbbbbbbb) * 0.5
    dt2_bbbb += einsum("melc,mijleabc->ijab", v_aabb_ovov, t4_abbbabbb)
    dt2_bbbb += einsum("menf,minjeafb->ijab", v_aaaa_ovov, t4_abababab) * 0.5
    x0 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x0 += einsum("ib,jakb->ijka", t1_bb, v_bbbb_ovov)
    x1 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1 += einsum("iklc,jklabc->ijab", x0, t3_bbbbbb) * -1.0
    x2 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=np.float64)
    x2 += einsum("ib,kajb->ijka", t1_bb, v_aabb_ovov)
    x3 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x3 += einsum("iklc,jlkacb->ijab", x2, t3_babbab)
    x4 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x4 += einsum("ijba->ijab", x1) * -1.0
    x4 += einsum("ijba->ijab", x3) * -1.0
    dt2_bbbb += einsum("ijab->ijab", x4)
    dt2_bbbb += einsum("jiab->ijab", x4) * -1.0
    x5 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x5 += einsum("kclb,ijlabc->ijka", v_bbbb_ovov, t3_bbbbbb) * -1.0
    x6 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x6 += einsum("lckb,iljacb->ijka", v_aabb_ovov, t3_babbab)
    x7 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x7 += einsum("ijka->ijka", x5)
    x7 += einsum("ijka->ijka", x6)
    x8 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x8 += einsum("ka,ijkb->ijab", t1_bb, x7)
    dt2_bbbb += einsum("ijab->ijab", x8)
    dt2_bbbb += einsum("ijba->ijab", x8) * -1.0
    x9 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x9 += einsum("ibja->ijab", v_bbbb_ovov)
    x9 += einsum("iajb->ijab", v_bbbb_ovov) * -1.0
    x10 = np.zeros((nocc[1], nvir[1]), dtype=np.float64)
    x10 += einsum("ia->ia", f_bb_ov)
    x10 += einsum("jb,jbia->ia", t1_aa, v_aabb_ovov)
    x10 += einsum("kc,kica->ia", t1_bb, x9) * -1.0
    dt2_bbbb += einsum("lc,ijlabc->ijab", x10, t3_bbbbbb)
    x11 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x11 += einsum("ibja->ijab", v_aaaa_ovov)
    x11 += einsum("iajb->ijab", v_aaaa_ovov) * -1.0
    x12 = np.zeros((nocc[0], nvir[0]), dtype=np.float64)
    x12 += einsum("ia->ia", f_aa_ov)
    x12 += einsum("jb,iajb->ia", t1_bb, v_aabb_ovov)
    x12 += einsum("kc,kica->ia", t1_aa, x11) * -1.0
    dt2_bbbb += einsum("me,imjaeb->ijab", x12, t3_babbab)
    dt2_abab = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    dt2_abab += einsum("kcld,ikljacdb->ijab", v_aaaa_ovov, t4_aaabaaab) * 0.5
    dt2_abab += einsum("ldme,ijlmabde->ijab", v_aabb_ovov, t4_abababab)
    dt2_abab += einsum("mfne,ijmnabef->ijab", v_bbbb_ovov, t4_abbbabbb) * -0.5
    x0 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x0 += einsum("ib,jakb->ijka", t1_aa, v_aaaa_ovov)
    dt2_abab += einsum("ikld,kjlabd->ijab", x0, t3_abaaba)
    x1 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=np.float64)
    x1 += einsum("ib,kajb->ijka", t1_bb, v_aabb_ovov)
    dt2_abab += einsum("jmld,imlabd->ijab", x1, t3_abaaba) * -1.0
    x2 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=np.float64)
    x2 += einsum("jb,kbia->iajk", t1_aa, v_aabb_ovov)
    dt2_abab += einsum("meil,jlmbae->ijab", x2, t3_babbab) * -1.0
    x3 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x3 += einsum("ib,jakb->ijka", t1_bb, v_bbbb_ovov)
    dt2_abab += einsum("jmne,minbae->ijab", x3, t3_babbab)
    x4 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x4 += einsum("ibja->ijab", v_bbbb_ovov)
    x4 += einsum("iajb->ijab", v_bbbb_ovov) * -1.0
    x5 = np.zeros((nocc[1], nvir[1]), dtype=np.float64)
    x5 += einsum("ia->ia", f_bb_ov)
    x5 += einsum("jb,jbia->ia", t1_aa, v_aabb_ovov)
    x5 += einsum("kc,kica->ia", t1_bb, x4) * -1.0
    dt2_abab += einsum("me,jimbae->ijab", x5, t3_babbab)
    x6 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x6 += einsum("ibja->ijab", v_aaaa_ovov)
    x6 += einsum("iajb->ijab", v_aaaa_ovov) * -1.0
    x7 = np.zeros((nocc[0], nvir[0]), dtype=np.float64)
    x7 += einsum("ia->ia", f_aa_ov)
    x7 += einsum("jb,iajb->ia", t1_bb, v_aabb_ovov)
    x7 += einsum("kc,kica->ia", t1_aa, x6) * -1.0
    dt2_abab += einsum("ld,ijlabd->ijab", x7, t3_abaaba)
    x8 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=np.float64)
    x8 += einsum("kclb,ijlacb->iajk", v_aabb_ovov, t3_babbab)
    x8 += einsum("kcmd,jimcad->iajk", v_aaaa_ovov, t3_abaaba)
    dt2_abab += einsum("la,jbil->ijab", t1_aa, x8) * -1.0
    x9 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=np.float64)
    x9 += einsum("jclb,iklbac->ijka", v_bbbb_ovov, t3_babbab) * -1.0
    x9 += einsum("mdjc,kimacd->ijka", v_aabb_ovov, t3_abaaba)
    dt2_abab += einsum("mb,jmia->ijab", t1_bb, x9) * -1.0

    if include_t3v:
        dt2_t3v = t2_residual_uhf_t3v(solver, fragment, t3, v)
        dt2_aaaa += dt2_t3v[0]
        dt2_abab += dt2_t3v[1]
        dt2_bbbb += dt2_t3v[2]

    dt1 = (dt1_aa, dt1_bb)
    dt2 = (dt2_aaaa, dt2_abab, dt2_bbbb)

    return dt1, dt2
