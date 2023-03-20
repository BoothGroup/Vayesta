import numpy as np
from vayesta.core.util import einsum

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
        
        # Note that this requires (vv|ov) [first term], (oo|ov) and (ov|oo) [second term]
        t3v = np.zeros_like(dt2)
        # First term: 1/2 P_ab [t_ijmaef v_efbm]
        t3v += einsum('bemf, jimeaf -> ijab', gvvov - gvvov.transpose(0,3,2,1), t3) / 2
        t3v += einsum('bemf, ijmaef -> ijab', gvvov, t3)
        # Second term: -1/2 P_ij [t_imnabe v_jemn]
        t3v -= einsum('mjne, minbae -> ijab', gooov - govoo.transpose(0,3,2,1), t3) / 2
        t3v -= einsum('mjne, imnabe -> ijab', gooov, t3)
        # Permutation
        t3v += t3v.transpose(1,0,3,2)
        dt2 += t3v

    return dt1, dt2

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
    dt2_aaaa += einsum("kcld,ijklabcd->ijab", v_aaaa_ovov, t4_aaaaaaaa) * 0.5
    dt2_aaaa += einsum("menf,imjnaebf->ijab", v_bbbb_ovov, t4_abababab) * 0.5
    dt2_aaaa += einsum("lcne,ijlnabce->ijab", v_aabb_ovov, t4_aaabaaab)
    x0 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x0 += einsum("jlkc,iklacb->ijab", v_aabb_ooov, t3_abaaba)
    x1 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x1 += einsum("jklc,iklabc->ijab", v_aaaa_ooov, t3_aaaaaa)
    x2 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=np.float64)
    x2 += einsum("jb,kbia->iajk", t1_aa, v_aabb_ovov)
    x3 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x3 += einsum("kcil,jklacb->ijab", x2, t3_abaaba)
    x4 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x4 += einsum("ib,jakb->ijka", t1_aa, v_aaaa_ovov)
    x5 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x5 += einsum("iklc,jklabc->ijab", x4, t3_aaaaaa) * -1.0
    x6 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x6 += einsum("ijba->ijab", x0) * -1.0
    x6 += einsum("ijba->ijab", x1) * -1.0
    x6 += einsum("ijba->ijab", x3)
    x6 += einsum("ijba->ijab", x5)
    dt2_aaaa += einsum("ijab->ijab", x6) * -1.0
    dt2_aaaa += einsum("jiab->ijab", x6)
    x7 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x7 += einsum("bdkc,ikjacd->ijab", v_aabb_vvov, t3_abaaba)
    x8 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x8 += einsum("kdbc,ijkacd->ijab", v_aaaa_ovvv, t3_aaaaaa) * -1.0
    x9 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x9 += einsum("kclb,iljabc->ijka", v_aabb_ovov, t3_abaaba)
    x10 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x10 += einsum("kclb,ijlabc->ijka", v_aaaa_ovov, t3_aaaaaa) * -1.0
    x11 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x11 += einsum("ijka->ijka", x9)
    x11 += einsum("ijka->ijka", x10)
    x12 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x12 += einsum("ka,ijkb->ijab", t1_aa, x11)
    x13 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x13 += einsum("ijab->ijab", x7)
    x13 += einsum("ijab->ijab", x8) * -1.0
    x13 += einsum("ijab->ijab", x12)
    dt2_aaaa += einsum("ijab->ijab", x13)
    dt2_aaaa += einsum("ijba->ijab", x13) * -1.0
    x14 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x14 += einsum("ibja->ijab", v_aaaa_ovov) * -1.0
    x14 += einsum("iajb->ijab", v_aaaa_ovov)
    x15 = np.zeros((nocc[0], nvir[0]), dtype=np.float64)
    x15 += einsum("ia->ia", f_aa_ov)
    x15 += einsum("jb,iajb->ia", t1_bb, v_aabb_ovov)
    x15 += einsum("kc,kiac->ia", t1_aa, x14) * -1.0
    dt2_aaaa += einsum("lc,ijlabc->ijab", x15, t3_aaaaaa)
    x16 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x16 += einsum("ibja->ijab", v_bbbb_ovov)
    x16 += einsum("iajb->ijab", v_bbbb_ovov) * -1.0
    x17 = np.zeros((nocc[1], nvir[1]), dtype=np.float64)
    x17 += einsum("ia->ia", f_bb_ov)
    x17 += einsum("jb,jbia->ia", t1_aa, v_aabb_ovov)
    x17 += einsum("kc,kica->ia", t1_bb, x16) * -1.0
    dt2_aaaa += einsum("ne,injaeb->ijab", x17, t3_abaaba)
    dt2_bbbb = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    dt2_bbbb += einsum("ldkc,lijkdabc->ijab", v_aabb_ovov, t4_abbbabbb)
    dt2_bbbb += einsum("ldme,limjdaeb->ijab", v_aaaa_ovov, t4_abababab) * 0.5
    dt2_bbbb += einsum("kcnf,ijknabcf->ijab", v_bbbb_ovov, t4_bbbbbbbb) * 0.5
    x0 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x0 += einsum("jklc,iklabc->ijab", v_bbbb_ooov, t3_bbbbbb)
    x1 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1 += einsum("lcjk,ilkacb->ijab", v_aabb_ovoo, t3_babbab)
    x2 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x2 += einsum("ib,jakb->ijka", t1_bb, v_bbbb_ovov)
    x3 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x3 += einsum("ilkc,jklabc->ijab", x2, t3_bbbbbb)
    x4 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=np.float64)
    x4 += einsum("ib,kajb->ijka", t1_bb, v_aabb_ovov)
    x5 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x5 += einsum("iklc,jlkacb->ijab", x4, t3_babbab)
    x6 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x6 += einsum("ijba->ijab", x0) * -1.0
    x6 += einsum("ijba->ijab", x1) * -1.0
    x6 += einsum("ijba->ijab", x3)
    x6 += einsum("ijba->ijab", x5)
    dt2_bbbb += einsum("ijab->ijab", x6) * -1.0
    dt2_bbbb += einsum("jiab->ijab", x6)
    x7 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x7 += einsum("kcbd,ijkacd->ijab", v_bbbb_ovvv, t3_bbbbbb)
    x8 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x8 += einsum("kdbc,ikjadc->ijab", v_aabb_ovvv, t3_babbab)
    x9 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x9 += einsum("kclb,ijlabc->ijka", v_bbbb_ovov, t3_bbbbbb) * -1.0
    x10 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x10 += einsum("lckb,iljacb->ijka", v_aabb_ovov, t3_babbab)
    x11 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x11 += einsum("ijka->ijka", x9)
    x11 += einsum("ijka->ijka", x10)
    x12 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x12 += einsum("ka,ijkb->ijab", t1_bb, x11)
    x13 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x13 += einsum("ijab->ijab", x7) * -1.0
    x13 += einsum("ijab->ijab", x8)
    x13 += einsum("ijab->ijab", x12)
    dt2_bbbb += einsum("ijab->ijab", x13)
    dt2_bbbb += einsum("ijba->ijab", x13) * -1.0
    x14 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x14 += einsum("ibja->ijab", v_bbbb_ovov)
    x14 += einsum("iajb->ijab", v_bbbb_ovov) * -1.0
    x15 = np.zeros((nocc[1], nvir[1]), dtype=np.float64)
    x15 += einsum("ia->ia", f_bb_ov)
    x15 += einsum("jb,jbia->ia", t1_aa, v_aabb_ovov)
    x15 += einsum("kc,kica->ia", t1_bb, x14) * -1.0
    dt2_bbbb += einsum("kc,ijkabc->ijab", x15, t3_bbbbbb)
    x16 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x16 += einsum("ibja->ijab", v_aaaa_ovov) * -1.0
    x16 += einsum("iajb->ijab", v_aaaa_ovov)
    x17 = np.zeros((nocc[0], nvir[0]), dtype=np.float64)
    x17 += einsum("ia->ia", f_aa_ov)
    x17 += einsum("jb,iajb->ia", t1_bb, v_aabb_ovov)
    x17 += einsum("kc,kiac->ia", t1_aa, x16) * -1.0
    dt2_bbbb += einsum("ld,iljadb->ijab", x17, t3_babbab)
    dt2_abab = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    dt2_abab += einsum("kcld,ikljacdb->ijab", v_aaaa_ovov, t4_aaabaaab) * 0.5
    dt2_abab += einsum("kcbe,ijkaec->ijab", v_aabb_ovvv, t3_abaaba)
    dt2_abab += einsum("mebf,jimeaf->ijab", v_bbbb_ovvv, t3_babbab) * -1.0
    dt2_abab += einsum("kcad,ijkcbd->ijab", v_aaaa_ovvv, t3_abaaba) * -1.0
    dt2_abab += einsum("acme,jimbce->ijab", v_aabb_vvov, t3_babbab)
    dt2_abab += einsum("mfne,ijmnabef->ijab", v_bbbb_ovov, t4_abbbabbb) * -0.5
    dt2_abab += einsum("kcme,ijkmabce->ijab", v_aabb_ovov, t4_abababab)
    x0 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=np.float64)
    x0 += einsum("jkia->iajk", v_aabb_ooov)
    x0 += einsum("kb,jbia->iajk", t1_aa, v_aabb_ovov)
    dt2_abab += einsum("meki,jkmbae->ijab", x0, t3_babbab) * -1.0
    x1 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x1 += einsum("ikja->ijka", v_aaaa_ooov)
    x1 += einsum("ib,jakb->ijka", t1_aa, v_aaaa_ovov)
    dt2_abab += einsum("iklc,kjlabc->ijab", x1, t3_abaaba)
    x2 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x2 += einsum("ikja->ijka", v_bbbb_ooov)
    x2 += einsum("ib,jakb->ijka", t1_bb, v_bbbb_ovov)
    dt2_abab += einsum("jnme,minbae->ijab", x2, t3_babbab) * -1.0
    x3 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=np.float64)
    x3 += einsum("kaij->ijka", v_aabb_ovoo)
    x3 += einsum("jb,kaib->ijka", t1_bb, v_aabb_ovov)
    dt2_abab += einsum("mjkc,imkabc->ijab", x3, t3_abaaba) * -1.0
    x4 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x4 += einsum("ibja->ijab", v_aaaa_ovov) * -1.0
    x4 += einsum("iajb->ijab", v_aaaa_ovov)
    x5 = np.zeros((nocc[0], nvir[0]), dtype=np.float64)
    x5 += einsum("ia->ia", f_aa_ov)
    x5 += einsum("jb,iajb->ia", t1_bb, v_aabb_ovov)
    x5 += einsum("kc,kiac->ia", t1_aa, x4) * -1.0
    dt2_abab += einsum("kc,ijkabc->ijab", x5, t3_abaaba)
    x6 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x6 += einsum("ibja->ijab", v_bbbb_ovov)
    x6 += einsum("iajb->ijab", v_bbbb_ovov) * -1.0
    x7 = np.zeros((nocc[1], nvir[1]), dtype=np.float64)
    x7 += einsum("ia->ia", f_bb_ov)
    x7 += einsum("jb,jbia->ia", t1_aa, v_aabb_ovov)
    x7 += einsum("kc,kica->ia", t1_bb, x6) * -1.0
    dt2_abab += einsum("me,jimbae->ijab", x7, t3_babbab)
    x8 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=np.float64)
    x8 += einsum("kclb,ijlacb->iajk", v_aabb_ovov, t3_babbab)
    x8 += einsum("kcmd,jimcad->iajk", v_aaaa_ovov, t3_abaaba)
    dt2_abab += einsum("ka,jbik->ijab", t1_aa, x8) * -1.0
    x9 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=np.float64)
    x9 += einsum("jclb,iklbac->ijka", v_bbbb_ovov, t3_babbab) * -1.0
    x9 += einsum("mdjc,kimacd->ijka", v_aabb_ovov, t3_abaaba)
    dt2_abab += einsum("mb,jmia->ijab", t1_bb, x9) * -1.0

    dt1 = (dt1_aa, dt1_bb)
    dt2 = (dt2_aaaa, dt2_abab, dt2_bbbb)

    return dt1, dt2
