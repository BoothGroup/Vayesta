import numpy as np
from vayesta.core.util import einsum

def t1_residual_uhf(t1, t2, t3, t4, f, v):
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
    t1new_aa = np.zeros((nocc[0], nvir[0]), dtype=np.float64)
    t1new_aa += einsum("jbkc,ijkabc->ia", v_aaaa_ovov, t3_aaaaaa) * 0.5
    t1new_aa += einsum("ldme,limdae->ia", v_bbbb_ovov, t3_babbab) * 0.5
    t1new_aa += einsum("jbmd,imjadb->ia", v_aabb_ovov, t3_abaaba)
    t1new_bb = np.zeros((nocc[1], nvir[1]), dtype=np.float64)
    t1new_bb += einsum("jbkc,jikbac->ia", v_aaaa_ovov, t3_abaaba) * 0.5
    t1new_bb += einsum("ldme,ilmade->ia", v_bbbb_ovov, t3_bbbbbb) * 0.5
    t1new_bb += einsum("kcmd,ikmacd->ia", v_aabb_ovov, t3_babbab)
    return t1new_aa, t1new_bb

def t2_residual_uhf(t1, t2, t3, t4, f, v):
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
    t2new_aaaa = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    t2new_aaaa += einsum("kcld,ijklabcd->ijab", v_aaaa_ovov, t4_aaaaaaaa) * 0.5
    t2new_aaaa += einsum("menf,imjnaebf->ijab", v_bbbb_ovov, t4_abababab) * 0.5
    t2new_aaaa += einsum("lcne,ijlnabce->ijab", v_aabb_ovov, t4_aaabaaab)
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
    t2new_aaaa += einsum("ijab->ijab", x6) * -1.0
    t2new_aaaa += einsum("jiab->ijab", x6)
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
    t2new_aaaa += einsum("ijab->ijab", x13)
    t2new_aaaa += einsum("ijba->ijab", x13) * -1.0
    x14 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x14 += einsum("ibja->ijab", v_aaaa_ovov) * -1.0
    x14 += einsum("iajb->ijab", v_aaaa_ovov)
    x15 = np.zeros((nocc[0], nvir[0]), dtype=np.float64)
    x15 += einsum("ia->ia", f_aa_ov)
    x15 += einsum("jb,iajb->ia", t1_bb, v_aabb_ovov)
    x15 += einsum("kc,kiac->ia", t1_aa, x14) * -1.0
    t2new_aaaa += einsum("lc,ijlabc->ijab", x15, t3_aaaaaa)
    x16 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x16 += einsum("ibja->ijab", v_bbbb_ovov)
    x16 += einsum("iajb->ijab", v_bbbb_ovov) * -1.0
    x17 = np.zeros((nocc[1], nvir[1]), dtype=np.float64)
    x17 += einsum("ia->ia", f_bb_ov)
    x17 += einsum("jb,jbia->ia", t1_aa, v_aabb_ovov)
    x17 += einsum("kc,kica->ia", t1_bb, x16) * -1.0
    t2new_aaaa += einsum("ne,injaeb->ijab", x17, t3_abaaba)
    t2new_bbbb = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    t2new_bbbb += einsum("ldkc,lijkdabc->ijab", v_aabb_ovov, t4_abbbabbb)
    t2new_bbbb += einsum("ldme,limjdaeb->ijab", v_aaaa_ovov, t4_abababab) * 0.5
    t2new_bbbb += einsum("kcnf,ijknabcf->ijab", v_bbbb_ovov, t4_bbbbbbbb) * 0.5
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
    t2new_bbbb += einsum("ijab->ijab", x6) * -1.0
    t2new_bbbb += einsum("jiab->ijab", x6)
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
    t2new_bbbb += einsum("ijab->ijab", x13)
    t2new_bbbb += einsum("ijba->ijab", x13) * -1.0
    x14 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x14 += einsum("ibja->ijab", v_bbbb_ovov)
    x14 += einsum("iajb->ijab", v_bbbb_ovov) * -1.0
    x15 = np.zeros((nocc[1], nvir[1]), dtype=np.float64)
    x15 += einsum("ia->ia", f_bb_ov)
    x15 += einsum("jb,jbia->ia", t1_aa, v_aabb_ovov)
    x15 += einsum("kc,kica->ia", t1_bb, x14) * -1.0
    t2new_bbbb += einsum("kc,ijkabc->ijab", x15, t3_bbbbbb)
    x16 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x16 += einsum("ibja->ijab", v_aaaa_ovov) * -1.0
    x16 += einsum("iajb->ijab", v_aaaa_ovov)
    x17 = np.zeros((nocc[0], nvir[0]), dtype=np.float64)
    x17 += einsum("ia->ia", f_aa_ov)
    x17 += einsum("jb,iajb->ia", t1_bb, v_aabb_ovov)
    x17 += einsum("kc,kiac->ia", t1_aa, x16) * -1.0
    t2new_bbbb += einsum("ld,iljadb->ijab", x17, t3_babbab)
    t2new_abab = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    t2new_abab += einsum("kcld,ikljacdb->ijab", v_aaaa_ovov, t4_aaabaaab) * 0.5
    t2new_abab += einsum("kcbe,ijkaec->ijab", v_aabb_ovvv, t3_abaaba)
    t2new_abab += einsum("mebf,jimeaf->ijab", v_bbbb_ovvv, t3_babbab) * -1.0
    t2new_abab += einsum("kcad,ijkcbd->ijab", v_aaaa_ovvv, t3_abaaba) * -1.0
    t2new_abab += einsum("acme,jimbce->ijab", v_aabb_vvov, t3_babbab)
    t2new_abab += einsum("mfne,ijmnabef->ijab", v_bbbb_ovov, t4_abbbabbb) * -0.5
    t2new_abab += einsum("kcme,ijkmabce->ijab", v_aabb_ovov, t4_abababab)
    x0 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=np.float64)
    x0 += einsum("jkia->iajk", v_aabb_ooov)
    x0 += einsum("kb,jbia->iajk", t1_aa, v_aabb_ovov)
    t2new_abab += einsum("meki,jkmbae->ijab", x0, t3_babbab) * -1.0
    x1 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0]), dtype=np.float64)
    x1 += einsum("ikja->ijka", v_aaaa_ooov)
    x1 += einsum("ib,jakb->ijka", t1_aa, v_aaaa_ovov)
    t2new_abab += einsum("iklc,kjlabc->ijab", x1, t3_abaaba)
    x2 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1]), dtype=np.float64)
    x2 += einsum("ikja->ijka", v_bbbb_ooov)
    x2 += einsum("ib,jakb->ijka", t1_bb, v_bbbb_ovov)
    t2new_abab += einsum("jnme,minbae->ijab", x2, t3_babbab) * -1.0
    x3 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=np.float64)
    x3 += einsum("kaij->ijka", v_aabb_ovoo)
    x3 += einsum("jb,kaib->ijka", t1_bb, v_aabb_ovov)
    t2new_abab += einsum("mjkc,imkabc->ijab", x3, t3_abaaba) * -1.0
    x4 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x4 += einsum("ibja->ijab", v_aaaa_ovov) * -1.0
    x4 += einsum("iajb->ijab", v_aaaa_ovov)
    x5 = np.zeros((nocc[0], nvir[0]), dtype=np.float64)
    x5 += einsum("ia->ia", f_aa_ov)
    x5 += einsum("jb,iajb->ia", t1_bb, v_aabb_ovov)
    x5 += einsum("kc,kiac->ia", t1_aa, x4) * -1.0
    t2new_abab += einsum("kc,ijkabc->ijab", x5, t3_abaaba)
    x6 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x6 += einsum("ibja->ijab", v_bbbb_ovov)
    x6 += einsum("iajb->ijab", v_bbbb_ovov) * -1.0
    x7 = np.zeros((nocc[1], nvir[1]), dtype=np.float64)
    x7 += einsum("ia->ia", f_bb_ov)
    x7 += einsum("jb,jbia->ia", t1_aa, v_aabb_ovov)
    x7 += einsum("kc,kica->ia", t1_bb, x6) * -1.0
    t2new_abab += einsum("me,jimbae->ijab", x7, t3_babbab)
    x8 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0]), dtype=np.float64)
    x8 += einsum("kclb,ijlacb->iajk", v_aabb_ovov, t3_babbab)
    x8 += einsum("kcmd,jimcad->iajk", v_aaaa_ovov, t3_abaaba)
    t2new_abab += einsum("ka,jbik->ijab", t1_aa, x8) * -1.0
    x9 = np.zeros((nocc[1], nocc[1], nocc[0], nvir[0]), dtype=np.float64)
    x9 += einsum("jclb,iklbac->ijka", v_bbbb_ovov, t3_babbab) * -1.0
    x9 += einsum("mdjc,kimacd->ijka", v_aabb_ovov, t3_abaaba)
    t2new_abab += einsum("mb,jmia->ijab", t1_bb, x9) * -1.0
    return t2new_aaaa, t2new_abab, t2new_bbbb
