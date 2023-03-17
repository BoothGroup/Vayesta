import numpy as np
from vayesta.core.util import einsum

def t1_uhf(c1):
    c1_aa, c1_bb = c1
    nocc = (c1_aa.shape[0], c1_bb.shape[0])
    nvir = (c1_aa.shape[1], c1_bb.shape[1])
    t1_aa = np.zeros((nocc[0], nvir[0]), dtype=np.float64)
    t1_aa += einsum("ia->ia", c1_aa)
    t1_bb = np.zeros((nocc[1], nvir[1]), dtype=np.float64)
    t1_bb += einsum("ia->ia", c1_bb)
    return t1_aa, t1_bb

def t1_rhf(c1):
    nocc, nvir = c1.shape
    t1 = np.zeros((nocc, nvir), dtype=np.float64)
    t1 += einsum("ia->ia", c1)
    return t1

def t2_uhf(t1, c2):
    t1_aa, t1_bb = t1
    c2_aaaa, c2_abab, c2_bbbb = c2
    nocc = (t1_aa.shape[0], t1_bb.shape[0])
    nvir = (t1_aa.shape[1], t1_bb.shape[1])
    t2_aaaa = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    t2_aaaa += einsum("ijab->ijab", c2_aaaa)
    t2_aaaa += einsum("ib,ja->ijab", t1_aa, t1_aa)
    t2_aaaa += einsum("ia,jb->ijab", t1_aa, t1_aa) * -1.0
    t2_abab = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    t2_abab += einsum("ijab->ijab", c2_abab)
    t2_abab += einsum("ia,jb->ijab", t1_aa, t1_bb) * -1.0
    t2_bbbb = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    t2_bbbb += einsum("ijab->ijab", c2_bbbb)
    t2_bbbb += einsum("ib,ja->ijab", t1_bb, t1_bb)
    t2_bbbb += einsum("ia,jb->ijab", t1_bb, t1_bb) * -1.0
    return t2_aaaa, t2_abab, t2_bbbb

def t2_rhf(t1, c2):
    nocc, nvir = t1.shape
    t2 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    t2 += einsum("ijab->ijab", c2)
    t2 += einsum("ia,jb->ijab", t1, t1) * -1.0
    return t2

def t3_uhf(t1, t2, c3):
    t1_aa, t1_bb = t1
    t2_aaaa, t2_abab, t2_bbbb = t2
    c3_aaaaaa, c3_abaaba, c3_babbab, c3_bbbbbb = c3
    nocc = (t1_aa.shape[0], t1_bb.shape[0])
    nvir = (t1_aa.shape[1], t1_bb.shape[1])
    t3_aaaaaa = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=np.float64)
    t3_aaaaaa += einsum("ijkabc->ijkabc", c3_aaaaaa)
    t3_aaaaaa += einsum("ia,jkbc->ijkabc", t1_aa, t2_aaaa) * -1.0
    t3_aaaaaa += einsum("ic,jkab->ijkabc", t1_aa, t2_aaaa) * -1.0
    t3_aaaaaa += einsum("ja,ikbc->ijkabc", t1_aa, t2_aaaa)
    t3_aaaaaa += einsum("jc,ikab->ijkabc", t1_aa, t2_aaaa)
    t3_aaaaaa += einsum("ka,ijbc->ijkabc", t1_aa, t2_aaaa) * -1.0
    t3_aaaaaa += einsum("kc,ijab->ijkabc", t1_aa, t2_aaaa) * -1.0
    x0 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x0 += einsum("ijab->ijab", t2_aaaa)
    x0 += einsum("ib,ja->ijab", t1_aa, t1_aa) * -1.0
    x0 += einsum("ia,jb->ijab", t1_aa, t1_aa)
    t3_aaaaaa += einsum("ib,kjca->ijkabc", t1_aa, x0)
    t3_aaaaaa += einsum("kb,jica->ijkabc", t1_aa, x0)
    t3_aaaaaa += einsum("jb,kica->ijkabc", t1_aa, x0) * -1.0
    t3_abaaba = np.zeros((nocc[0], nocc[1], nocc[0], nvir[0], nvir[1], nvir[0]), dtype=np.float64)
    t3_abaaba += einsum("ijkabc->ijkabc", c3_abaaba)
    t3_abaaba += einsum("ia,kjcb->ijkabc", t1_aa, t2_abab) * -1.0
    t3_abaaba += einsum("ic,kjab->ijkabc", t1_aa, t2_abab)
    t3_abaaba += einsum("ka,ijcb->ijkabc", t1_aa, t2_abab)
    t3_abaaba += einsum("kc,ijab->ijkabc", t1_aa, t2_abab) * -1.0
    x0 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x0 += einsum("ijab->ijab", t2_aaaa)
    x0 += einsum("ib,ja->ijab", t1_aa, t1_aa) * -1.0
    x0 += einsum("ia,jb->ijab", t1_aa, t1_aa)
    t3_abaaba += einsum("jb,kica->ijkabc", t1_bb, x0) * -1.0
    t3_babbab = np.zeros((nocc[1], nocc[0], nocc[1], nvir[1], nvir[0], nvir[1]), dtype=np.float64)
    t3_babbab += einsum("ijkabc->ijkabc", c3_babbab)
    t3_babbab += einsum("ia,jkbc->ijkabc", t1_bb, t2_abab) * -1.0
    t3_babbab += einsum("ic,jkba->ijkabc", t1_bb, t2_abab)
    t3_babbab += einsum("ka,jibc->ijkabc", t1_bb, t2_abab)
    t3_babbab += einsum("kc,jiba->ijkabc", t1_bb, t2_abab) * -1.0
    x0 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x0 += einsum("ijab->ijab", t2_bbbb)
    x0 += einsum("ib,ja->ijab", t1_bb, t1_bb) * -1.0
    x0 += einsum("ia,jb->ijab", t1_bb, t1_bb)
    t3_babbab += einsum("jb,kica->ijkabc", t1_aa, x0) * -1.0
    t3_bbbbbb = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=np.float64)
    t3_bbbbbb += einsum("ijkabc->ijkabc", c3_bbbbbb)
    t3_bbbbbb += einsum("ia,jkbc->ijkabc", t1_bb, t2_bbbb) * -1.0
    t3_bbbbbb += einsum("ic,jkab->ijkabc", t1_bb, t2_bbbb) * -1.0
    t3_bbbbbb += einsum("ja,ikbc->ijkabc", t1_bb, t2_bbbb)
    t3_bbbbbb += einsum("jc,ikab->ijkabc", t1_bb, t2_bbbb)
    t3_bbbbbb += einsum("ka,ijbc->ijkabc", t1_bb, t2_bbbb) * -1.0
    t3_bbbbbb += einsum("kc,ijab->ijkabc", t1_bb, t2_bbbb) * -1.0
    x0 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x0 += einsum("ijab->ijab", t2_bbbb)
    x0 += einsum("ib,ja->ijab", t1_bb, t1_bb) * -1.0
    x0 += einsum("ia,jb->ijab", t1_bb, t1_bb)
    t3_bbbbbb += einsum("ib,kjca->ijkabc", t1_bb, x0)
    t3_bbbbbb += einsum("kb,jica->ijkabc", t1_bb, x0)
    t3_bbbbbb += einsum("jb,kica->ijkabc", t1_bb, x0) * -1.0
    return t3_aaaaaa, t3_abaaba, t3_babbab, t3_bbbbbb

def t3_rhf(t1, t2, c3):
    nocc, nvir = t1.shape
    t3 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    t3 += einsum("ijkabc->ijkabc", c3)
    t3 += einsum("ia,kjcb->ijkabc", t1, t2) * -1.0
    t3 += einsum("ic,kjab->ijkabc", t1, t2)
    t3 += einsum("ka,ijcb->ijkabc", t1, t2)
    t3 += einsum("kc,ijab->ijkabc", t1, t2) * -1.0
    x0 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x0 += einsum("ijab->ijab", t2) * -1.0
    x0 += einsum("ijba->ijab", t2)
    x0 += einsum("ib,ja->ijab", t1, t1)
    x0 += einsum("ia,jb->ijab", t1, t1) * -1.0
    t3 += einsum("jb,ikca->ijkabc", t1, x0) * -1.0
    return t3

def t4_uhf(t1, t2, t3, c4):
    t1_aa, t1_bb = t1
    t2_aaaa, t2_abab, t2_bbbb = t2
    t3_aaaaaa, t3_abaaba, t3_babbab, t3_bbbbbb = t3
    c4_aaaaaaaa, c4_aaabaaab, c4_abababab, c4_abbbabbb, c4_bbbbbbbb = c4
    nocc = (t1_aa.shape[0], t1_bb.shape[0])
    nvir = (t1_aa.shape[1], t1_bb.shape[1])
    t4_aaaaaaaa = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0], nvir[0]), dtype=np.float64)
    t4_aaaaaaaa += einsum("ijklabcd->ijklabcd", c4_aaaaaaaa)
    t4_aaaaaaaa += einsum("la,ijkbcd->ijklabcd", t1_aa, t3_aaaaaa)
    t4_aaaaaaaa += einsum("lb,ijkacd->ijklabcd", t1_aa, t3_aaaaaa) * -1.0
    t4_aaaaaaaa += einsum("lc,ijkabd->ijklabcd", t1_aa, t3_aaaaaa)
    t4_aaaaaaaa += einsum("ld,ijkabc->ijklabcd", t1_aa, t3_aaaaaa) * -1.0
    t4_aaaaaaaa += einsum("ilcd,jkab->ijklabcd", t2_aaaa, t2_aaaa) * -1.0
    t4_aaaaaaaa += einsum("ilbd,jkac->ijklabcd", t2_aaaa, t2_aaaa)
    t4_aaaaaaaa += einsum("ilad,jkbc->ijklabcd", t2_aaaa, t2_aaaa) * -1.0
    t4_aaaaaaaa += einsum("ilbc,jkad->ijklabcd", t2_aaaa, t2_aaaa) * -1.0
    t4_aaaaaaaa += einsum("ilac,jkbd->ijklabcd", t2_aaaa, t2_aaaa)
    t4_aaaaaaaa += einsum("ilab,jkcd->ijklabcd", t2_aaaa, t2_aaaa) * -1.0
    t4_aaaaaaaa += einsum("ikcd,jlab->ijklabcd", t2_aaaa, t2_aaaa)
    t4_aaaaaaaa += einsum("ikbd,jlac->ijklabcd", t2_aaaa, t2_aaaa) * -1.0
    t4_aaaaaaaa += einsum("ikad,jlbc->ijklabcd", t2_aaaa, t2_aaaa)
    t4_aaaaaaaa += einsum("ikbc,jlad->ijklabcd", t2_aaaa, t2_aaaa)
    t4_aaaaaaaa += einsum("ikac,jlbd->ijklabcd", t2_aaaa, t2_aaaa) * -1.0
    t4_aaaaaaaa += einsum("ikab,jlcd->ijklabcd", t2_aaaa, t2_aaaa)
    t4_aaaaaaaa += einsum("ijcd,klab->ijklabcd", t2_aaaa, t2_aaaa) * -1.0
    t4_aaaaaaaa += einsum("ijbd,klac->ijklabcd", t2_aaaa, t2_aaaa)
    t4_aaaaaaaa += einsum("ijad,klbc->ijklabcd", t2_aaaa, t2_aaaa) * -1.0
    t4_aaaaaaaa += einsum("ijbc,klad->ijklabcd", t2_aaaa, t2_aaaa) * -1.0
    t4_aaaaaaaa += einsum("ijac,klbd->ijklabcd", t2_aaaa, t2_aaaa)
    t4_aaaaaaaa += einsum("ijab,klcd->ijklabcd", t2_aaaa, t2_aaaa) * -1.0
    x0 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x0 += einsum("ijab->ijab", t2_aaaa)
    x0 += einsum("ib,ja->ijab", t1_aa, t1_aa) * -1.0
    x0 += einsum("ia,jb->ijab", t1_aa, t1_aa)
    x1 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=np.float64)
    x1 += einsum("ijkabc->ijkabc", t3_aaaaaa)
    x1 += einsum("ja,ikbc->ijkabc", t1_aa, t2_aaaa) * -1.0
    x1 += einsum("jb,ikac->ijkabc", t1_aa, t2_aaaa)
    x1 += einsum("jc,ikab->ijkabc", t1_aa, t2_aaaa) * -1.0
    x1 += einsum("ka,ijbc->ijkabc", t1_aa, t2_aaaa)
    x1 += einsum("kb,ijac->ijkabc", t1_aa, t2_aaaa) * -1.0
    x1 += einsum("kc,ijab->ijkabc", t1_aa, t2_aaaa)
    x1 += einsum("ia,kjcb->ijkabc", t1_aa, x0)
    x1 += einsum("ic,kjba->ijkabc", t1_aa, x0)
    x1 += einsum("ib,kjca->ijkabc", t1_aa, x0) * -1.0
    t4_aaaaaaaa += einsum("ib,ljkdac->ijklabcd", t1_aa, x1)
    t4_aaaaaaaa += einsum("id,ljkcab->ijklabcd", t1_aa, x1)
    t4_aaaaaaaa += einsum("ia,ljkdbc->ijklabcd", t1_aa, x1) * -1.0
    t4_aaaaaaaa += einsum("ic,ljkdab->ijklabcd", t1_aa, x1) * -1.0
    x2 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=np.float64)
    x2 += einsum("ijkabc->ijkabc", t3_aaaaaa)
    x2 += einsum("ia,jkbc->ijkabc", t1_aa, t2_aaaa)
    x2 += einsum("ib,jkac->ijkabc", t1_aa, t2_aaaa) * -1.0
    x2 += einsum("ic,jkab->ijkabc", t1_aa, t2_aaaa)
    x2 += einsum("ka,ijbc->ijkabc", t1_aa, t2_aaaa)
    x2 += einsum("kb,ijac->ijkabc", t1_aa, t2_aaaa) * -1.0
    x2 += einsum("kc,ijab->ijkabc", t1_aa, t2_aaaa)
    t4_aaaaaaaa += einsum("ja,likdbc->ijklabcd", t1_aa, x2)
    t4_aaaaaaaa += einsum("jc,likdab->ijklabcd", t1_aa, x2)
    t4_aaaaaaaa += einsum("jb,likdac->ijklabcd", t1_aa, x2) * -1.0
    t4_aaaaaaaa += einsum("jd,likcab->ijklabcd", t1_aa, x2) * -1.0
    x3 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=np.float64)
    x3 += einsum("ijkabc->ijkabc", t3_aaaaaa)
    x3 += einsum("ia,jkbc->ijkabc", t1_aa, t2_aaaa)
    x3 += einsum("ib,jkac->ijkabc", t1_aa, t2_aaaa) * -1.0
    x3 += einsum("ic,jkab->ijkabc", t1_aa, t2_aaaa)
    t4_aaaaaaaa += einsum("kb,lijdac->ijklabcd", t1_aa, x3)
    t4_aaaaaaaa += einsum("kd,lijcab->ijklabcd", t1_aa, x3)
    t4_aaaaaaaa += einsum("ka,lijdbc->ijklabcd", t1_aa, x3) * -1.0
    t4_aaaaaaaa += einsum("kc,lijdab->ijklabcd", t1_aa, x3) * -1.0
    t4_aaabaaab = np.zeros((nocc[0], nocc[0], nocc[0], nocc[1], nvir[0], nvir[0], nvir[0], nvir[1]), dtype=np.float64)
    t4_aaabaaab += einsum("ijklabcd->ijklabcd", c4_aaabaaab)
    t4_aaabaaab += einsum("ia,jlkbdc->ijklabcd", t1_aa, t3_abaaba) * -1.0
    t4_aaabaaab += einsum("ic,jlkadb->ijklabcd", t1_aa, t3_abaaba) * -1.0
    t4_aaabaaab += einsum("ja,ilkbdc->ijklabcd", t1_aa, t3_abaaba)
    t4_aaabaaab += einsum("jc,ilkadb->ijklabcd", t1_aa, t3_abaaba)
    t4_aaabaaab += einsum("ka,iljbdc->ijklabcd", t1_aa, t3_abaaba) * -1.0
    t4_aaabaaab += einsum("kc,iljadb->ijklabcd", t1_aa, t3_abaaba) * -1.0
    t4_aaabaaab += einsum("klcd,ijab->ijklabcd", t2_abab, t2_aaaa) * -1.0
    t4_aaabaaab += einsum("klad,ijbc->ijklabcd", t2_abab, t2_aaaa) * -1.0
    t4_aaabaaab += einsum("jlcd,ikab->ijklabcd", t2_abab, t2_aaaa)
    t4_aaabaaab += einsum("jlad,ikbc->ijklabcd", t2_abab, t2_aaaa)
    t4_aaabaaab += einsum("ilcd,jkab->ijklabcd", t2_abab, t2_aaaa) * -1.0
    t4_aaabaaab += einsum("ilad,jkbc->ijklabcd", t2_abab, t2_aaaa) * -1.0
    x0 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x0 += einsum("ijab->ijab", t2_aaaa)
    x0 += einsum("ib,ja->ijab", t1_aa, t1_aa) * -1.0
    x0 += einsum("ia,jb->ijab", t1_aa, t1_aa)
    t4_aaabaaab += einsum("ilbd,kjca->ijklabcd", t2_abab, x0)
    t4_aaabaaab += einsum("klbd,jica->ijklabcd", t2_abab, x0)
    t4_aaabaaab += einsum("jlbd,kica->ijklabcd", t2_abab, x0) * -1.0
    x1 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=np.float64)
    x1 += einsum("ijkabc->ijkabc", t3_aaaaaa)
    x1 += einsum("ja,ikbc->ijkabc", t1_aa, t2_aaaa) * -1.0
    x1 += einsum("jb,ikac->ijkabc", t1_aa, t2_aaaa)
    x1 += einsum("jc,ikab->ijkabc", t1_aa, t2_aaaa) * -1.0
    x1 += einsum("ka,ijbc->ijkabc", t1_aa, t2_aaaa)
    x1 += einsum("kb,ijac->ijkabc", t1_aa, t2_aaaa) * -1.0
    x1 += einsum("kc,ijab->ijkabc", t1_aa, t2_aaaa)
    x1 += einsum("ia,kjcb->ijkabc", t1_aa, x0)
    x1 += einsum("ic,kjba->ijkabc", t1_aa, x0)
    x1 += einsum("ib,kjca->ijkabc", t1_aa, x0) * -1.0
    t4_aaabaaab += einsum("ld,kijcab->ijklabcd", t1_bb, x1) * -1.0
    x2 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x2 += einsum("jikbac->iajkbc", t3_abaaba)
    x2 += einsum("jb,kica->iajkbc", t1_aa, t2_abab)
    x2 += einsum("jc,kiba->iajkbc", t1_aa, t2_abab) * -1.0
    x2 += einsum("kb,jica->iajkbc", t1_aa, t2_abab) * -1.0
    x2 += einsum("kc,jiba->iajkbc", t1_aa, t2_abab)
    t4_aaabaaab += einsum("ib,ldkjca->ijklabcd", t1_aa, x2)
    t4_aaabaaab += einsum("kb,ldjica->ijklabcd", t1_aa, x2)
    t4_aaabaaab += einsum("jb,ldkica->ijklabcd", t1_aa, x2) * -1.0
    t4_abababab = np.zeros((nocc[0], nocc[1], nocc[0], nocc[1], nvir[0], nvir[1], nvir[0], nvir[1]), dtype=np.float64)
    t4_abababab += einsum("ijklabcd->ijklabcd", c4_abababab)
    t4_abababab += einsum("ia,jklbcd->ijklabcd", t1_aa, t3_babbab) * -1.0
    t4_abababab += einsum("ic,jklbad->ijklabcd", t1_aa, t3_babbab)
    t4_abababab += einsum("ka,jilbcd->ijklabcd", t1_aa, t3_babbab)
    t4_abababab += einsum("kc,jilbad->ijklabcd", t1_aa, t3_babbab) * -1.0
    t4_abababab += einsum("ilab,kjcd->ijklabcd", t2_abab, t2_abab)
    t4_abababab += einsum("ilcb,kjad->ijklabcd", t2_abab, t2_abab) * -1.0
    t4_abababab += einsum("ijcd,klab->ijklabcd", t2_abab, t2_abab) * -1.0
    t4_abababab += einsum("ijad,klcb->ijklabcd", t2_abab, t2_abab)
    t4_abababab += einsum("ilad,kjcb->ijklabcd", t2_abab, t2_abab) * -1.0
    t4_abababab += einsum("ilcd,kjab->ijklabcd", t2_abab, t2_abab)
    t4_abababab += einsum("ijcb,klad->ijklabcd", t2_abab, t2_abab)
    t4_abababab += einsum("ijab,klcd->ijklabcd", t2_abab, t2_abab) * -1.0
    x0 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x0 += einsum("ijab->ijab", t2_aaaa)
    x0 += einsum("ib,ja->ijab", t1_aa, t1_aa) * -1.0
    x0 += einsum("ia,jb->ijab", t1_aa, t1_aa)
    x1 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1 += einsum("ijab->ijab", t2_bbbb)
    x1 += einsum("ib,ja->ijab", t1_bb, t1_bb) * -1.0
    x1 += einsum("ia,jb->ijab", t1_bb, t1_bb)
    t4_abababab += einsum("kica,ljdb->ijklabcd", x0, x1) * -1.0
    x2 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x2 += einsum("jikbac->iajkbc", t3_abaaba)
    x2 += einsum("jb,kica->iajkbc", t1_aa, t2_abab)
    x2 += einsum("jc,kiba->iajkbc", t1_aa, t2_abab) * -1.0
    x2 += einsum("kb,jica->iajkbc", t1_aa, t2_abab) * -1.0
    x2 += einsum("kc,jiba->iajkbc", t1_aa, t2_abab)
    t4_abababab += einsum("jd,lbkica->ijklabcd", t1_bb, x2)
    t4_abababab += einsum("lb,jdkica->ijklabcd", t1_bb, x2)
    t4_abababab += einsum("jb,ldkica->ijklabcd", t1_bb, x2) * -1.0
    t4_abababab += einsum("ld,jbkica->ijklabcd", t1_bb, x2) * -1.0
    t4_abbbabbb = np.zeros((nocc[0], nocc[1], nocc[1], nocc[1], nvir[0], nvir[1], nvir[1], nvir[1]), dtype=np.float64)
    t4_abbbabbb += einsum("ijklabcd->ijklabcd", c4_abbbabbb)
    t4_abbbabbb += einsum("jb,kilcad->ijklabcd", t1_bb, t3_babbab) * -1.0
    t4_abbbabbb += einsum("jd,kilbac->ijklabcd", t1_bb, t3_babbab) * -1.0
    t4_abbbabbb += einsum("kb,jilcad->ijklabcd", t1_bb, t3_babbab)
    t4_abbbabbb += einsum("kd,jilbac->ijklabcd", t1_bb, t3_babbab)
    t4_abbbabbb += einsum("lb,jikcad->ijklabcd", t1_bb, t3_babbab) * -1.0
    t4_abbbabbb += einsum("ld,jikbac->ijklabcd", t1_bb, t3_babbab) * -1.0
    t4_abbbabbb += einsum("ilad,jkbc->ijklabcd", t2_abab, t2_bbbb) * -1.0
    t4_abbbabbb += einsum("ilab,jkcd->ijklabcd", t2_abab, t2_bbbb) * -1.0
    t4_abbbabbb += einsum("ikad,jlbc->ijklabcd", t2_abab, t2_bbbb)
    t4_abbbabbb += einsum("ikab,jlcd->ijklabcd", t2_abab, t2_bbbb)
    t4_abbbabbb += einsum("ijad,klbc->ijklabcd", t2_abab, t2_bbbb) * -1.0
    t4_abbbabbb += einsum("ijab,klcd->ijklabcd", t2_abab, t2_bbbb) * -1.0
    x0 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x0 += einsum("ijab->ijab", t2_bbbb)
    x0 += einsum("ib,ja->ijab", t1_bb, t1_bb) * -1.0
    x0 += einsum("ia,jb->ijab", t1_bb, t1_bb)
    t4_abbbabbb += einsum("ilac,kjdb->ijklabcd", t2_abab, x0)
    t4_abbbabbb += einsum("ijac,lkdb->ijklabcd", t2_abab, x0)
    t4_abbbabbb += einsum("ikac,ljdb->ijklabcd", t2_abab, x0) * -1.0
    x1 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=np.float64)
    x1 += einsum("ijkabc->ijkabc", t3_bbbbbb)
    x1 += einsum("ja,ikbc->ijkabc", t1_bb, t2_bbbb) * -1.0
    x1 += einsum("jb,ikac->ijkabc", t1_bb, t2_bbbb)
    x1 += einsum("jc,ikab->ijkabc", t1_bb, t2_bbbb) * -1.0
    x1 += einsum("ka,ijbc->ijkabc", t1_bb, t2_bbbb)
    x1 += einsum("kb,ijac->ijkabc", t1_bb, t2_bbbb) * -1.0
    x1 += einsum("kc,ijab->ijkabc", t1_bb, t2_bbbb)
    x1 += einsum("ia,kjcb->ijkabc", t1_bb, x0)
    x1 += einsum("ic,kjba->ijkabc", t1_bb, x0)
    x1 += einsum("ib,kjca->ijkabc", t1_bb, x0) * -1.0
    t4_abbbabbb += einsum("ia,ljkdbc->ijklabcd", t1_aa, x1) * -1.0
    x2 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=np.float64)
    x2 += einsum("ikjacb->ijabkc", t3_babbab)
    x2 += einsum("ia,kjcb->ijabkc", t1_bb, t2_abab)
    x2 += einsum("ib,kjca->ijabkc", t1_bb, t2_abab) * -1.0
    x2 += einsum("ja,kicb->ijabkc", t1_bb, t2_abab) * -1.0
    x2 += einsum("jb,kica->ijabkc", t1_bb, t2_abab)
    t4_abbbabbb += einsum("jc,lkdbia->ijklabcd", t1_bb, x2)
    t4_abbbabbb += einsum("lc,kjdbia->ijklabcd", t1_bb, x2)
    t4_abbbabbb += einsum("kc,ljdbia->ijklabcd", t1_bb, x2) * -1.0
    t4_bbbbbbbb = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1], nvir[1]), dtype=np.float64)
    t4_bbbbbbbb += einsum("ijklabcd->ijklabcd", c4_bbbbbbbb)
    t4_bbbbbbbb += einsum("la,ijkbcd->ijklabcd", t1_bb, t3_bbbbbb)
    t4_bbbbbbbb += einsum("lb,ijkacd->ijklabcd", t1_bb, t3_bbbbbb) * -1.0
    t4_bbbbbbbb += einsum("lc,ijkabd->ijklabcd", t1_bb, t3_bbbbbb)
    t4_bbbbbbbb += einsum("ld,ijkabc->ijklabcd", t1_bb, t3_bbbbbb) * -1.0
    t4_bbbbbbbb += einsum("ilcd,jkab->ijklabcd", t2_bbbb, t2_bbbb) * -1.0
    t4_bbbbbbbb += einsum("ilbd,jkac->ijklabcd", t2_bbbb, t2_bbbb)
    t4_bbbbbbbb += einsum("ilad,jkbc->ijklabcd", t2_bbbb, t2_bbbb) * -1.0
    t4_bbbbbbbb += einsum("ilbc,jkad->ijklabcd", t2_bbbb, t2_bbbb) * -1.0
    t4_bbbbbbbb += einsum("ilac,jkbd->ijklabcd", t2_bbbb, t2_bbbb)
    t4_bbbbbbbb += einsum("ilab,jkcd->ijklabcd", t2_bbbb, t2_bbbb) * -1.0
    t4_bbbbbbbb += einsum("ikcd,jlab->ijklabcd", t2_bbbb, t2_bbbb)
    t4_bbbbbbbb += einsum("ikbd,jlac->ijklabcd", t2_bbbb, t2_bbbb) * -1.0
    t4_bbbbbbbb += einsum("ikad,jlbc->ijklabcd", t2_bbbb, t2_bbbb)
    t4_bbbbbbbb += einsum("ikbc,jlad->ijklabcd", t2_bbbb, t2_bbbb)
    t4_bbbbbbbb += einsum("ikac,jlbd->ijklabcd", t2_bbbb, t2_bbbb) * -1.0
    t4_bbbbbbbb += einsum("ikab,jlcd->ijklabcd", t2_bbbb, t2_bbbb)
    t4_bbbbbbbb += einsum("ijcd,klab->ijklabcd", t2_bbbb, t2_bbbb) * -1.0
    t4_bbbbbbbb += einsum("ijbd,klac->ijklabcd", t2_bbbb, t2_bbbb)
    t4_bbbbbbbb += einsum("ijad,klbc->ijklabcd", t2_bbbb, t2_bbbb) * -1.0
    t4_bbbbbbbb += einsum("ijbc,klad->ijklabcd", t2_bbbb, t2_bbbb) * -1.0
    t4_bbbbbbbb += einsum("ijac,klbd->ijklabcd", t2_bbbb, t2_bbbb)
    t4_bbbbbbbb += einsum("ijab,klcd->ijklabcd", t2_bbbb, t2_bbbb) * -1.0
    x0 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x0 += einsum("ijab->ijab", t2_bbbb)
    x0 += einsum("ib,ja->ijab", t1_bb, t1_bb) * -1.0
    x0 += einsum("ia,jb->ijab", t1_bb, t1_bb)
    x1 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=np.float64)
    x1 += einsum("ijkabc->ijkabc", t3_bbbbbb)
    x1 += einsum("ja,ikbc->ijkabc", t1_bb, t2_bbbb) * -1.0
    x1 += einsum("jb,ikac->ijkabc", t1_bb, t2_bbbb)
    x1 += einsum("jc,ikab->ijkabc", t1_bb, t2_bbbb) * -1.0
    x1 += einsum("ka,ijbc->ijkabc", t1_bb, t2_bbbb)
    x1 += einsum("kb,ijac->ijkabc", t1_bb, t2_bbbb) * -1.0
    x1 += einsum("kc,ijab->ijkabc", t1_bb, t2_bbbb)
    x1 += einsum("ia,kjcb->ijkabc", t1_bb, x0)
    x1 += einsum("ic,kjba->ijkabc", t1_bb, x0)
    x1 += einsum("ib,kjca->ijkabc", t1_bb, x0) * -1.0
    t4_bbbbbbbb += einsum("ib,ljkdac->ijklabcd", t1_bb, x1)
    t4_bbbbbbbb += einsum("id,ljkcab->ijklabcd", t1_bb, x1)
    t4_bbbbbbbb += einsum("ia,ljkdbc->ijklabcd", t1_bb, x1) * -1.0
    t4_bbbbbbbb += einsum("ic,ljkdab->ijklabcd", t1_bb, x1) * -1.0
    x2 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=np.float64)
    x2 += einsum("ijkabc->ijkabc", t3_bbbbbb)
    x2 += einsum("ia,jkbc->ijkabc", t1_bb, t2_bbbb)
    x2 += einsum("ib,jkac->ijkabc", t1_bb, t2_bbbb) * -1.0
    x2 += einsum("ic,jkab->ijkabc", t1_bb, t2_bbbb)
    x2 += einsum("ka,ijbc->ijkabc", t1_bb, t2_bbbb)
    x2 += einsum("kb,ijac->ijkabc", t1_bb, t2_bbbb) * -1.0
    x2 += einsum("kc,ijab->ijkabc", t1_bb, t2_bbbb)
    t4_bbbbbbbb += einsum("ja,likdbc->ijklabcd", t1_bb, x2)
    t4_bbbbbbbb += einsum("jc,likdab->ijklabcd", t1_bb, x2)
    t4_bbbbbbbb += einsum("jb,likdac->ijklabcd", t1_bb, x2) * -1.0
    t4_bbbbbbbb += einsum("jd,likcab->ijklabcd", t1_bb, x2) * -1.0
    x3 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=np.float64)
    x3 += einsum("ijkabc->ijkabc", t3_bbbbbb)
    x3 += einsum("ia,jkbc->ijkabc", t1_bb, t2_bbbb)
    x3 += einsum("ib,jkac->ijkabc", t1_bb, t2_bbbb) * -1.0
    x3 += einsum("ic,jkab->ijkabc", t1_bb, t2_bbbb)
    t4_bbbbbbbb += einsum("kb,lijdac->ijklabcd", t1_bb, x3)
    t4_bbbbbbbb += einsum("kd,lijcab->ijklabcd", t1_bb, x3)
    t4_bbbbbbbb += einsum("ka,lijdbc->ijklabcd", t1_bb, x3) * -1.0
    t4_bbbbbbbb += einsum("kc,lijdab->ijklabcd", t1_bb, x3) * -1.0
    return t4_aaaaaaaa, t4_aaabaaab, t4_abababab, t4_abbbabbb, t4_bbbbbbbb

def t4_rhf(t1, t2, t3, c4):
    nocc, nvir = t1.shape
    c4_abaaabaa, c4_abababab = c4
    t4_abababab = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir, nvir, nvir), dtype=np.float64)
    t4_abababab += einsum("ijklabcd->ijklabcd", c4_abababab)
    t4_abababab += einsum("jb,ilkadc->ijklabcd", t1, t3) * -1.0
    t4_abababab += einsum("jd,ilkabc->ijklabcd", t1, t3)
    t4_abababab += einsum("lb,ijkadc->ijklabcd", t1, t3)
    t4_abababab += einsum("ld,ijkabc->ijklabcd", t1, t3) * -1.0
    t4_abababab += einsum("ijab,klcd->ijklabcd", t2, t2) * -1.0
    t4_abababab += einsum("ijad,klcb->ijklabcd", t2, t2)
    t4_abababab += einsum("ijcb,klad->ijklabcd", t2, t2)
    t4_abababab += einsum("ijcd,klab->ijklabcd", t2, t2) * -1.0
    t4_abababab += einsum("ilab,kjcd->ijklabcd", t2, t2)
    t4_abababab += einsum("ilad,kjcb->ijklabcd", t2, t2) * -1.0
    t4_abababab += einsum("ilcb,kjad->ijklabcd", t2, t2) * -1.0
    t4_abababab += einsum("ilcd,kjab->ijklabcd", t2, t2)
    x0 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x0 += einsum("ijab->ijab", t2) * -1.0
    x0 += einsum("ijba->ijab", t2)
    x0 += einsum("ib,ja->ijab", t1, t1)
    x0 += einsum("ia,jb->ijab", t1, t1) * -1.0
    t4_abababab += einsum("ikac,jldb->ijklabcd", x0, x0)
    x1 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    x1 += einsum("ijkabc->ijkabc", t3)
    x1 += einsum("ia,jkbc->ijkabc", t1, t2)
    x1 += einsum("ic,jkba->ijkabc", t1, t2) * -1.0
    x1 += einsum("ka,jibc->ijkabc", t1, t2) * -1.0
    x1 += einsum("kc,jiba->ijkabc", t1, t2)
    t4_abababab += einsum("ic,jklbad->ijklabcd", t1, x1)
    t4_abababab += einsum("ka,jilbcd->ijklabcd", t1, x1)
    t4_abababab += einsum("ia,jklbcd->ijklabcd", t1, x1) * -1.0
    t4_abababab += einsum("kc,jilbad->ijklabcd", t1, x1) * -1.0
    t4_abaaabaa = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir, nvir, nvir), dtype=np.float64)
    #t4_abaaabaa += einsum("ikljacdb->ijklabcd", c4_abaaabaa)  # NOTE incorrect in generated eqns
    t4_abaaabaa += einsum("ijklabcd->ijklabcd", c4_abaaabaa)
    t4_abaaabaa += einsum("ia,kjlcbd->ijklabcd", t1, t3) * -1.0
    t4_abaaabaa += einsum("id,kjlabc->ijklabcd", t1, t3) * -1.0
    t4_abaaabaa += einsum("ka,ijlcbd->ijklabcd", t1, t3)
    t4_abaaabaa += einsum("kd,ijlabc->ijklabcd", t1, t3)
    t4_abaaabaa += einsum("la,ijkcbd->ijklabcd", t1, t3) * -1.0
    t4_abaaabaa += einsum("ld,ijkabc->ijklabcd", t1, t3) * -1.0
    t4_abaaabaa += einsum("ijab,klcd->ijklabcd", t2, t2) * -1.0
    t4_abaaabaa += einsum("ijab,kldc->ijklabcd", t2, t2)
    t4_abaaabaa += einsum("ijdb,klac->ijklabcd", t2, t2) * -1.0
    t4_abaaabaa += einsum("ijdb,klca->ijklabcd", t2, t2)
    t4_abaaabaa += einsum("ilac,kjdb->ijklabcd", t2, t2)
    t4_abaaabaa += einsum("ilcd,kjab->ijklabcd", t2, t2)
    t4_abaaabaa += einsum("ilca,kjdb->ijklabcd", t2, t2) * -1.0
    t4_abaaabaa += einsum("ildc,kjab->ijklabcd", t2, t2) * -1.0
    t4_abaaabaa += einsum("ikac,ljdb->ijklabcd", t2, t2) * -1.0
    t4_abaaabaa += einsum("ikcd,ljab->ijklabcd", t2, t2) * -1.0
    t4_abaaabaa += einsum("ikca,ljdb->ijklabcd", t2, t2)
    t4_abaaabaa += einsum("ikdc,ljab->ijklabcd", t2, t2)
    x0 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x0 += einsum("ijab->ijab", t2) * -1.0
    x0 += einsum("ijba->ijab", t2)
    x0 += einsum("ib,ja->ijab", t1, t1)
    x0 += einsum("ia,jb->ijab", t1, t1) * -1.0
    t4_abaaabaa += einsum("kjcb,ilda->ijklabcd", t2, x0) * -1.0
    x1 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x1 += einsum("ijab->ijab", t2)
    x1 += einsum("ijba->ijab", t2) * -1.0
    x1 += einsum("ib,ja->ijab", t1, t1) * -1.0
    x1 += einsum("ia,jb->ijab", t1, t1)
    t4_abaaabaa += einsum("ijcb,klda->ijklabcd", t2, x1) * -1.0
    t4_abaaabaa += einsum("ljcb,ikda->ijklabcd", t2, x1) * -1.0
    x2 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    x2 += einsum("ijkabc->ijkabc", t3) * -1.0
    x2 += einsum("ijkacb->ijkabc", t3)
    x2 += einsum("ijkbac->ijkabc", t3)
    x2 += einsum("ja,ikbc->ijkabc", t1, t2)
    x2 += einsum("ja,ikcb->ijkabc", t1, t2) * -1.0
    x2 += einsum("jb,ikac->ijkabc", t1, t2) * -1.0
    x2 += einsum("jb,ikca->ijkabc", t1, t2)
    x2 += einsum("jc,ikab->ijkabc", t1, t2)
    x2 += einsum("jc,ikba->ijkabc", t1, t2) * -1.0
    x2 += einsum("ka,ijbc->ijkabc", t1, t2) * -1.0
    x2 += einsum("ka,ijcb->ijkabc", t1, t2)
    x2 += einsum("kb,ijac->ijkabc", t1, t2)
    x2 += einsum("kb,ijca->ijkabc", t1, t2) * -1.0
    x2 += einsum("kc,ijab->ijkabc", t1, t2) * -1.0
    x2 += einsum("kc,ijba->ijkabc", t1, t2)
    x2 += einsum("ia,jkcb->ijkabc", t1, x0) * -1.0
    x2 += einsum("ib,jkca->ijkabc", t1, x1) * -1.0
    x2 += einsum("ic,jkab->ijkabc", t1, x1) * -1.0
    t4_abaaabaa += einsum("jb,iklacd->ijklabcd", t1, x2)
    x3 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    x3 += einsum("ijkabc->ijkabc", t3)
    x3 += einsum("ia,kjcb->ijkabc", t1, t2)
    x3 += einsum("ic,kjab->ijkabc", t1, t2) * -1.0
    x3 += einsum("ka,ijcb->ijkabc", t1, t2) * -1.0
    x3 += einsum("kc,ijab->ijkabc", t1, t2)
    t4_abaaabaa += einsum("ic,kjlabd->ijklabcd", t1, x3)
    t4_abaaabaa += einsum("lc,ijkabd->ijklabcd", t1, x3)
    t4_abaaabaa += einsum("kc,ijlabd->ijklabcd", t1, x3) * -1.0
    return t4_abaaabaa, t4_abababab
