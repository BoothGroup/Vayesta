import numpy as np
from vayesta.core.util import einsum

def t1_uhf_aa(c1=None, nocc=None, nvir=None):
    t1_aa = np.zeros((nocc[0], nvir[0]), dtype=np.float64)
    t1_aa += einsum("ia->ia", c1.aa)
    return t1_aa

def t1_uhf_bb(c1=None, nocc=None, nvir=None):
    t1_bb = np.zeros((nocc[1], nvir[1]), dtype=np.float64)
    t1_bb += einsum("ia->ia", c1.bb)
    return t1_bb

def t1_rhf(c1=None, nocc=None, nvir=None):
    t1 = np.zeros((nocc, nvir), dtype=np.float64)
    t1 += einsum("ia->ia", c1)
    return t1

def t2_uhf_aaaa(c2=None, t1=None, nocc=None, nvir=None):
    t2_aaaa = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    t2_aaaa += einsum("ijab->ijab", c2.aaaa)
    t2_aaaa += einsum("ib,ja->ijab", t1.aa, t1.aa)
    t2_aaaa += einsum("ia,jb->ijab", t1.aa, t1.aa) * -1.0
    return t2_aaaa

def t2_uhf_abab(c2=None, t1=None, nocc=None, nvir=None):
    t2_abab = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    t2_abab += einsum("ijab->ijab", c2.abab)
    t2_abab += einsum("ia,jb->ijab", t1.aa, t1.bb) * -1.0
    return t2_abab

def t2_uhf_baba(c2=None, t1=None, nocc=None, nvir=None):
    t2_baba = np.zeros((nocc[1], nocc[0], nvir[1], nvir[0]), dtype=np.float64)
    t2_baba += einsum("jiba->ijab", c2.abab)
    t2_baba += einsum("jb,ia->ijab", t1.aa, t1.bb) * -1.0
    return t2_baba

def t2_uhf_bbbb(c2=None, t1=None, nocc=None, nvir=None):
    t2_bbbb = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    t2_bbbb += einsum("ijab->ijab", c2.bbbb)
    t2_bbbb += einsum("ib,ja->ijab", t1.bb, t1.bb)
    t2_bbbb += einsum("ia,jb->ijab", t1.bb, t1.bb) * -1.0
    return t2_bbbb

def t2_rhf(c2=None, t1=None, nocc=None, nvir=None):
    t2 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    t2 += einsum("ijab->ijab", c2)
    t2 += einsum("ia,jb->ijab", t1, t1) * -1.0
    return t2

def t3_uhf_aaaaaa(c3=None, t1=None, t2=None, nocc=None, nvir=None):
    t3_aaaaaa = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=np.float64)
    t3_aaaaaa += einsum("ijkabc->ijkabc", c3.aaaaaa)
    t3_aaaaaa += einsum("ia,jkbc->ijkabc", t1.aa, t2.aaaa) * -1.0
    t3_aaaaaa += einsum("ic,jkab->ijkabc", t1.aa, t2.aaaa) * -1.0
    t3_aaaaaa += einsum("ja,ikbc->ijkabc", t1.aa, t2.aaaa)
    t3_aaaaaa += einsum("jc,ikab->ijkabc", t1.aa, t2.aaaa)
    t3_aaaaaa += einsum("ka,ijbc->ijkabc", t1.aa, t2.aaaa) * -1.0
    t3_aaaaaa += einsum("kc,ijab->ijkabc", t1.aa, t2.aaaa) * -1.0
    x0 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x0 += einsum("ijab->ijab", t2.aaaa)
    x0 += einsum("ib,ja->ijab", t1.aa, t1.aa) * -1.0
    x0 += einsum("ia,jb->ijab", t1.aa, t1.aa)
    t3_aaaaaa += einsum("ib,kjca->ijkabc", t1.aa, x0)
    t3_aaaaaa += einsum("kb,jica->ijkabc", t1.aa, x0)
    t3_aaaaaa += einsum("jb,kica->ijkabc", t1.aa, x0) * -1.0
    return t3_aaaaaa

def t3_uhf_aabaab(c3=None, t1=None, t2=None, nocc=None, nvir=None):
    t3_aabaab = np.zeros((nocc[0], nocc[0], nocc[1], nvir[0], nvir[0], nvir[1]), dtype=np.float64)
    t3_aabaab += einsum("ikjacb->ijkabc", c3.abaaba)
    t3_aabaab += einsum("ia,jkbc->ijkabc", t1.aa, t2.abab) * -1.0
    t3_aabaab += einsum("ib,jkac->ijkabc", t1.aa, t2.abab)
    t3_aabaab += einsum("ja,ikbc->ijkabc", t1.aa, t2.abab)
    t3_aabaab += einsum("jb,ikac->ijkabc", t1.aa, t2.abab) * -1.0
    x0 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x0 += einsum("ijab->ijab", t2.aaaa)
    x0 += einsum("ib,ja->ijab", t1.aa, t1.aa) * -1.0
    x0 += einsum("ia,jb->ijab", t1.aa, t1.aa)
    t3_aabaab += einsum("kc,jiba->ijkabc", t1.bb, x0) * -1.0
    return t3_aabaab

def t3_uhf_abaaba(c3=None, t1=None, t2=None, nocc=None, nvir=None):
    t3_abaaba = np.zeros((nocc[0], nocc[1], nocc[0], nvir[0], nvir[1], nvir[0]), dtype=np.float64)
    t3_abaaba += einsum("ijkabc->ijkabc", c3.abaaba)
    t3_abaaba += einsum("ia,kjcb->ijkabc", t1.aa, t2.abab) * -1.0
    t3_abaaba += einsum("ic,kjab->ijkabc", t1.aa, t2.abab)
    t3_abaaba += einsum("ka,ijcb->ijkabc", t1.aa, t2.abab)
    t3_abaaba += einsum("kc,ijab->ijkabc", t1.aa, t2.abab) * -1.0
    x0 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x0 += einsum("ijab->ijab", t2.aaaa)
    x0 += einsum("ib,ja->ijab", t1.aa, t1.aa) * -1.0
    x0 += einsum("ia,jb->ijab", t1.aa, t1.aa)
    t3_abaaba += einsum("jb,kica->ijkabc", t1.bb, x0) * -1.0
    return t3_abaaba

def t3_uhf_abbabb(c3=None, t1=None, t2=None, nocc=None, nvir=None):
    t3_abbabb = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0], nvir[1], nvir[1]), dtype=np.float64)
    t3_abbabb += einsum("jikbac->ijkabc", c3.babbab)
    t3_abbabb += einsum("jb,ikac->ijkabc", t1.bb, t2.abab) * -1.0
    t3_abbabb += einsum("jc,ikab->ijkabc", t1.bb, t2.abab)
    t3_abbabb += einsum("kb,ijac->ijkabc", t1.bb, t2.abab)
    t3_abbabb += einsum("kc,ijab->ijkabc", t1.bb, t2.abab) * -1.0
    x0 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x0 += einsum("ijab->ijab", t2.bbbb)
    x0 += einsum("ib,ja->ijab", t1.bb, t1.bb) * -1.0
    x0 += einsum("ia,jb->ijab", t1.bb, t1.bb)
    t3_abbabb += einsum("ia,kjcb->ijkabc", t1.aa, x0) * -1.0
    return t3_abbabb

def t3_uhf_baabaa(c3=None, t1=None, t2=None, nocc=None, nvir=None):
    t3_baabaa = np.zeros((nocc[1], nocc[0], nocc[0], nvir[1], nvir[0], nvir[0]), dtype=np.float64)
    t3_baabaa += einsum("jikbac->ijkabc", c3.abaaba)
    t3_baabaa += einsum("jb,kica->ijkabc", t1.aa, t2.abab) * -1.0
    t3_baabaa += einsum("jc,kiba->ijkabc", t1.aa, t2.abab)
    t3_baabaa += einsum("kb,jica->ijkabc", t1.aa, t2.abab)
    t3_baabaa += einsum("kc,jiba->ijkabc", t1.aa, t2.abab) * -1.0
    x0 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x0 += einsum("ijab->ijab", t2.aaaa)
    x0 += einsum("ib,ja->ijab", t1.aa, t1.aa) * -1.0
    x0 += einsum("ia,jb->ijab", t1.aa, t1.aa)
    t3_baabaa += einsum("ia,kjcb->ijkabc", t1.bb, x0) * -1.0
    return t3_baabaa

def t3_uhf_babbab(c3=None, t1=None, t2=None, nocc=None, nvir=None):
    t3_babbab = np.zeros((nocc[1], nocc[0], nocc[1], nvir[1], nvir[0], nvir[1]), dtype=np.float64)
    t3_babbab += einsum("ijkabc->ijkabc", c3.babbab)
    t3_babbab += einsum("ia,jkbc->ijkabc", t1.bb, t2.abab) * -1.0
    t3_babbab += einsum("ic,jkba->ijkabc", t1.bb, t2.abab)
    t3_babbab += einsum("ka,jibc->ijkabc", t1.bb, t2.abab)
    t3_babbab += einsum("kc,jiba->ijkabc", t1.bb, t2.abab) * -1.0
    x0 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x0 += einsum("ijab->ijab", t2.bbbb)
    x0 += einsum("ib,ja->ijab", t1.bb, t1.bb) * -1.0
    x0 += einsum("ia,jb->ijab", t1.bb, t1.bb)
    t3_babbab += einsum("jb,kica->ijkabc", t1.aa, x0) * -1.0
    return t3_babbab

def t3_uhf_bbabba(c3=None, t1=None, t2=None, nocc=None, nvir=None):
    t3_bbabba = np.zeros((nocc[1], nocc[1], nocc[0], nvir[1], nvir[1], nvir[0]), dtype=np.float64)
    t3_bbabba += einsum("ikjacb->ijkabc", c3.babbab)
    t3_bbabba += einsum("ia,kjcb->ijkabc", t1.bb, t2.abab) * -1.0
    t3_bbabba += einsum("ib,kjca->ijkabc", t1.bb, t2.abab)
    t3_bbabba += einsum("ja,kicb->ijkabc", t1.bb, t2.abab)
    t3_bbabba += einsum("jb,kica->ijkabc", t1.bb, t2.abab) * -1.0
    x0 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x0 += einsum("ijab->ijab", t2.bbbb)
    x0 += einsum("ib,ja->ijab", t1.bb, t1.bb) * -1.0
    x0 += einsum("ia,jb->ijab", t1.bb, t1.bb)
    t3_bbabba += einsum("kc,jiba->ijkabc", t1.aa, x0) * -1.0
    return t3_bbabba

def t3_uhf_bbbbbb(c3=None, t1=None, t2=None, nocc=None, nvir=None):
    t3_bbbbbb = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=np.float64)
    t3_bbbbbb += einsum("ijkabc->ijkabc", c3.bbbbbb)
    t3_bbbbbb += einsum("ia,jkbc->ijkabc", t1.bb, t2.bbbb) * -1.0
    t3_bbbbbb += einsum("ic,jkab->ijkabc", t1.bb, t2.bbbb) * -1.0
    t3_bbbbbb += einsum("ja,ikbc->ijkabc", t1.bb, t2.bbbb)
    t3_bbbbbb += einsum("jc,ikab->ijkabc", t1.bb, t2.bbbb)
    t3_bbbbbb += einsum("ka,ijbc->ijkabc", t1.bb, t2.bbbb) * -1.0
    t3_bbbbbb += einsum("kc,ijab->ijkabc", t1.bb, t2.bbbb) * -1.0
    x0 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x0 += einsum("ijab->ijab", t2.bbbb)
    x0 += einsum("ib,ja->ijab", t1.bb, t1.bb) * -1.0
    x0 += einsum("ia,jb->ijab", t1.bb, t1.bb)
    t3_bbbbbb += einsum("ib,kjca->ijkabc", t1.bb, x0)
    t3_bbbbbb += einsum("kb,jica->ijkabc", t1.bb, x0)
    t3_bbbbbb += einsum("jb,kica->ijkabc", t1.bb, x0) * -1.0
    return t3_bbbbbb

def t3_rhf(c3=None, t1=None, t2=None, nocc=None, nvir=None):
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

def t4_uhf_aaaaaaaa(c4=None, t1=None, t2=None, t3=None, nocc=None, nvir=None):
    t4_aaaaaaaa = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0], nvir[0]), dtype=np.float64)
    t4_aaaaaaaa += einsum("ijklabcd->ijklabcd", c4.aaaaaaaa)
    t4_aaaaaaaa += einsum("la,ijkbcd->ijklabcd", t1.aa, t3.aaaaaa)
    t4_aaaaaaaa += einsum("lb,ijkacd->ijklabcd", t1.aa, t3.aaaaaa) * -1.0
    t4_aaaaaaaa += einsum("lc,ijkabd->ijklabcd", t1.aa, t3.aaaaaa)
    t4_aaaaaaaa += einsum("ld,ijkabc->ijklabcd", t1.aa, t3.aaaaaa) * -1.0
    t4_aaaaaaaa += einsum("ilcd,jkab->ijklabcd", t2.aaaa, t2.aaaa) * -1.0
    t4_aaaaaaaa += einsum("ilbd,jkac->ijklabcd", t2.aaaa, t2.aaaa)
    t4_aaaaaaaa += einsum("ilad,jkbc->ijklabcd", t2.aaaa, t2.aaaa) * -1.0
    t4_aaaaaaaa += einsum("ilbc,jkad->ijklabcd", t2.aaaa, t2.aaaa) * -1.0
    t4_aaaaaaaa += einsum("ilac,jkbd->ijklabcd", t2.aaaa, t2.aaaa)
    t4_aaaaaaaa += einsum("ilab,jkcd->ijklabcd", t2.aaaa, t2.aaaa) * -1.0
    t4_aaaaaaaa += einsum("ikcd,jlab->ijklabcd", t2.aaaa, t2.aaaa)
    t4_aaaaaaaa += einsum("ikbd,jlac->ijklabcd", t2.aaaa, t2.aaaa) * -1.0
    t4_aaaaaaaa += einsum("ikad,jlbc->ijklabcd", t2.aaaa, t2.aaaa)
    t4_aaaaaaaa += einsum("ikbc,jlad->ijklabcd", t2.aaaa, t2.aaaa)
    t4_aaaaaaaa += einsum("ikac,jlbd->ijklabcd", t2.aaaa, t2.aaaa) * -1.0
    t4_aaaaaaaa += einsum("ikab,jlcd->ijklabcd", t2.aaaa, t2.aaaa)
    t4_aaaaaaaa += einsum("ijcd,klab->ijklabcd", t2.aaaa, t2.aaaa) * -1.0
    t4_aaaaaaaa += einsum("ijbd,klac->ijklabcd", t2.aaaa, t2.aaaa)
    t4_aaaaaaaa += einsum("ijad,klbc->ijklabcd", t2.aaaa, t2.aaaa) * -1.0
    t4_aaaaaaaa += einsum("ijbc,klad->ijklabcd", t2.aaaa, t2.aaaa) * -1.0
    t4_aaaaaaaa += einsum("ijac,klbd->ijklabcd", t2.aaaa, t2.aaaa)
    t4_aaaaaaaa += einsum("ijab,klcd->ijklabcd", t2.aaaa, t2.aaaa) * -1.0
    x0 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x0 += einsum("ijab->ijab", t2.aaaa)
    x0 += einsum("ib,ja->ijab", t1.aa, t1.aa) * -1.0
    x0 += einsum("ia,jb->ijab", t1.aa, t1.aa)
    x1 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=np.float64)
    x1 += einsum("ijkabc->ijkabc", t3.aaaaaa)
    x1 += einsum("ja,ikbc->ijkabc", t1.aa, t2.aaaa) * -1.0
    x1 += einsum("jb,ikac->ijkabc", t1.aa, t2.aaaa)
    x1 += einsum("jc,ikab->ijkabc", t1.aa, t2.aaaa) * -1.0
    x1 += einsum("ka,ijbc->ijkabc", t1.aa, t2.aaaa)
    x1 += einsum("kb,ijac->ijkabc", t1.aa, t2.aaaa) * -1.0
    x1 += einsum("kc,ijab->ijkabc", t1.aa, t2.aaaa)
    x1 += einsum("ia,kjcb->ijkabc", t1.aa, x0)
    x1 += einsum("ic,kjba->ijkabc", t1.aa, x0)
    x1 += einsum("ib,kjca->ijkabc", t1.aa, x0) * -1.0
    t4_aaaaaaaa += einsum("ib,ljkdac->ijklabcd", t1.aa, x1)
    t4_aaaaaaaa += einsum("id,ljkcab->ijklabcd", t1.aa, x1)
    t4_aaaaaaaa += einsum("ia,ljkdbc->ijklabcd", t1.aa, x1) * -1.0
    t4_aaaaaaaa += einsum("ic,ljkdab->ijklabcd", t1.aa, x1) * -1.0
    x2 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=np.float64)
    x2 += einsum("ijkabc->ijkabc", t3.aaaaaa)
    x2 += einsum("ia,jkbc->ijkabc", t1.aa, t2.aaaa)
    x2 += einsum("ib,jkac->ijkabc", t1.aa, t2.aaaa) * -1.0
    x2 += einsum("ic,jkab->ijkabc", t1.aa, t2.aaaa)
    x2 += einsum("ka,ijbc->ijkabc", t1.aa, t2.aaaa)
    x2 += einsum("kb,ijac->ijkabc", t1.aa, t2.aaaa) * -1.0
    x2 += einsum("kc,ijab->ijkabc", t1.aa, t2.aaaa)
    t4_aaaaaaaa += einsum("ja,likdbc->ijklabcd", t1.aa, x2)
    t4_aaaaaaaa += einsum("jc,likdab->ijklabcd", t1.aa, x2)
    t4_aaaaaaaa += einsum("jb,likdac->ijklabcd", t1.aa, x2) * -1.0
    t4_aaaaaaaa += einsum("jd,likcab->ijklabcd", t1.aa, x2) * -1.0
    x3 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=np.float64)
    x3 += einsum("ijkabc->ijkabc", t3.aaaaaa)
    x3 += einsum("ia,jkbc->ijkabc", t1.aa, t2.aaaa)
    x3 += einsum("ib,jkac->ijkabc", t1.aa, t2.aaaa) * -1.0
    x3 += einsum("ic,jkab->ijkabc", t1.aa, t2.aaaa)
    t4_aaaaaaaa += einsum("kb,lijdac->ijklabcd", t1.aa, x3)
    t4_aaaaaaaa += einsum("kd,lijcab->ijklabcd", t1.aa, x3)
    t4_aaaaaaaa += einsum("ka,lijdbc->ijklabcd", t1.aa, x3) * -1.0
    t4_aaaaaaaa += einsum("kc,lijdab->ijklabcd", t1.aa, x3) * -1.0
    return t4_aaaaaaaa

def t4_uhf_aaabaaab(c4=None, t1=None, t2=None, t3=None, nocc=None, nvir=None):
    t4_aaabaaab = np.zeros((nocc[0], nocc[0], nocc[0], nocc[1], nvir[0], nvir[0], nvir[0], nvir[1]), dtype=np.float64)
    t4_aaabaaab += einsum("ijklabcd->ijklabcd", c4.aaabaaab)
    t4_aaabaaab += einsum("ia,jlkbdc->ijklabcd", t1.aa, t3.abaaba) * -1.0
    t4_aaabaaab += einsum("ic,jlkadb->ijklabcd", t1.aa, t3.abaaba) * -1.0
    t4_aaabaaab += einsum("ja,ilkbdc->ijklabcd", t1.aa, t3.abaaba)
    t4_aaabaaab += einsum("jc,ilkadb->ijklabcd", t1.aa, t3.abaaba)
    t4_aaabaaab += einsum("ka,iljbdc->ijklabcd", t1.aa, t3.abaaba) * -1.0
    t4_aaabaaab += einsum("kc,iljadb->ijklabcd", t1.aa, t3.abaaba) * -1.0
    t4_aaabaaab += einsum("klcd,ijab->ijklabcd", t2.abab, t2.aaaa) * -1.0
    t4_aaabaaab += einsum("klad,ijbc->ijklabcd", t2.abab, t2.aaaa) * -1.0
    t4_aaabaaab += einsum("jlcd,ikab->ijklabcd", t2.abab, t2.aaaa)
    t4_aaabaaab += einsum("jlad,ikbc->ijklabcd", t2.abab, t2.aaaa)
    t4_aaabaaab += einsum("ilcd,jkab->ijklabcd", t2.abab, t2.aaaa) * -1.0
    t4_aaabaaab += einsum("ilad,jkbc->ijklabcd", t2.abab, t2.aaaa) * -1.0
    x0 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x0 += einsum("ijab->ijab", t2.aaaa)
    x0 += einsum("ib,ja->ijab", t1.aa, t1.aa) * -1.0
    x0 += einsum("ia,jb->ijab", t1.aa, t1.aa)
    t4_aaabaaab += einsum("ilbd,kjca->ijklabcd", t2.abab, x0)
    t4_aaabaaab += einsum("klbd,jica->ijklabcd", t2.abab, x0)
    t4_aaabaaab += einsum("jlbd,kica->ijklabcd", t2.abab, x0) * -1.0
    x1 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=np.float64)
    x1 += einsum("ijkabc->ijkabc", t3.aaaaaa)
    x1 += einsum("ja,ikbc->ijkabc", t1.aa, t2.aaaa) * -1.0
    x1 += einsum("jb,ikac->ijkabc", t1.aa, t2.aaaa)
    x1 += einsum("jc,ikab->ijkabc", t1.aa, t2.aaaa) * -1.0
    x1 += einsum("ka,ijbc->ijkabc", t1.aa, t2.aaaa)
    x1 += einsum("kb,ijac->ijkabc", t1.aa, t2.aaaa) * -1.0
    x1 += einsum("kc,ijab->ijkabc", t1.aa, t2.aaaa)
    x1 += einsum("ia,kjcb->ijkabc", t1.aa, x0)
    x1 += einsum("ic,kjba->ijkabc", t1.aa, x0)
    x1 += einsum("ib,kjca->ijkabc", t1.aa, x0) * -1.0
    t4_aaabaaab += einsum("ld,kijcab->ijklabcd", t1.bb, x1) * -1.0
    x2 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x2 += einsum("jikbac->iajkbc", t3.abaaba)
    x2 += einsum("jb,kica->iajkbc", t1.aa, t2.abab)
    x2 += einsum("jc,kiba->iajkbc", t1.aa, t2.abab) * -1.0
    x2 += einsum("kb,jica->iajkbc", t1.aa, t2.abab) * -1.0
    x2 += einsum("kc,jiba->iajkbc", t1.aa, t2.abab)
    t4_aaabaaab += einsum("ib,ldkjca->ijklabcd", t1.aa, x2)
    t4_aaabaaab += einsum("kb,ldjica->ijklabcd", t1.aa, x2)
    t4_aaabaaab += einsum("jb,ldkica->ijklabcd", t1.aa, x2) * -1.0
    return t4_aaabaaab

def t4_uhf_aabaaaba(c4=None, t1=None, t2=None, t3=None, nocc=None, nvir=None):
    t4_aabaaaba = np.zeros((nocc[0], nocc[0], nocc[1], nocc[0], nvir[0], nvir[0], nvir[1], nvir[0]), dtype=np.float64)
    t4_aabaaaba += einsum("ijlkabdc->ijklabcd", c4.aaabaaab)
    t4_aabaaaba += einsum("ia,jklbcd->ijklabcd", t1.aa, t3.abaaba) * -1.0
    t4_aabaaaba += einsum("id,jklacb->ijklabcd", t1.aa, t3.abaaba) * -1.0
    t4_aabaaaba += einsum("ja,iklbcd->ijklabcd", t1.aa, t3.abaaba)
    t4_aabaaaba += einsum("jd,iklacb->ijklabcd", t1.aa, t3.abaaba)
    t4_aabaaaba += einsum("la,ikjbcd->ijklabcd", t1.aa, t3.abaaba) * -1.0
    t4_aabaaaba += einsum("ld,ikjacb->ijklabcd", t1.aa, t3.abaaba) * -1.0
    t4_aabaaaba += einsum("lkdc,ijab->ijklabcd", t2.abab, t2.aaaa) * -1.0
    t4_aabaaaba += einsum("lkac,ijbd->ijklabcd", t2.abab, t2.aaaa) * -1.0
    t4_aabaaaba += einsum("jkdc,ilab->ijklabcd", t2.abab, t2.aaaa)
    t4_aabaaaba += einsum("jkac,ilbd->ijklabcd", t2.abab, t2.aaaa)
    t4_aabaaaba += einsum("ikdc,jlab->ijklabcd", t2.abab, t2.aaaa) * -1.0
    t4_aabaaaba += einsum("ikac,jlbd->ijklabcd", t2.abab, t2.aaaa) * -1.0
    x0 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x0 += einsum("ijab->ijab", t2.aaaa)
    x0 += einsum("ib,ja->ijab", t1.aa, t1.aa) * -1.0
    x0 += einsum("ia,jb->ijab", t1.aa, t1.aa)
    t4_aabaaaba += einsum("ikbc,ljda->ijklabcd", t2.abab, x0)
    t4_aabaaaba += einsum("lkbc,jida->ijklabcd", t2.abab, x0)
    t4_aabaaaba += einsum("jkbc,lida->ijklabcd", t2.abab, x0) * -1.0
    x1 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=np.float64)
    x1 += einsum("ijkabc->ijkabc", t3.aaaaaa)
    x1 += einsum("ja,ikbc->ijkabc", t1.aa, t2.aaaa) * -1.0
    x1 += einsum("jb,ikac->ijkabc", t1.aa, t2.aaaa)
    x1 += einsum("jc,ikab->ijkabc", t1.aa, t2.aaaa) * -1.0
    x1 += einsum("ka,ijbc->ijkabc", t1.aa, t2.aaaa)
    x1 += einsum("kb,ijac->ijkabc", t1.aa, t2.aaaa) * -1.0
    x1 += einsum("kc,ijab->ijkabc", t1.aa, t2.aaaa)
    x1 += einsum("ia,kjcb->ijkabc", t1.aa, x0)
    x1 += einsum("ic,kjba->ijkabc", t1.aa, x0)
    x1 += einsum("ib,kjca->ijkabc", t1.aa, x0) * -1.0
    t4_aabaaaba += einsum("kc,lijdab->ijklabcd", t1.bb, x1) * -1.0
    x2 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x2 += einsum("jikbac->iajkbc", t3.abaaba)
    x2 += einsum("jb,kica->iajkbc", t1.aa, t2.abab)
    x2 += einsum("jc,kiba->iajkbc", t1.aa, t2.abab) * -1.0
    x2 += einsum("kb,jica->iajkbc", t1.aa, t2.abab) * -1.0
    x2 += einsum("kc,jiba->iajkbc", t1.aa, t2.abab)
    t4_aabaaaba += einsum("ib,kcljda->ijklabcd", t1.aa, x2)
    t4_aabaaaba += einsum("lb,kcjida->ijklabcd", t1.aa, x2)
    t4_aabaaaba += einsum("jb,kclida->ijklabcd", t1.aa, x2) * -1.0
    return t4_aabaaaba

def t4_uhf_aabbaabb(c4=None, t1=None, t2=None, t3=None, nocc=None, nvir=None):
    t4_aabbaabb = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1], nvir[0], nvir[0], nvir[1], nvir[1]), dtype=np.float64)
    t4_aabbaabb += einsum("ikjlacbd->ijklabcd", c4.abababab)
    t4_aabbaabb += einsum("ia,kjlcbd->ijklabcd", t1.aa, t3.babbab) * -1.0
    t4_aabbaabb += einsum("ib,kjlcad->ijklabcd", t1.aa, t3.babbab)
    t4_aabbaabb += einsum("ja,kilcbd->ijklabcd", t1.aa, t3.babbab)
    t4_aabbaabb += einsum("jb,kilcad->ijklabcd", t1.aa, t3.babbab) * -1.0
    t4_aabbaabb += einsum("ilac,jkbd->ijklabcd", t2.abab, t2.abab)
    t4_aabbaabb += einsum("ilbc,jkad->ijklabcd", t2.abab, t2.abab) * -1.0
    t4_aabbaabb += einsum("ikbd,jlac->ijklabcd", t2.abab, t2.abab) * -1.0
    t4_aabbaabb += einsum("ikad,jlbc->ijklabcd", t2.abab, t2.abab)
    t4_aabbaabb += einsum("ilad,jkbc->ijklabcd", t2.abab, t2.abab) * -1.0
    t4_aabbaabb += einsum("ilbd,jkac->ijklabcd", t2.abab, t2.abab)
    t4_aabbaabb += einsum("ikbc,jlad->ijklabcd", t2.abab, t2.abab)
    t4_aabbaabb += einsum("ikac,jlbd->ijklabcd", t2.abab, t2.abab) * -1.0
    x0 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x0 += einsum("ijab->ijab", t2.aaaa)
    x0 += einsum("ib,ja->ijab", t1.aa, t1.aa) * -1.0
    x0 += einsum("ia,jb->ijab", t1.aa, t1.aa)
    x1 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1 += einsum("ijab->ijab", t2.bbbb)
    x1 += einsum("ib,ja->ijab", t1.bb, t1.bb) * -1.0
    x1 += einsum("ia,jb->ijab", t1.bb, t1.bb)
    t4_aabbaabb += einsum("jiba,lkdc->ijklabcd", x0, x1) * -1.0
    x2 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x2 += einsum("jikbac->iajkbc", t3.abaaba)
    x2 += einsum("jb,kica->iajkbc", t1.aa, t2.abab)
    x2 += einsum("jc,kiba->iajkbc", t1.aa, t2.abab) * -1.0
    x2 += einsum("kb,jica->iajkbc", t1.aa, t2.abab) * -1.0
    x2 += einsum("kc,jiba->iajkbc", t1.aa, t2.abab)
    t4_aabbaabb += einsum("kd,lcjiba->ijklabcd", t1.bb, x2)
    t4_aabbaabb += einsum("lc,kdjiba->ijklabcd", t1.bb, x2)
    t4_aabbaabb += einsum("kc,ldjiba->ijklabcd", t1.bb, x2) * -1.0
    t4_aabbaabb += einsum("ld,kcjiba->ijklabcd", t1.bb, x2) * -1.0
    return t4_aabbaabb

def t4_uhf_abaaabaa(c4=None, t1=None, t2=None, t3=None, nocc=None, nvir=None):
    t4_abaaabaa = np.zeros((nocc[0], nocc[1], nocc[0], nocc[0], nvir[0], nvir[1], nvir[0], nvir[0]), dtype=np.float64)
    t4_abaaabaa += einsum("ikljacdb->ijklabcd", c4.aaabaaab)
    t4_abaaabaa += einsum("ia,kjlcbd->ijklabcd", t1.aa, t3.abaaba) * -1.0
    t4_abaaabaa += einsum("id,kjlabc->ijklabcd", t1.aa, t3.abaaba) * -1.0
    t4_abaaabaa += einsum("ka,ijlcbd->ijklabcd", t1.aa, t3.abaaba)
    t4_abaaabaa += einsum("kd,ijlabc->ijklabcd", t1.aa, t3.abaaba)
    t4_abaaabaa += einsum("la,ijkcbd->ijklabcd", t1.aa, t3.abaaba) * -1.0
    t4_abaaabaa += einsum("ld,ijkabc->ijklabcd", t1.aa, t3.abaaba) * -1.0
    t4_abaaabaa += einsum("ljdb,ikac->ijklabcd", t2.abab, t2.aaaa) * -1.0
    t4_abaaabaa += einsum("ljab,ikcd->ijklabcd", t2.abab, t2.aaaa) * -1.0
    t4_abaaabaa += einsum("kjdb,ilac->ijklabcd", t2.abab, t2.aaaa)
    t4_abaaabaa += einsum("kjab,ilcd->ijklabcd", t2.abab, t2.aaaa)
    t4_abaaabaa += einsum("ijdb,klac->ijklabcd", t2.abab, t2.aaaa) * -1.0
    t4_abaaabaa += einsum("ijab,klcd->ijklabcd", t2.abab, t2.aaaa) * -1.0
    x0 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x0 += einsum("ijab->ijab", t2.aaaa)
    x0 += einsum("ib,ja->ijab", t1.aa, t1.aa) * -1.0
    x0 += einsum("ia,jb->ijab", t1.aa, t1.aa)
    t4_abaaabaa += einsum("ijcb,lkda->ijklabcd", t2.abab, x0)
    t4_abaaabaa += einsum("ljcb,kida->ijklabcd", t2.abab, x0)
    t4_abaaabaa += einsum("kjcb,lida->ijklabcd", t2.abab, x0) * -1.0
    x1 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=np.float64)
    x1 += einsum("ijkabc->ijkabc", t3.aaaaaa)
    x1 += einsum("ja,ikbc->ijkabc", t1.aa, t2.aaaa) * -1.0
    x1 += einsum("jb,ikac->ijkabc", t1.aa, t2.aaaa)
    x1 += einsum("jc,ikab->ijkabc", t1.aa, t2.aaaa) * -1.0
    x1 += einsum("ka,ijbc->ijkabc", t1.aa, t2.aaaa)
    x1 += einsum("kb,ijac->ijkabc", t1.aa, t2.aaaa) * -1.0
    x1 += einsum("kc,ijab->ijkabc", t1.aa, t2.aaaa)
    x1 += einsum("ia,kjcb->ijkabc", t1.aa, x0)
    x1 += einsum("ic,kjba->ijkabc", t1.aa, x0)
    x1 += einsum("ib,kjca->ijkabc", t1.aa, x0) * -1.0
    t4_abaaabaa += einsum("jb,likdac->ijklabcd", t1.bb, x1) * -1.0
    x2 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x2 += einsum("jikbac->iajkbc", t3.abaaba)
    x2 += einsum("jb,kica->iajkbc", t1.aa, t2.abab)
    x2 += einsum("jc,kiba->iajkbc", t1.aa, t2.abab) * -1.0
    x2 += einsum("kb,jica->iajkbc", t1.aa, t2.abab) * -1.0
    x2 += einsum("kc,jiba->iajkbc", t1.aa, t2.abab)
    t4_abaaabaa += einsum("ic,jblkda->ijklabcd", t1.aa, x2)
    t4_abaaabaa += einsum("lc,jbkida->ijklabcd", t1.aa, x2)
    t4_abaaabaa += einsum("kc,jblida->ijklabcd", t1.aa, x2) * -1.0
    return t4_abaaabaa

def t4_uhf_abababab(c4=None, t1=None, t2=None, t3=None, nocc=None, nvir=None):
    t4_abababab = np.zeros((nocc[0], nocc[1], nocc[0], nocc[1], nvir[0], nvir[1], nvir[0], nvir[1]), dtype=np.float64)
    t4_abababab += einsum("ijklabcd->ijklabcd", c4.abababab)
    t4_abababab += einsum("ia,jklbcd->ijklabcd", t1.aa, t3.babbab) * -1.0
    t4_abababab += einsum("ic,jklbad->ijklabcd", t1.aa, t3.babbab)
    t4_abababab += einsum("ka,jilbcd->ijklabcd", t1.aa, t3.babbab)
    t4_abababab += einsum("kc,jilbad->ijklabcd", t1.aa, t3.babbab) * -1.0
    t4_abababab += einsum("ilab,kjcd->ijklabcd", t2.abab, t2.abab)
    t4_abababab += einsum("ilcb,kjad->ijklabcd", t2.abab, t2.abab) * -1.0
    t4_abababab += einsum("ijcd,klab->ijklabcd", t2.abab, t2.abab) * -1.0
    t4_abababab += einsum("ijad,klcb->ijklabcd", t2.abab, t2.abab)
    t4_abababab += einsum("ilad,kjcb->ijklabcd", t2.abab, t2.abab) * -1.0
    t4_abababab += einsum("ilcd,kjab->ijklabcd", t2.abab, t2.abab)
    t4_abababab += einsum("ijcb,klad->ijklabcd", t2.abab, t2.abab)
    t4_abababab += einsum("ijab,klcd->ijklabcd", t2.abab, t2.abab) * -1.0
    x0 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x0 += einsum("ijab->ijab", t2.aaaa)
    x0 += einsum("ib,ja->ijab", t1.aa, t1.aa) * -1.0
    x0 += einsum("ia,jb->ijab", t1.aa, t1.aa)
    x1 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1 += einsum("ijab->ijab", t2.bbbb)
    x1 += einsum("ib,ja->ijab", t1.bb, t1.bb) * -1.0
    x1 += einsum("ia,jb->ijab", t1.bb, t1.bb)
    t4_abababab += einsum("kica,ljdb->ijklabcd", x0, x1) * -1.0
    x2 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x2 += einsum("jikbac->iajkbc", t3.abaaba)
    x2 += einsum("jb,kica->iajkbc", t1.aa, t2.abab)
    x2 += einsum("jc,kiba->iajkbc", t1.aa, t2.abab) * -1.0
    x2 += einsum("kb,jica->iajkbc", t1.aa, t2.abab) * -1.0
    x2 += einsum("kc,jiba->iajkbc", t1.aa, t2.abab)
    t4_abababab += einsum("jd,lbkica->ijklabcd", t1.bb, x2)
    t4_abababab += einsum("lb,jdkica->ijklabcd", t1.bb, x2)
    t4_abababab += einsum("jb,ldkica->ijklabcd", t1.bb, x2) * -1.0
    t4_abababab += einsum("ld,jbkica->ijklabcd", t1.bb, x2) * -1.0
    return t4_abababab

def t4_uhf_abbaabba(c4=None, t1=None, t2=None, t3=None, nocc=None, nvir=None):
    t4_abbaabba = np.zeros((nocc[0], nocc[1], nocc[1], nocc[0], nvir[0], nvir[1], nvir[1], nvir[0]), dtype=np.float64)
    t4_abbaabba += einsum("ijlkabdc->ijklabcd", c4.abababab)
    t4_abbaabba += einsum("ia,jlkbdc->ijklabcd", t1.aa, t3.babbab) * -1.0
    t4_abbaabba += einsum("id,jlkbac->ijklabcd", t1.aa, t3.babbab)
    t4_abbaabba += einsum("la,jikbdc->ijklabcd", t1.aa, t3.babbab)
    t4_abbaabba += einsum("ld,jikbac->ijklabcd", t1.aa, t3.babbab) * -1.0
    t4_abbaabba += einsum("ikab,ljdc->ijklabcd", t2.abab, t2.abab)
    t4_abbaabba += einsum("ikdb,ljac->ijklabcd", t2.abab, t2.abab) * -1.0
    t4_abbaabba += einsum("ijdc,lkab->ijklabcd", t2.abab, t2.abab) * -1.0
    t4_abbaabba += einsum("ijac,lkdb->ijklabcd", t2.abab, t2.abab)
    t4_abbaabba += einsum("ikac,ljdb->ijklabcd", t2.abab, t2.abab) * -1.0
    t4_abbaabba += einsum("ikdc,ljab->ijklabcd", t2.abab, t2.abab)
    t4_abbaabba += einsum("ijdb,lkac->ijklabcd", t2.abab, t2.abab)
    t4_abbaabba += einsum("ijab,lkdc->ijklabcd", t2.abab, t2.abab) * -1.0
    x0 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x0 += einsum("ijab->ijab", t2.bbbb)
    x0 += einsum("ib,ja->ijab", t1.bb, t1.bb) * -1.0
    x0 += einsum("ia,jb->ijab", t1.bb, t1.bb)
    x1 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x1 += einsum("ijab->ijab", t2.aaaa)
    x1 += einsum("ib,ja->ijab", t1.aa, t1.aa) * -1.0
    x1 += einsum("ia,jb->ijab", t1.aa, t1.aa)
    t4_abbaabba += einsum("kjcb,lida->ijklabcd", x0, x1) * -1.0
    x2 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x2 += einsum("jikbac->iajkbc", t3.abaaba)
    x2 += einsum("jb,kica->iajkbc", t1.aa, t2.abab)
    x2 += einsum("jc,kiba->iajkbc", t1.aa, t2.abab) * -1.0
    x2 += einsum("kb,jica->iajkbc", t1.aa, t2.abab) * -1.0
    x2 += einsum("kc,jiba->iajkbc", t1.aa, t2.abab)
    t4_abbaabba += einsum("jc,kblida->ijklabcd", t1.bb, x2)
    t4_abbaabba += einsum("kb,jclida->ijklabcd", t1.bb, x2)
    t4_abbaabba += einsum("jb,kclida->ijklabcd", t1.bb, x2) * -1.0
    t4_abbaabba += einsum("kc,jblida->ijklabcd", t1.bb, x2) * -1.0
    return t4_abbaabba

def t4_uhf_abbbabbb(c4=None, t1=None, t2=None, t3=None, nocc=None, nvir=None):
    t4_abbbabbb = np.zeros((nocc[0], nocc[1], nocc[1], nocc[1], nvir[0], nvir[1], nvir[1], nvir[1]), dtype=np.float64)
    t4_abbbabbb += einsum("ijklabcd->ijklabcd", c4.abbbabbb)
    t4_abbbabbb += einsum("jb,kilcad->ijklabcd", t1.bb, t3.babbab) * -1.0
    t4_abbbabbb += einsum("jd,kilbac->ijklabcd", t1.bb, t3.babbab) * -1.0
    t4_abbbabbb += einsum("kb,jilcad->ijklabcd", t1.bb, t3.babbab)
    t4_abbbabbb += einsum("kd,jilbac->ijklabcd", t1.bb, t3.babbab)
    t4_abbbabbb += einsum("lb,jikcad->ijklabcd", t1.bb, t3.babbab) * -1.0
    t4_abbbabbb += einsum("ld,jikbac->ijklabcd", t1.bb, t3.babbab) * -1.0
    t4_abbbabbb += einsum("ilad,jkbc->ijklabcd", t2.abab, t2.bbbb) * -1.0
    t4_abbbabbb += einsum("ilab,jkcd->ijklabcd", t2.abab, t2.bbbb) * -1.0
    t4_abbbabbb += einsum("ikad,jlbc->ijklabcd", t2.abab, t2.bbbb)
    t4_abbbabbb += einsum("ikab,jlcd->ijklabcd", t2.abab, t2.bbbb)
    t4_abbbabbb += einsum("ijad,klbc->ijklabcd", t2.abab, t2.bbbb) * -1.0
    t4_abbbabbb += einsum("ijab,klcd->ijklabcd", t2.abab, t2.bbbb) * -1.0
    x0 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x0 += einsum("ijab->ijab", t2.bbbb)
    x0 += einsum("ib,ja->ijab", t1.bb, t1.bb) * -1.0
    x0 += einsum("ia,jb->ijab", t1.bb, t1.bb)
    t4_abbbabbb += einsum("ilac,kjdb->ijklabcd", t2.abab, x0)
    t4_abbbabbb += einsum("ijac,lkdb->ijklabcd", t2.abab, x0)
    t4_abbbabbb += einsum("ikac,ljdb->ijklabcd", t2.abab, x0) * -1.0
    x1 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=np.float64)
    x1 += einsum("ijkabc->ijkabc", t3.bbbbbb)
    x1 += einsum("ja,ikbc->ijkabc", t1.bb, t2.bbbb) * -1.0
    x1 += einsum("jb,ikac->ijkabc", t1.bb, t2.bbbb)
    x1 += einsum("jc,ikab->ijkabc", t1.bb, t2.bbbb) * -1.0
    x1 += einsum("ka,ijbc->ijkabc", t1.bb, t2.bbbb)
    x1 += einsum("kb,ijac->ijkabc", t1.bb, t2.bbbb) * -1.0
    x1 += einsum("kc,ijab->ijkabc", t1.bb, t2.bbbb)
    x1 += einsum("ia,kjcb->ijkabc", t1.bb, x0)
    x1 += einsum("ic,kjba->ijkabc", t1.bb, x0)
    x1 += einsum("ib,kjca->ijkabc", t1.bb, x0) * -1.0
    t4_abbbabbb += einsum("ia,ljkdbc->ijklabcd", t1.aa, x1) * -1.0
    x2 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=np.float64)
    x2 += einsum("ikjacb->ijabkc", t3.babbab)
    x2 += einsum("ia,kjcb->ijabkc", t1.bb, t2.abab)
    x2 += einsum("ib,kjca->ijabkc", t1.bb, t2.abab) * -1.0
    x2 += einsum("ja,kicb->ijabkc", t1.bb, t2.abab) * -1.0
    x2 += einsum("jb,kica->ijabkc", t1.bb, t2.abab)
    t4_abbbabbb += einsum("jc,lkdbia->ijklabcd", t1.bb, x2)
    t4_abbbabbb += einsum("lc,kjdbia->ijklabcd", t1.bb, x2)
    t4_abbbabbb += einsum("kc,ljdbia->ijklabcd", t1.bb, x2) * -1.0
    return t4_abbbabbb

def t4_uhf_baaabaaa(c4=None, t1=None, t2=None, t3=None, nocc=None, nvir=None):
    t4_baaabaaa = np.zeros((nocc[1], nocc[0], nocc[0], nocc[0], nvir[1], nvir[0], nvir[0], nvir[0]), dtype=np.float64)
    t4_baaabaaa += einsum("jklibcda->ijklabcd", c4.aaabaaab)
    t4_baaabaaa += einsum("jb,kilcad->ijklabcd", t1.aa, t3.abaaba) * -1.0
    t4_baaabaaa += einsum("jd,kilbac->ijklabcd", t1.aa, t3.abaaba) * -1.0
    t4_baaabaaa += einsum("kb,jilcad->ijklabcd", t1.aa, t3.abaaba)
    t4_baaabaaa += einsum("kd,jilbac->ijklabcd", t1.aa, t3.abaaba)
    t4_baaabaaa += einsum("lb,jikcad->ijklabcd", t1.aa, t3.abaaba) * -1.0
    t4_baaabaaa += einsum("ld,jikbac->ijklabcd", t1.aa, t3.abaaba) * -1.0
    t4_baaabaaa += einsum("lida,jkbc->ijklabcd", t2.abab, t2.aaaa) * -1.0
    t4_baaabaaa += einsum("liba,jkcd->ijklabcd", t2.abab, t2.aaaa) * -1.0
    t4_baaabaaa += einsum("kida,jlbc->ijklabcd", t2.abab, t2.aaaa)
    t4_baaabaaa += einsum("kiba,jlcd->ijklabcd", t2.abab, t2.aaaa)
    t4_baaabaaa += einsum("jida,klbc->ijklabcd", t2.abab, t2.aaaa) * -1.0
    t4_baaabaaa += einsum("jiba,klcd->ijklabcd", t2.abab, t2.aaaa) * -1.0
    x0 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x0 += einsum("ijab->ijab", t2.aaaa)
    x0 += einsum("ib,ja->ijab", t1.aa, t1.aa) * -1.0
    x0 += einsum("ia,jb->ijab", t1.aa, t1.aa)
    t4_baaabaaa += einsum("jica,lkdb->ijklabcd", t2.abab, x0)
    t4_baaabaaa += einsum("lica,kjdb->ijklabcd", t2.abab, x0)
    t4_baaabaaa += einsum("kica,ljdb->ijklabcd", t2.abab, x0) * -1.0
    x1 = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=np.float64)
    x1 += einsum("ijkabc->ijkabc", t3.aaaaaa)
    x1 += einsum("ja,ikbc->ijkabc", t1.aa, t2.aaaa) * -1.0
    x1 += einsum("jb,ikac->ijkabc", t1.aa, t2.aaaa)
    x1 += einsum("jc,ikab->ijkabc", t1.aa, t2.aaaa) * -1.0
    x1 += einsum("ka,ijbc->ijkabc", t1.aa, t2.aaaa)
    x1 += einsum("kb,ijac->ijkabc", t1.aa, t2.aaaa) * -1.0
    x1 += einsum("kc,ijab->ijkabc", t1.aa, t2.aaaa)
    x1 += einsum("ia,kjcb->ijkabc", t1.aa, x0)
    x1 += einsum("ic,kjba->ijkabc", t1.aa, x0)
    x1 += einsum("ib,kjca->ijkabc", t1.aa, x0) * -1.0
    t4_baaabaaa += einsum("ia,ljkdbc->ijklabcd", t1.bb, x1) * -1.0
    x2 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x2 += einsum("jikbac->iajkbc", t3.abaaba)
    x2 += einsum("jb,kica->iajkbc", t1.aa, t2.abab)
    x2 += einsum("jc,kiba->iajkbc", t1.aa, t2.abab) * -1.0
    x2 += einsum("kb,jica->iajkbc", t1.aa, t2.abab) * -1.0
    x2 += einsum("kc,jiba->iajkbc", t1.aa, t2.abab)
    t4_baaabaaa += einsum("jc,ialkdb->ijklabcd", t1.aa, x2)
    t4_baaabaaa += einsum("lc,iakjdb->ijklabcd", t1.aa, x2)
    t4_baaabaaa += einsum("kc,ialjdb->ijklabcd", t1.aa, x2) * -1.0
    return t4_baaabaaa

def t4_uhf_baabbaab(c4=None, t1=None, t2=None, t3=None, nocc=None, nvir=None):
    t4_baabbaab = np.zeros((nocc[1], nocc[0], nocc[0], nocc[1], nvir[1], nvir[0], nvir[0], nvir[1]), dtype=np.float64)
    t4_baabbaab += einsum("jiklbacd->ijklabcd", c4.abababab)
    t4_baabbaab += einsum("jb,iklacd->ijklabcd", t1.aa, t3.babbab) * -1.0
    t4_baabbaab += einsum("jc,iklabd->ijklabcd", t1.aa, t3.babbab)
    t4_baabbaab += einsum("kb,ijlacd->ijklabcd", t1.aa, t3.babbab)
    t4_baabbaab += einsum("kc,ijlabd->ijklabcd", t1.aa, t3.babbab) * -1.0
    t4_baabbaab += einsum("jiba,klcd->ijklabcd", t2.abab, t2.abab) * -1.0
    t4_baabbaab += einsum("jica,klbd->ijklabcd", t2.abab, t2.abab)
    t4_baabbaab += einsum("jlcd,kiba->ijklabcd", t2.abab, t2.abab)
    t4_baabbaab += einsum("jlbd,kica->ijklabcd", t2.abab, t2.abab) * -1.0
    t4_baabbaab += einsum("jibd,klca->ijklabcd", t2.abab, t2.abab)
    t4_baabbaab += einsum("jicd,klba->ijklabcd", t2.abab, t2.abab) * -1.0
    t4_baabbaab += einsum("jlca,kibd->ijklabcd", t2.abab, t2.abab) * -1.0
    t4_baabbaab += einsum("jlba,kicd->ijklabcd", t2.abab, t2.abab)
    x0 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x0 += einsum("ijab->ijab", t2.aaaa)
    x0 += einsum("ib,ja->ijab", t1.aa, t1.aa) * -1.0
    x0 += einsum("ia,jb->ijab", t1.aa, t1.aa)
    x1 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x1 += einsum("ijab->ijab", t2.bbbb)
    x1 += einsum("ib,ja->ijab", t1.bb, t1.bb) * -1.0
    x1 += einsum("ia,jb->ijab", t1.bb, t1.bb)
    t4_baabbaab += einsum("kjcb,lida->ijklabcd", x0, x1) * -1.0
    x2 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x2 += einsum("jikbac->iajkbc", t3.abaaba)
    x2 += einsum("jb,kica->iajkbc", t1.aa, t2.abab)
    x2 += einsum("jc,kiba->iajkbc", t1.aa, t2.abab) * -1.0
    x2 += einsum("kb,jica->iajkbc", t1.aa, t2.abab) * -1.0
    x2 += einsum("kc,jiba->iajkbc", t1.aa, t2.abab)
    t4_baabbaab += einsum("id,lakjcb->ijklabcd", t1.bb, x2)
    t4_baabbaab += einsum("la,idkjcb->ijklabcd", t1.bb, x2)
    t4_baabbaab += einsum("ia,ldkjcb->ijklabcd", t1.bb, x2) * -1.0
    t4_baabbaab += einsum("ld,iakjcb->ijklabcd", t1.bb, x2) * -1.0
    return t4_baabbaab

def t4_uhf_babababa(c4=None, t1=None, t2=None, t3=None, nocc=None, nvir=None):
    t4_babababa = np.zeros((nocc[1], nocc[0], nocc[1], nocc[0], nvir[1], nvir[0], nvir[1], nvir[0]), dtype=np.float64)
    t4_babababa += einsum("jilkbadc->ijklabcd", c4.abababab)
    t4_babababa += einsum("jb,ilkadc->ijklabcd", t1.aa, t3.babbab) * -1.0
    t4_babababa += einsum("jd,ilkabc->ijklabcd", t1.aa, t3.babbab)
    t4_babababa += einsum("lb,ijkadc->ijklabcd", t1.aa, t3.babbab)
    t4_babababa += einsum("ld,ijkabc->ijklabcd", t1.aa, t3.babbab) * -1.0
    t4_babababa += einsum("jiba,lkdc->ijklabcd", t2.abab, t2.abab) * -1.0
    t4_babababa += einsum("jida,lkbc->ijklabcd", t2.abab, t2.abab)
    t4_babababa += einsum("jkdc,liba->ijklabcd", t2.abab, t2.abab)
    t4_babababa += einsum("jkbc,lida->ijklabcd", t2.abab, t2.abab) * -1.0
    t4_babababa += einsum("jibc,lkda->ijklabcd", t2.abab, t2.abab)
    t4_babababa += einsum("jidc,lkba->ijklabcd", t2.abab, t2.abab) * -1.0
    t4_babababa += einsum("jkda,libc->ijklabcd", t2.abab, t2.abab) * -1.0
    t4_babababa += einsum("jkba,lidc->ijklabcd", t2.abab, t2.abab)
    x0 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x0 += einsum("ijab->ijab", t2.bbbb)
    x0 += einsum("ib,ja->ijab", t1.bb, t1.bb) * -1.0
    x0 += einsum("ia,jb->ijab", t1.bb, t1.bb)
    x1 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x1 += einsum("ijab->ijab", t2.aaaa)
    x1 += einsum("ib,ja->ijab", t1.aa, t1.aa) * -1.0
    x1 += einsum("ia,jb->ijab", t1.aa, t1.aa)
    t4_babababa += einsum("kica,ljdb->ijklabcd", x0, x1) * -1.0
    x2 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x2 += einsum("jikbac->iajkbc", t3.abaaba)
    x2 += einsum("jb,kica->iajkbc", t1.aa, t2.abab)
    x2 += einsum("jc,kiba->iajkbc", t1.aa, t2.abab) * -1.0
    x2 += einsum("kb,jica->iajkbc", t1.aa, t2.abab) * -1.0
    x2 += einsum("kc,jiba->iajkbc", t1.aa, t2.abab)
    t4_babababa += einsum("ic,kaljdb->ijklabcd", t1.bb, x2)
    t4_babababa += einsum("ka,icljdb->ijklabcd", t1.bb, x2)
    t4_babababa += einsum("ia,kcljdb->ijklabcd", t1.bb, x2) * -1.0
    t4_babababa += einsum("kc,ialjdb->ijklabcd", t1.bb, x2) * -1.0
    return t4_babababa

def t4_uhf_babbbabb(c4=None, t1=None, t2=None, t3=None, nocc=None, nvir=None):
    t4_babbbabb = np.zeros((nocc[1], nocc[0], nocc[1], nocc[1], nvir[1], nvir[0], nvir[1], nvir[1]), dtype=np.float64)
    t4_babbbabb += einsum("jiklbacd->ijklabcd", c4.abbbabbb)
    t4_babbbabb += einsum("ia,kjlcbd->ijklabcd", t1.bb, t3.babbab) * -1.0
    t4_babbbabb += einsum("id,kjlabc->ijklabcd", t1.bb, t3.babbab) * -1.0
    t4_babbbabb += einsum("ka,ijlcbd->ijklabcd", t1.bb, t3.babbab)
    t4_babbbabb += einsum("kd,ijlabc->ijklabcd", t1.bb, t3.babbab)
    t4_babbbabb += einsum("la,ijkcbd->ijklabcd", t1.bb, t3.babbab) * -1.0
    t4_babbbabb += einsum("ld,ijkabc->ijklabcd", t1.bb, t3.babbab) * -1.0
    t4_babbbabb += einsum("jlbd,ikac->ijklabcd", t2.abab, t2.bbbb) * -1.0
    t4_babbbabb += einsum("jlba,ikcd->ijklabcd", t2.abab, t2.bbbb) * -1.0
    t4_babbbabb += einsum("jkbd,ilac->ijklabcd", t2.abab, t2.bbbb)
    t4_babbbabb += einsum("jkba,ilcd->ijklabcd", t2.abab, t2.bbbb)
    t4_babbbabb += einsum("jibd,klac->ijklabcd", t2.abab, t2.bbbb) * -1.0
    t4_babbbabb += einsum("jiba,klcd->ijklabcd", t2.abab, t2.bbbb) * -1.0
    x0 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x0 += einsum("ijab->ijab", t2.bbbb)
    x0 += einsum("ib,ja->ijab", t1.bb, t1.bb) * -1.0
    x0 += einsum("ia,jb->ijab", t1.bb, t1.bb)
    t4_babbbabb += einsum("jlbc,kida->ijklabcd", t2.abab, x0)
    t4_babbbabb += einsum("jibc,lkda->ijklabcd", t2.abab, x0)
    t4_babbbabb += einsum("jkbc,lida->ijklabcd", t2.abab, x0) * -1.0
    x1 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=np.float64)
    x1 += einsum("ijkabc->ijkabc", t3.bbbbbb)
    x1 += einsum("ja,ikbc->ijkabc", t1.bb, t2.bbbb) * -1.0
    x1 += einsum("jb,ikac->ijkabc", t1.bb, t2.bbbb)
    x1 += einsum("jc,ikab->ijkabc", t1.bb, t2.bbbb) * -1.0
    x1 += einsum("ka,ijbc->ijkabc", t1.bb, t2.bbbb)
    x1 += einsum("kb,ijac->ijkabc", t1.bb, t2.bbbb) * -1.0
    x1 += einsum("kc,ijab->ijkabc", t1.bb, t2.bbbb)
    x1 += einsum("ia,kjcb->ijkabc", t1.bb, x0)
    x1 += einsum("ic,kjba->ijkabc", t1.bb, x0)
    x1 += einsum("ib,kjca->ijkabc", t1.bb, x0) * -1.0
    t4_babbbabb += einsum("jb,likdac->ijklabcd", t1.aa, x1) * -1.0
    x2 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=np.float64)
    x2 += einsum("ikjacb->ijabkc", t3.babbab)
    x2 += einsum("ia,kjcb->ijabkc", t1.bb, t2.abab)
    x2 += einsum("ib,kjca->ijabkc", t1.bb, t2.abab) * -1.0
    x2 += einsum("ja,kicb->ijabkc", t1.bb, t2.abab) * -1.0
    x2 += einsum("jb,kica->ijabkc", t1.bb, t2.abab)
    t4_babbbabb += einsum("ic,lkdajb->ijklabcd", t1.bb, x2)
    t4_babbbabb += einsum("lc,kidajb->ijklabcd", t1.bb, x2)
    t4_babbbabb += einsum("kc,lidajb->ijklabcd", t1.bb, x2) * -1.0
    return t4_babbbabb

def t4_uhf_bbaabbaa(c4=None, t1=None, t2=None, t3=None, nocc=None, nvir=None):
    t4_bbaabbaa = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0], nvir[1], nvir[1], nvir[0], nvir[0]), dtype=np.float64)
    t4_bbaabbaa += einsum("kiljcadb->ijklabcd", c4.abababab)
    t4_bbaabbaa += einsum("kc,iljadb->ijklabcd", t1.aa, t3.babbab) * -1.0
    t4_bbaabbaa += einsum("kd,iljacb->ijklabcd", t1.aa, t3.babbab)
    t4_bbaabbaa += einsum("lc,ikjadb->ijklabcd", t1.aa, t3.babbab)
    t4_bbaabbaa += einsum("ld,ikjacb->ijklabcd", t1.aa, t3.babbab) * -1.0
    t4_bbaabbaa += einsum("kica,ljdb->ijklabcd", t2.abab, t2.abab) * -1.0
    t4_bbaabbaa += einsum("kida,ljcb->ijklabcd", t2.abab, t2.abab)
    t4_bbaabbaa += einsum("kjdb,lica->ijklabcd", t2.abab, t2.abab)
    t4_bbaabbaa += einsum("kjcb,lida->ijklabcd", t2.abab, t2.abab) * -1.0
    t4_bbaabbaa += einsum("kicb,ljda->ijklabcd", t2.abab, t2.abab)
    t4_bbaabbaa += einsum("kidb,ljca->ijklabcd", t2.abab, t2.abab) * -1.0
    t4_bbaabbaa += einsum("kjda,licb->ijklabcd", t2.abab, t2.abab) * -1.0
    t4_bbaabbaa += einsum("kjca,lidb->ijklabcd", t2.abab, t2.abab)
    x0 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x0 += einsum("ijab->ijab", t2.bbbb)
    x0 += einsum("ib,ja->ijab", t1.bb, t1.bb) * -1.0
    x0 += einsum("ia,jb->ijab", t1.bb, t1.bb)
    x1 = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x1 += einsum("ijab->ijab", t2.aaaa)
    x1 += einsum("ib,ja->ijab", t1.aa, t1.aa) * -1.0
    x1 += einsum("ia,jb->ijab", t1.aa, t1.aa)
    t4_bbaabbaa += einsum("jiba,lkdc->ijklabcd", x0, x1) * -1.0
    x2 = np.zeros((nocc[1], nvir[1], nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    x2 += einsum("jikbac->iajkbc", t3.abaaba)
    x2 += einsum("jb,kica->iajkbc", t1.aa, t2.abab)
    x2 += einsum("jc,kiba->iajkbc", t1.aa, t2.abab) * -1.0
    x2 += einsum("kb,jica->iajkbc", t1.aa, t2.abab) * -1.0
    x2 += einsum("kc,jiba->iajkbc", t1.aa, t2.abab)
    t4_bbaabbaa += einsum("ib,jalkdc->ijklabcd", t1.bb, x2)
    t4_bbaabbaa += einsum("ja,iblkdc->ijklabcd", t1.bb, x2)
    t4_bbaabbaa += einsum("ia,jblkdc->ijklabcd", t1.bb, x2) * -1.0
    t4_bbaabbaa += einsum("jb,ialkdc->ijklabcd", t1.bb, x2) * -1.0
    return t4_bbaabbaa

def t4_uhf_bbabbbab(c4=None, t1=None, t2=None, t3=None, nocc=None, nvir=None):
    t4_bbabbbab = np.zeros((nocc[1], nocc[1], nocc[0], nocc[1], nvir[1], nvir[1], nvir[0], nvir[1]), dtype=np.float64)
    t4_bbabbbab += einsum("kijlcabd->ijklabcd", c4.abbbabbb)
    t4_bbabbbab += einsum("ia,jklbcd->ijklabcd", t1.bb, t3.babbab) * -1.0
    t4_bbabbbab += einsum("id,jklacb->ijklabcd", t1.bb, t3.babbab) * -1.0
    t4_bbabbbab += einsum("ja,iklbcd->ijklabcd", t1.bb, t3.babbab)
    t4_bbabbbab += einsum("jd,iklacb->ijklabcd", t1.bb, t3.babbab)
    t4_bbabbbab += einsum("la,ikjbcd->ijklabcd", t1.bb, t3.babbab) * -1.0
    t4_bbabbbab += einsum("ld,ikjacb->ijklabcd", t1.bb, t3.babbab) * -1.0
    t4_bbabbbab += einsum("klcd,ijab->ijklabcd", t2.abab, t2.bbbb) * -1.0
    t4_bbabbbab += einsum("klca,ijbd->ijklabcd", t2.abab, t2.bbbb) * -1.0
    t4_bbabbbab += einsum("kjcd,ilab->ijklabcd", t2.abab, t2.bbbb)
    t4_bbabbbab += einsum("kjca,ilbd->ijklabcd", t2.abab, t2.bbbb)
    t4_bbabbbab += einsum("kicd,jlab->ijklabcd", t2.abab, t2.bbbb) * -1.0
    t4_bbabbbab += einsum("kica,jlbd->ijklabcd", t2.abab, t2.bbbb) * -1.0
    x0 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x0 += einsum("ijab->ijab", t2.bbbb)
    x0 += einsum("ib,ja->ijab", t1.bb, t1.bb) * -1.0
    x0 += einsum("ia,jb->ijab", t1.bb, t1.bb)
    t4_bbabbbab += einsum("klcb,jida->ijklabcd", t2.abab, x0)
    t4_bbabbbab += einsum("kicb,ljda->ijklabcd", t2.abab, x0)
    t4_bbabbbab += einsum("kjcb,lida->ijklabcd", t2.abab, x0) * -1.0
    x1 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=np.float64)
    x1 += einsum("ijkabc->ijkabc", t3.bbbbbb)
    x1 += einsum("ja,ikbc->ijkabc", t1.bb, t2.bbbb) * -1.0
    x1 += einsum("jb,ikac->ijkabc", t1.bb, t2.bbbb)
    x1 += einsum("jc,ikab->ijkabc", t1.bb, t2.bbbb) * -1.0
    x1 += einsum("ka,ijbc->ijkabc", t1.bb, t2.bbbb)
    x1 += einsum("kb,ijac->ijkabc", t1.bb, t2.bbbb) * -1.0
    x1 += einsum("kc,ijab->ijkabc", t1.bb, t2.bbbb)
    x1 += einsum("ia,kjcb->ijkabc", t1.bb, x0)
    x1 += einsum("ic,kjba->ijkabc", t1.bb, x0)
    x1 += einsum("ib,kjca->ijkabc", t1.bb, x0) * -1.0
    t4_bbabbbab += einsum("kc,lijdab->ijklabcd", t1.aa, x1) * -1.0
    x2 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=np.float64)
    x2 += einsum("ikjacb->ijabkc", t3.babbab)
    x2 += einsum("ia,kjcb->ijabkc", t1.bb, t2.abab)
    x2 += einsum("ib,kjca->ijabkc", t1.bb, t2.abab) * -1.0
    x2 += einsum("ja,kicb->ijabkc", t1.bb, t2.abab) * -1.0
    x2 += einsum("jb,kica->ijabkc", t1.bb, t2.abab)
    t4_bbabbbab += einsum("ib,ljdakc->ijklabcd", t1.bb, x2)
    t4_bbabbbab += einsum("lb,jidakc->ijklabcd", t1.bb, x2)
    t4_bbabbbab += einsum("jb,lidakc->ijklabcd", t1.bb, x2) * -1.0
    return t4_bbabbbab

def t4_uhf_bbbabbba(c4=None, t1=None, t2=None, t3=None, nocc=None, nvir=None):
    t4_bbbabbba = np.zeros((nocc[1], nocc[1], nocc[1], nocc[0], nvir[1], nvir[1], nvir[1], nvir[0]), dtype=np.float64)
    t4_bbbabbba += einsum("lijkdabc->ijklabcd", c4.abbbabbb)
    t4_bbbabbba += einsum("ia,jlkbdc->ijklabcd", t1.bb, t3.babbab) * -1.0
    t4_bbbabbba += einsum("ic,jlkadb->ijklabcd", t1.bb, t3.babbab) * -1.0
    t4_bbbabbba += einsum("ja,ilkbdc->ijklabcd", t1.bb, t3.babbab)
    t4_bbbabbba += einsum("jc,ilkadb->ijklabcd", t1.bb, t3.babbab)
    t4_bbbabbba += einsum("ka,iljbdc->ijklabcd", t1.bb, t3.babbab) * -1.0
    t4_bbbabbba += einsum("kc,iljadb->ijklabcd", t1.bb, t3.babbab) * -1.0
    t4_bbbabbba += einsum("lkdc,ijab->ijklabcd", t2.abab, t2.bbbb) * -1.0
    t4_bbbabbba += einsum("lkda,ijbc->ijklabcd", t2.abab, t2.bbbb) * -1.0
    t4_bbbabbba += einsum("ljdc,ikab->ijklabcd", t2.abab, t2.bbbb)
    t4_bbbabbba += einsum("ljda,ikbc->ijklabcd", t2.abab, t2.bbbb)
    t4_bbbabbba += einsum("lidc,jkab->ijklabcd", t2.abab, t2.bbbb) * -1.0
    t4_bbbabbba += einsum("lida,jkbc->ijklabcd", t2.abab, t2.bbbb) * -1.0
    x0 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x0 += einsum("ijab->ijab", t2.bbbb)
    x0 += einsum("ib,ja->ijab", t1.bb, t1.bb) * -1.0
    x0 += einsum("ia,jb->ijab", t1.bb, t1.bb)
    t4_bbbabbba += einsum("lkdb,jica->ijklabcd", t2.abab, x0)
    t4_bbbabbba += einsum("lidb,kjca->ijklabcd", t2.abab, x0)
    t4_bbbabbba += einsum("ljdb,kica->ijklabcd", t2.abab, x0) * -1.0
    x1 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=np.float64)
    x1 += einsum("ijkabc->ijkabc", t3.bbbbbb)
    x1 += einsum("ja,ikbc->ijkabc", t1.bb, t2.bbbb) * -1.0
    x1 += einsum("jb,ikac->ijkabc", t1.bb, t2.bbbb)
    x1 += einsum("jc,ikab->ijkabc", t1.bb, t2.bbbb) * -1.0
    x1 += einsum("ka,ijbc->ijkabc", t1.bb, t2.bbbb)
    x1 += einsum("kb,ijac->ijkabc", t1.bb, t2.bbbb) * -1.0
    x1 += einsum("kc,ijab->ijkabc", t1.bb, t2.bbbb)
    x1 += einsum("ia,kjcb->ijkabc", t1.bb, x0)
    x1 += einsum("ic,kjba->ijkabc", t1.bb, x0)
    x1 += einsum("ib,kjca->ijkabc", t1.bb, x0) * -1.0
    t4_bbbabbba += einsum("ld,kijcab->ijklabcd", t1.aa, x1) * -1.0
    x2 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1], nocc[0], nvir[0]), dtype=np.float64)
    x2 += einsum("ikjacb->ijabkc", t3.babbab)
    x2 += einsum("ia,kjcb->ijabkc", t1.bb, t2.abab)
    x2 += einsum("ib,kjca->ijabkc", t1.bb, t2.abab) * -1.0
    x2 += einsum("ja,kicb->ijabkc", t1.bb, t2.abab) * -1.0
    x2 += einsum("jb,kica->ijabkc", t1.bb, t2.abab)
    t4_bbbabbba += einsum("ib,kjcald->ijklabcd", t1.bb, x2)
    t4_bbbabbba += einsum("kb,jicald->ijklabcd", t1.bb, x2)
    t4_bbbabbba += einsum("jb,kicald->ijklabcd", t1.bb, x2) * -1.0
    return t4_bbbabbba

def t4_uhf_bbbbbbbb(c4=None, t1=None, t2=None, t3=None, nocc=None, nvir=None):
    t4_bbbbbbbb = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1], nvir[1]), dtype=np.float64)
    t4_bbbbbbbb += einsum("ijklabcd->ijklabcd", c4.bbbbbbbb)
    t4_bbbbbbbb += einsum("la,ijkbcd->ijklabcd", t1.bb, t3.bbbbbb)
    t4_bbbbbbbb += einsum("lb,ijkacd->ijklabcd", t1.bb, t3.bbbbbb) * -1.0
    t4_bbbbbbbb += einsum("lc,ijkabd->ijklabcd", t1.bb, t3.bbbbbb)
    t4_bbbbbbbb += einsum("ld,ijkabc->ijklabcd", t1.bb, t3.bbbbbb) * -1.0
    t4_bbbbbbbb += einsum("ilcd,jkab->ijklabcd", t2.bbbb, t2.bbbb) * -1.0
    t4_bbbbbbbb += einsum("ilbd,jkac->ijklabcd", t2.bbbb, t2.bbbb)
    t4_bbbbbbbb += einsum("ilad,jkbc->ijklabcd", t2.bbbb, t2.bbbb) * -1.0
    t4_bbbbbbbb += einsum("ilbc,jkad->ijklabcd", t2.bbbb, t2.bbbb) * -1.0
    t4_bbbbbbbb += einsum("ilac,jkbd->ijklabcd", t2.bbbb, t2.bbbb)
    t4_bbbbbbbb += einsum("ilab,jkcd->ijklabcd", t2.bbbb, t2.bbbb) * -1.0
    t4_bbbbbbbb += einsum("ikcd,jlab->ijklabcd", t2.bbbb, t2.bbbb)
    t4_bbbbbbbb += einsum("ikbd,jlac->ijklabcd", t2.bbbb, t2.bbbb) * -1.0
    t4_bbbbbbbb += einsum("ikad,jlbc->ijklabcd", t2.bbbb, t2.bbbb)
    t4_bbbbbbbb += einsum("ikbc,jlad->ijklabcd", t2.bbbb, t2.bbbb)
    t4_bbbbbbbb += einsum("ikac,jlbd->ijklabcd", t2.bbbb, t2.bbbb) * -1.0
    t4_bbbbbbbb += einsum("ikab,jlcd->ijklabcd", t2.bbbb, t2.bbbb)
    t4_bbbbbbbb += einsum("ijcd,klab->ijklabcd", t2.bbbb, t2.bbbb) * -1.0
    t4_bbbbbbbb += einsum("ijbd,klac->ijklabcd", t2.bbbb, t2.bbbb)
    t4_bbbbbbbb += einsum("ijad,klbc->ijklabcd", t2.bbbb, t2.bbbb) * -1.0
    t4_bbbbbbbb += einsum("ijbc,klad->ijklabcd", t2.bbbb, t2.bbbb) * -1.0
    t4_bbbbbbbb += einsum("ijac,klbd->ijklabcd", t2.bbbb, t2.bbbb)
    t4_bbbbbbbb += einsum("ijab,klcd->ijklabcd", t2.bbbb, t2.bbbb) * -1.0
    x0 = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    x0 += einsum("ijab->ijab", t2.bbbb)
    x0 += einsum("ib,ja->ijab", t1.bb, t1.bb) * -1.0
    x0 += einsum("ia,jb->ijab", t1.bb, t1.bb)
    x1 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=np.float64)
    x1 += einsum("ijkabc->ijkabc", t3.bbbbbb)
    x1 += einsum("ja,ikbc->ijkabc", t1.bb, t2.bbbb) * -1.0
    x1 += einsum("jb,ikac->ijkabc", t1.bb, t2.bbbb)
    x1 += einsum("jc,ikab->ijkabc", t1.bb, t2.bbbb) * -1.0
    x1 += einsum("ka,ijbc->ijkabc", t1.bb, t2.bbbb)
    x1 += einsum("kb,ijac->ijkabc", t1.bb, t2.bbbb) * -1.0
    x1 += einsum("kc,ijab->ijkabc", t1.bb, t2.bbbb)
    x1 += einsum("ia,kjcb->ijkabc", t1.bb, x0)
    x1 += einsum("ic,kjba->ijkabc", t1.bb, x0)
    x1 += einsum("ib,kjca->ijkabc", t1.bb, x0) * -1.0
    t4_bbbbbbbb += einsum("ib,ljkdac->ijklabcd", t1.bb, x1)
    t4_bbbbbbbb += einsum("id,ljkcab->ijklabcd", t1.bb, x1)
    t4_bbbbbbbb += einsum("ia,ljkdbc->ijklabcd", t1.bb, x1) * -1.0
    t4_bbbbbbbb += einsum("ic,ljkdab->ijklabcd", t1.bb, x1) * -1.0
    x2 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=np.float64)
    x2 += einsum("ijkabc->ijkabc", t3.bbbbbb)
    x2 += einsum("ia,jkbc->ijkabc", t1.bb, t2.bbbb)
    x2 += einsum("ib,jkac->ijkabc", t1.bb, t2.bbbb) * -1.0
    x2 += einsum("ic,jkab->ijkabc", t1.bb, t2.bbbb)
    x2 += einsum("ka,ijbc->ijkabc", t1.bb, t2.bbbb)
    x2 += einsum("kb,ijac->ijkabc", t1.bb, t2.bbbb) * -1.0
    x2 += einsum("kc,ijab->ijkabc", t1.bb, t2.bbbb)
    t4_bbbbbbbb += einsum("ja,likdbc->ijklabcd", t1.bb, x2)
    t4_bbbbbbbb += einsum("jc,likdab->ijklabcd", t1.bb, x2)
    t4_bbbbbbbb += einsum("jb,likdac->ijklabcd", t1.bb, x2) * -1.0
    t4_bbbbbbbb += einsum("jd,likcab->ijklabcd", t1.bb, x2) * -1.0
    x3 = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=np.float64)
    x3 += einsum("ijkabc->ijkabc", t3.bbbbbb)
    x3 += einsum("ia,jkbc->ijkabc", t1.bb, t2.bbbb)
    x3 += einsum("ib,jkac->ijkabc", t1.bb, t2.bbbb) * -1.0
    x3 += einsum("ic,jkab->ijkabc", t1.bb, t2.bbbb)
    t4_bbbbbbbb += einsum("kb,lijdac->ijklabcd", t1.bb, x3)
    t4_bbbbbbbb += einsum("kd,lijcab->ijklabcd", t1.bb, x3)
    t4_bbbbbbbb += einsum("ka,lijdbc->ijklabcd", t1.bb, x3) * -1.0
    t4_bbbbbbbb += einsum("kc,lijdab->ijklabcd", t1.bb, x3) * -1.0
    return t4_bbbbbbbb

def t4_rhf_abab(c4=None, t1=None, t2=None, t3=None, nocc=None, nvir=None):
    t4 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir, nvir, nvir), dtype=np.float64)
    t4 += einsum("ijklabcd->ijklabcd", c4)
    t4 += einsum("jb,ilkadc->ijklabcd", t1, t3) * -1.0
    t4 += einsum("jd,ilkabc->ijklabcd", t1, t3)
    t4 += einsum("lb,ijkadc->ijklabcd", t1, t3)
    t4 += einsum("ld,ijkabc->ijklabcd", t1, t3) * -1.0
    t4 += einsum("ijab,klcd->ijklabcd", t2, t2) * -1.0
    t4 += einsum("ijad,klcb->ijklabcd", t2, t2)
    t4 += einsum("ijcb,klad->ijklabcd", t2, t2)
    t4 += einsum("ijcd,klab->ijklabcd", t2, t2) * -1.0
    t4 += einsum("ilab,kjcd->ijklabcd", t2, t2)
    t4 += einsum("ilad,kjcb->ijklabcd", t2, t2) * -1.0
    t4 += einsum("ilcb,kjad->ijklabcd", t2, t2) * -1.0
    t4 += einsum("ilcd,kjab->ijklabcd", t2, t2)
    x0 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x0 += einsum("ijab->ijab", t2) * -1.0
    x0 += einsum("ijba->ijab", t2)
    x0 += einsum("ib,ja->ijab", t1, t1)
    x0 += einsum("ia,jb->ijab", t1, t1) * -1.0
    t4 += einsum("ikac,jldb->ijklabcd", x0, x0)
    x1 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    x1 += einsum("ijkabc->ijkabc", t3)
    x1 += einsum("ia,jkbc->ijkabc", t1, t2)
    x1 += einsum("ic,jkba->ijkabc", t1, t2) * -1.0
    x1 += einsum("ka,jibc->ijkabc", t1, t2) * -1.0
    x1 += einsum("kc,jiba->ijkabc", t1, t2)
    t4 += einsum("ic,jklbad->ijklabcd", t1, x1)
    t4 += einsum("ka,jilbcd->ijklabcd", t1, x1)
    t4 += einsum("ia,jklbcd->ijklabcd", t1, x1) * -1.0
    t4 += einsum("kc,jilbad->ijklabcd", t1, x1) * -1.0
    return t4

def t4_rhf_abaa(c4=None, t1=None, t2=None, t3=None, nocc=None, nvir=None):
    t4 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir, nvir, nvir), dtype=np.float64)
    t4 += einsum("ikljacdb->ijklabcd", c4)
    t4 += einsum("ia,kjlcbd->ijklabcd", t1, t3) * -1.0
    t4 += einsum("id,kjlabc->ijklabcd", t1, t3) * -1.0
    t4 += einsum("ka,ijlcbd->ijklabcd", t1, t3)
    t4 += einsum("kd,ijlabc->ijklabcd", t1, t3)
    t4 += einsum("la,ijkcbd->ijklabcd", t1, t3) * -1.0
    t4 += einsum("ld,ijkabc->ijklabcd", t1, t3) * -1.0
    t4 += einsum("ijab,klcd->ijklabcd", t2, t2) * -1.0
    t4 += einsum("ijab,kldc->ijklabcd", t2, t2)
    t4 += einsum("ijdb,klac->ijklabcd", t2, t2) * -1.0
    t4 += einsum("ijdb,klca->ijklabcd", t2, t2)
    t4 += einsum("ilac,kjdb->ijklabcd", t2, t2)
    t4 += einsum("ilcd,kjab->ijklabcd", t2, t2)
    t4 += einsum("ilca,kjdb->ijklabcd", t2, t2) * -1.0
    t4 += einsum("ildc,kjab->ijklabcd", t2, t2) * -1.0
    t4 += einsum("ikac,ljdb->ijklabcd", t2, t2) * -1.0
    t4 += einsum("ikcd,ljab->ijklabcd", t2, t2) * -1.0
    t4 += einsum("ikca,ljdb->ijklabcd", t2, t2)
    t4 += einsum("ikdc,ljab->ijklabcd", t2, t2)
    x0 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x0 += einsum("ijab->ijab", t2) * -1.0
    x0 += einsum("ijba->ijab", t2)
    x0 += einsum("ib,ja->ijab", t1, t1)
    x0 += einsum("ia,jb->ijab", t1, t1) * -1.0
    t4 += einsum("kjcb,ilda->ijklabcd", t2, x0) * -1.0
    x1 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    x1 += einsum("ijab->ijab", t2)
    x1 += einsum("ijba->ijab", t2) * -1.0
    x1 += einsum("ib,ja->ijab", t1, t1) * -1.0
    x1 += einsum("ia,jb->ijab", t1, t1)
    t4 += einsum("ijcb,klda->ijklabcd", t2, x1) * -1.0
    t4 += einsum("ljcb,ikda->ijklabcd", t2, x1) * -1.0
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
    t4 += einsum("jb,iklacd->ijklabcd", t1, x2)
    x3 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    x3 += einsum("ijkabc->ijkabc", t3)
    x3 += einsum("ia,kjcb->ijkabc", t1, t2)
    x3 += einsum("ic,kjab->ijkabc", t1, t2) * -1.0
    x3 += einsum("ka,ijcb->ijkabc", t1, t2) * -1.0
    x3 += einsum("kc,ijab->ijkabc", t1, t2)
    t4 += einsum("ic,kjlabd->ijklabcd", t1, x3)
    t4 += einsum("lc,ijkabd->ijklabcd", t1, x3)
    t4 += einsum("kc,ijlabd->ijklabcd", t1, x3) * -1.0
    return t4

