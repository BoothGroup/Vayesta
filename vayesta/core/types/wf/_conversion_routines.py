import numpy as np
from ebcc.util import pack_2e, einsum, Namespace

def t1_uhf_aa(c1=None, nocc=None, nvir=None):
    t1_aa = np.zeros((nocc[0], nvir[0]), dtype=np.float64)
    t1_aa += einsum(c1.aa, (0, 1), (0, 1))
    return t1_aa

def t1_uhf_bb(c1=None, nocc=None, nvir=None):
    t1_bb = np.zeros((nocc[1], nvir[1]), dtype=np.float64)
    t1_bb += einsum(c1.bb, (0, 1), (0, 1))
    return t1_bb

def t1_rhf(c1=None, nocc=None, nvir=None):
    t1 = np.zeros((nocc, nvir), dtype=np.float64)
    t1 += einsum(c1, (0, 1), (0, 1))
    return t1

def t1_ghf(c1=None, nocc=None, nvir=None):
    t1 = np.zeros((nocc, nvir), dtype=np.float64)
    t1 += einsum(c1, (0, 1), (0, 1))
    return t1

def t2_uhf_aaaa(c2=None, t1=None, nocc=None, nvir=None):
    t2_aaaa = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
    t2_aaaa += einsum(c2.aaaa, (0, 1, 2, 3), (0, 1, 2, 3))
    t2_aaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), (0, 2, 1, 3)) * -1.0
    t2_aaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), (0, 2, 3, 1))
    return t2_aaaa

def t2_uhf_abab(c2=None, t1=None, nocc=None, nvir=None):
    t2_abab = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
    t2_abab += einsum(c2.abab, (0, 1, 2, 3), (0, 1, 2, 3))
    t2_abab += einsum(t1.aa, (0, 1), t1.bb, (2, 3), (0, 2, 1, 3)) * -1.0
    return t2_abab

def t2_uhf_baba(c2=None, t1=None, nocc=None, nvir=None):
    t2_baba = np.zeros((nocc[1], nocc[0], nvir[1], nvir[0]), dtype=np.float64)
    t2_baba += einsum(c2.abab, (0, 1, 2, 3), (1, 0, 3, 2))
    t2_baba += einsum(t1.aa, (0, 1), t1.bb, (2, 3), (2, 0, 3, 1)) * -1.0
    return t2_baba

def t2_uhf_bbbb(c2=None, t1=None, nocc=None, nvir=None):
    t2_bbbb = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
    t2_bbbb += einsum(c2.bbbb, (0, 1, 2, 3), (0, 1, 2, 3))
    t2_bbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), (0, 2, 1, 3)) * -1.0
    t2_bbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), (0, 2, 3, 1))
    return t2_bbbb

def t2_rhf(c2=None, t1=None, nocc=None, nvir=None):
    t2 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    t2 += einsum(c2, (0, 1, 2, 3), (0, 1, 2, 3))
    t2 += einsum(t1, (0, 1), t1, (2, 3), (0, 2, 1, 3)) * -1.0
    return t2

def t2_ghf(c2=None, t1=None, nocc=None, nvir=None):
    t2 = np.zeros((nocc, nocc, nvir, nvir), dtype=np.float64)
    t2 += einsum(c2, (0, 1, 2, 3), (0, 1, 2, 3))
    t2 += einsum(t1, (0, 1), t1, (2, 3), (0, 2, 1, 3)) * -1.0
    t2 += einsum(t1, (0, 1), t1, (2, 3), (0, 2, 3, 1))
    return t2

def t3_uhf_aaaaaa(c3=None, t1=None, t2=None, nocc=None, nvir=None):
    t3_aaaaaa = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=np.float64)
    t3_aaaaaa += einsum(c3.aaaaaa, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    t3_aaaaaa += einsum(t1.aa, (0, 1), t2.aaaa, (2, 3, 4, 5), (0, 2, 3, 1, 4, 5)) * -1.0
    t3_aaaaaa += einsum(t1.aa, (0, 1), t2.aaaa, (2, 3, 4, 5), (0, 2, 3, 4, 1, 5))
    t3_aaaaaa += einsum(t1.aa, (0, 1), t2.aaaa, (2, 3, 4, 5), (0, 2, 3, 4, 5, 1)) * -1.0
    t3_aaaaaa += einsum(t1.aa, (0, 1), t2.aaaa, (2, 3, 4, 5), (2, 0, 3, 1, 4, 5))
    t3_aaaaaa += einsum(t1.aa, (0, 1), t2.aaaa, (2, 3, 4, 5), (2, 0, 3, 4, 1, 5)) * -1.0
    t3_aaaaaa += einsum(t1.aa, (0, 1), t2.aaaa, (2, 3, 4, 5), (2, 0, 3, 4, 5, 1))
    t3_aaaaaa += einsum(t1.aa, (0, 1), t2.aaaa, (2, 3, 4, 5), (2, 3, 0, 1, 4, 5)) * -1.0
    t3_aaaaaa += einsum(t1.aa, (0, 1), t2.aaaa, (2, 3, 4, 5), (2, 3, 0, 4, 1, 5))
    t3_aaaaaa += einsum(t1.aa, (0, 1), t2.aaaa, (2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -1.0
    t3_aaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t1.aa, (4, 5), (0, 2, 4, 1, 3, 5)) * -1.0
    t3_aaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t1.aa, (4, 5), (0, 2, 4, 1, 5, 3))
    t3_aaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t1.aa, (4, 5), (0, 2, 4, 3, 1, 5))
    t3_aaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t1.aa, (4, 5), (0, 2, 4, 5, 1, 3)) * -1.0
    t3_aaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t1.aa, (4, 5), (0, 2, 4, 3, 5, 1)) * -1.0
    t3_aaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t1.aa, (4, 5), (0, 2, 4, 5, 3, 1))
    return t3_aaaaaa

def t3_uhf_aabaab(c3=None, t1=None, t2=None, nocc=None, nvir=None):
    t3_aabaab = np.zeros((nocc[0], nocc[0], nocc[1], nvir[0], nvir[0], nvir[1]), dtype=np.float64)
    t3_aabaab += einsum(c3.abaaba, (0, 1, 2, 3, 4, 5), (0, 2, 1, 3, 5, 4))
    t3_aabaab += einsum(t1.aa, (0, 1), t2.abab, (2, 3, 4, 5), (0, 2, 3, 1, 4, 5)) * -1.0
    t3_aabaab += einsum(t1.aa, (0, 1), t2.abab, (2, 3, 4, 5), (0, 2, 3, 4, 1, 5))
    t3_aabaab += einsum(t1.aa, (0, 1), t2.abab, (2, 3, 4, 5), (2, 0, 3, 1, 4, 5))
    t3_aabaab += einsum(t1.aa, (0, 1), t2.abab, (2, 3, 4, 5), (2, 0, 3, 4, 1, 5)) * -1.0
    t3_aabaab += einsum(t1.bb, (0, 1), t2.aaaa, (2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -1.0
    t3_aabaab += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t1.bb, (4, 5), (0, 2, 4, 1, 3, 5)) * -1.0
    t3_aabaab += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t1.bb, (4, 5), (0, 2, 4, 3, 1, 5))
    return t3_aabaab

def t3_uhf_abaaba(c3=None, t1=None, t2=None, nocc=None, nvir=None):
    t3_abaaba = np.zeros((nocc[0], nocc[1], nocc[0], nvir[0], nvir[1], nvir[0]), dtype=np.float64)
    t3_abaaba += einsum(c3.abaaba, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    t3_abaaba += einsum(t1.aa, (0, 1), t2.abab, (2, 3, 4, 5), (0, 3, 2, 1, 5, 4)) * -1.0
    t3_abaaba += einsum(t1.aa, (0, 1), t2.abab, (2, 3, 4, 5), (0, 3, 2, 4, 5, 1))
    t3_abaaba += einsum(t1.aa, (0, 1), t2.abab, (2, 3, 4, 5), (2, 3, 0, 1, 5, 4))
    t3_abaaba += einsum(t1.aa, (0, 1), t2.abab, (2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -1.0
    t3_abaaba += einsum(t1.bb, (0, 1), t2.aaaa, (2, 3, 4, 5), (2, 0, 3, 4, 1, 5)) * -1.0
    t3_abaaba += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t1.bb, (4, 5), (0, 4, 2, 1, 5, 3)) * -1.0
    t3_abaaba += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t1.bb, (4, 5), (0, 4, 2, 3, 5, 1))
    return t3_abaaba

def t3_uhf_abbabb(c3=None, t1=None, t2=None, nocc=None, nvir=None):
    t3_abbabb = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0], nvir[1], nvir[1]), dtype=np.float64)
    t3_abbabb += einsum(c3.abbabb, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    t3_abbabb += einsum(t1.aa, (0, 1), t2.bbbb, (2, 3, 4, 5), (0, 2, 3, 1, 4, 5)) * -1.0
    t3_abbabb += einsum(t1.bb, (0, 1), t2.abab, (2, 3, 4, 5), (2, 0, 3, 4, 1, 5)) * -1.0
    t3_abbabb += einsum(t1.bb, (0, 1), t2.abab, (2, 3, 4, 5), (2, 0, 3, 4, 5, 1))
    t3_abbabb += einsum(t1.bb, (0, 1), t2.abab, (2, 3, 4, 5), (2, 3, 0, 4, 1, 5))
    t3_abbabb += einsum(t1.bb, (0, 1), t2.abab, (2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -1.0
    t3_abbabb += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t1.bb, (4, 5), (0, 2, 4, 1, 3, 5)) * -1.0
    t3_abbabb += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t1.bb, (4, 5), (0, 2, 4, 1, 5, 3))
    return t3_abbabb

def t3_uhf_baabaa(c3=None, t1=None, t2=None, nocc=None, nvir=None):
    t3_baabaa = np.zeros((nocc[1], nocc[0], nocc[0], nvir[1], nvir[0], nvir[0]), dtype=np.float64)
    t3_baabaa += einsum(c3.abaaba, (0, 1, 2, 3, 4, 5), (1, 0, 2, 4, 3, 5))
    t3_baabaa += einsum(t1.aa, (0, 1), t2.abab, (2, 3, 4, 5), (3, 0, 2, 5, 1, 4)) * -1.0
    t3_baabaa += einsum(t1.aa, (0, 1), t2.abab, (2, 3, 4, 5), (3, 0, 2, 5, 4, 1))
    t3_baabaa += einsum(t1.aa, (0, 1), t2.abab, (2, 3, 4, 5), (3, 2, 0, 5, 1, 4))
    t3_baabaa += einsum(t1.aa, (0, 1), t2.abab, (2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -1.0
    t3_baabaa += einsum(t1.bb, (0, 1), t2.aaaa, (2, 3, 4, 5), (0, 2, 3, 1, 4, 5)) * -1.0
    t3_baabaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t1.bb, (4, 5), (4, 0, 2, 5, 1, 3)) * -1.0
    t3_baabaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t1.bb, (4, 5), (4, 0, 2, 5, 3, 1))
    return t3_baabaa

def t3_uhf_babbab(c3=None, t1=None, t2=None, nocc=None, nvir=None):
    t3_babbab = np.zeros((nocc[1], nocc[0], nocc[1], nvir[1], nvir[0], nvir[1]), dtype=np.float64)
    t3_babbab += einsum(c3.babbab, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    t3_babbab += einsum(t1.aa, (0, 1), t2.bbbb, (2, 3, 4, 5), (2, 0, 3, 4, 1, 5)) * -1.0
    t3_babbab += einsum(t1.bb, (0, 1), t2.abab, (2, 3, 4, 5), (0, 2, 3, 1, 4, 5)) * -1.0
    t3_babbab += einsum(t1.bb, (0, 1), t2.abab, (2, 3, 4, 5), (0, 2, 3, 5, 4, 1))
    t3_babbab += einsum(t1.bb, (0, 1), t2.abab, (2, 3, 4, 5), (3, 2, 0, 1, 4, 5))
    t3_babbab += einsum(t1.bb, (0, 1), t2.abab, (2, 3, 4, 5), (3, 2, 0, 5, 4, 1)) * -1.0
    t3_babbab += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t1.bb, (4, 5), (2, 0, 4, 3, 1, 5)) * -1.0
    t3_babbab += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t1.bb, (4, 5), (2, 0, 4, 5, 1, 3))
    return t3_babbab

def t3_uhf_bbabba(c3=None, t1=None, t2=None, nocc=None, nvir=None):
    t3_bbabba = np.zeros((nocc[1], nocc[1], nocc[0], nvir[1], nvir[1], nvir[0]), dtype=np.float64)
    t3_bbabba += einsum(c3.bbabba, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    t3_bbabba += einsum(t1.aa, (0, 1), t2.bbbb, (2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -1.0
    t3_bbabba += einsum(t1.bb, (0, 1), t2.abab, (2, 3, 4, 5), (0, 3, 2, 1, 5, 4)) * -1.0
    t3_bbabba += einsum(t1.bb, (0, 1), t2.abab, (2, 3, 4, 5), (0, 3, 2, 5, 1, 4))
    t3_bbabba += einsum(t1.bb, (0, 1), t2.abab, (2, 3, 4, 5), (3, 0, 2, 1, 5, 4))
    t3_bbabba += einsum(t1.bb, (0, 1), t2.abab, (2, 3, 4, 5), (3, 0, 2, 5, 1, 4)) * -1.0
    t3_bbabba += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t1.bb, (4, 5), (2, 4, 0, 3, 5, 1)) * -1.0
    t3_bbabba += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t1.bb, (4, 5), (2, 4, 0, 5, 3, 1))
    return t3_bbabba

def t3_uhf_bbbbbb(c3=None, t1=None, t2=None, nocc=None, nvir=None):
    t3_bbbbbb = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=np.float64)
    t3_bbbbbb += einsum(c3.bbbbbb, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    t3_bbbbbb += einsum(t1.bb, (0, 1), t2.bbbb, (2, 3, 4, 5), (0, 2, 3, 1, 4, 5)) * -1.0
    t3_bbbbbb += einsum(t1.bb, (0, 1), t2.bbbb, (2, 3, 4, 5), (0, 2, 3, 4, 1, 5))
    t3_bbbbbb += einsum(t1.bb, (0, 1), t2.bbbb, (2, 3, 4, 5), (0, 2, 3, 4, 5, 1)) * -1.0
    t3_bbbbbb += einsum(t1.bb, (0, 1), t2.bbbb, (2, 3, 4, 5), (2, 0, 3, 1, 4, 5))
    t3_bbbbbb += einsum(t1.bb, (0, 1), t2.bbbb, (2, 3, 4, 5), (2, 0, 3, 4, 1, 5)) * -1.0
    t3_bbbbbb += einsum(t1.bb, (0, 1), t2.bbbb, (2, 3, 4, 5), (2, 0, 3, 4, 5, 1))
    t3_bbbbbb += einsum(t1.bb, (0, 1), t2.bbbb, (2, 3, 4, 5), (2, 3, 0, 1, 4, 5)) * -1.0
    t3_bbbbbb += einsum(t1.bb, (0, 1), t2.bbbb, (2, 3, 4, 5), (2, 3, 0, 4, 1, 5))
    t3_bbbbbb += einsum(t1.bb, (0, 1), t2.bbbb, (2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -1.0
    t3_bbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t1.bb, (4, 5), (0, 2, 4, 1, 3, 5)) * -1.0
    t3_bbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t1.bb, (4, 5), (0, 2, 4, 1, 5, 3))
    t3_bbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t1.bb, (4, 5), (0, 2, 4, 3, 1, 5))
    t3_bbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t1.bb, (4, 5), (0, 2, 4, 5, 1, 3)) * -1.0
    t3_bbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t1.bb, (4, 5), (0, 2, 4, 3, 5, 1)) * -1.0
    t3_bbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t1.bb, (4, 5), (0, 2, 4, 5, 3, 1))
    return t3_bbbbbb

def t3_rhf(c3=None, t1=None, t2=None, nocc=None, nvir=None):
    t3 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    t3 += einsum(c3, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    t3 += einsum(t1, (0, 1), t2, (2, 3, 4, 5), (0, 3, 2, 1, 5, 4)) * -1.0
    t3 += einsum(t1, (0, 1), t2, (2, 3, 4, 5), (0, 3, 2, 4, 5, 1))
    t3 += einsum(t1, (0, 1), t2, (2, 3, 4, 5), (2, 0, 3, 4, 1, 5)) * -1.0
    t3 += einsum(t1, (0, 1), t2, (2, 3, 4, 5), (2, 0, 3, 5, 1, 4))
    t3 += einsum(t1, (0, 1), t2, (2, 3, 4, 5), (2, 3, 0, 1, 5, 4))
    t3 += einsum(t1, (0, 1), t2, (2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -1.0
    t3 += einsum(t1, (0, 1), t1, (2, 3), t1, (4, 5), (0, 2, 4, 1, 3, 5)) * -1.0
    t3 += einsum(t1, (0, 1), t1, (2, 3), t1, (4, 5), (0, 2, 4, 5, 3, 1))
    return t3

def t3_ghf(c3=None, t1=None, t2=None, nocc=None, nvir=None):
    t3 = np.zeros((nocc, nocc, nocc, nvir, nvir, nvir), dtype=np.float64)
    t3 += einsum(c3, (0, 1, 2, 3, 4, 5), (0, 1, 2, 3, 4, 5))
    t3 += einsum(t1, (0, 1), t2, (2, 3, 4, 5), (0, 2, 3, 1, 4, 5)) * -1.0
    t3 += einsum(t1, (0, 1), t2, (2, 3, 4, 5), (0, 2, 3, 4, 1, 5))
    t3 += einsum(t1, (0, 1), t2, (2, 3, 4, 5), (0, 2, 3, 4, 5, 1)) * -1.0
    t3 += einsum(t1, (0, 1), t2, (2, 3, 4, 5), (2, 0, 3, 1, 4, 5))
    t3 += einsum(t1, (0, 1), t2, (2, 3, 4, 5), (2, 0, 3, 4, 1, 5)) * -1.0
    t3 += einsum(t1, (0, 1), t2, (2, 3, 4, 5), (2, 0, 3, 4, 5, 1))
    t3 += einsum(t1, (0, 1), t2, (2, 3, 4, 5), (2, 3, 0, 1, 4, 5)) * -1.0
    t3 += einsum(t1, (0, 1), t2, (2, 3, 4, 5), (2, 3, 0, 4, 1, 5))
    t3 += einsum(t1, (0, 1), t2, (2, 3, 4, 5), (2, 3, 0, 4, 5, 1)) * -1.0
    t3 += einsum(t1, (0, 1), t1, (2, 3), t1, (4, 5), (0, 2, 4, 1, 3, 5)) * -1.0
    t3 += einsum(t1, (0, 1), t1, (2, 3), t1, (4, 5), (0, 2, 4, 1, 5, 3))
    t3 += einsum(t1, (0, 1), t1, (2, 3), t1, (4, 5), (0, 2, 4, 3, 1, 5))
    t3 += einsum(t1, (0, 1), t1, (2, 3), t1, (4, 5), (0, 2, 4, 5, 1, 3)) * -1.0
    t3 += einsum(t1, (0, 1), t1, (2, 3), t1, (4, 5), (0, 2, 4, 3, 5, 1)) * -1.0
    t3 += einsum(t1, (0, 1), t1, (2, 3), t1, (4, 5), (0, 2, 4, 5, 3, 1))
    return t3

def t4_uhf_aaaaaaaa(c4=None, t1=None, t2=None, t3=None, nocc=None, nvir=None):
    t4_aaaaaaaa = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0], nvir[0]), dtype=np.float64)
    t4_aaaaaaaa += einsum(c4.aaaaaaaa, (0, 1, 2, 3, 4, 5, 6, 7), (0, 1, 2, 3, 4, 5, 6, 7))
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t3.aaaaaa, (2, 3, 4, 5, 6, 7), (0, 2, 3, 4, 1, 5, 6, 7)) * -1.0
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t3.aaaaaa, (2, 3, 4, 5, 6, 7), (0, 2, 3, 4, 5, 1, 6, 7))
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t3.aaaaaa, (2, 3, 4, 5, 6, 7), (0, 2, 3, 4, 5, 6, 1, 7)) * -1.0
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t3.aaaaaa, (2, 3, 4, 5, 6, 7), (0, 2, 3, 4, 5, 6, 7, 1))
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t3.aaaaaa, (2, 3, 4, 5, 6, 7), (2, 0, 3, 4, 1, 5, 6, 7))
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t3.aaaaaa, (2, 3, 4, 5, 6, 7), (2, 0, 3, 4, 5, 1, 6, 7)) * -1.0
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t3.aaaaaa, (2, 3, 4, 5, 6, 7), (2, 0, 3, 4, 5, 6, 1, 7))
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t3.aaaaaa, (2, 3, 4, 5, 6, 7), (2, 0, 3, 4, 5, 6, 7, 1)) * -1.0
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t3.aaaaaa, (2, 3, 4, 5, 6, 7), (2, 3, 0, 4, 1, 5, 6, 7)) * -1.0
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t3.aaaaaa, (2, 3, 4, 5, 6, 7), (2, 3, 0, 4, 5, 1, 6, 7))
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t3.aaaaaa, (2, 3, 4, 5, 6, 7), (2, 3, 0, 4, 5, 6, 1, 7)) * -1.0
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t3.aaaaaa, (2, 3, 4, 5, 6, 7), (2, 3, 0, 4, 5, 6, 7, 1))
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t3.aaaaaa, (2, 3, 4, 5, 6, 7), (2, 3, 4, 0, 1, 5, 6, 7))
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t3.aaaaaa, (2, 3, 4, 5, 6, 7), (2, 3, 4, 0, 5, 1, 6, 7)) * -1.0
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t3.aaaaaa, (2, 3, 4, 5, 6, 7), (2, 3, 4, 0, 5, 6, 1, 7))
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t3.aaaaaa, (2, 3, 4, 5, 6, 7), (2, 3, 4, 0, 5, 6, 7, 1)) * -1.0
    t4_aaaaaaaa += einsum(t2.aaaa, (0, 1, 2, 3), t2.aaaa, (4, 5, 6, 7), (0, 4, 1, 5, 2, 6, 3, 7)) * -1.0
    t4_aaaaaaaa += einsum(t2.aaaa, (0, 1, 2, 3), t2.aaaa, (4, 5, 6, 7), (0, 4, 1, 5, 2, 6, 7, 3))
    t4_aaaaaaaa += einsum(t2.aaaa, (0, 1, 2, 3), t2.aaaa, (4, 5, 6, 7), (0, 4, 1, 5, 2, 3, 6, 7))
    t4_aaaaaaaa += einsum(t2.aaaa, (0, 1, 2, 3), t2.aaaa, (4, 5, 6, 7), (0, 4, 1, 5, 6, 2, 3, 7))
    t4_aaaaaaaa += einsum(t2.aaaa, (0, 1, 2, 3), t2.aaaa, (4, 5, 6, 7), (0, 4, 1, 5, 6, 2, 7, 3)) * -1.0
    t4_aaaaaaaa += einsum(t2.aaaa, (0, 1, 2, 3), t2.aaaa, (4, 5, 6, 7), (0, 4, 1, 5, 6, 7, 2, 3))
    t4_aaaaaaaa += einsum(t2.aaaa, (0, 1, 2, 3), t2.aaaa, (4, 5, 6, 7), (0, 4, 5, 1, 2, 6, 3, 7))
    t4_aaaaaaaa += einsum(t2.aaaa, (0, 1, 2, 3), t2.aaaa, (4, 5, 6, 7), (0, 4, 5, 1, 2, 6, 7, 3)) * -1.0
    t4_aaaaaaaa += einsum(t2.aaaa, (0, 1, 2, 3), t2.aaaa, (4, 5, 6, 7), (0, 4, 5, 1, 2, 3, 6, 7)) * -1.0
    t4_aaaaaaaa += einsum(t2.aaaa, (0, 1, 2, 3), t2.aaaa, (4, 5, 6, 7), (0, 4, 5, 1, 6, 2, 3, 7)) * -1.0
    t4_aaaaaaaa += einsum(t2.aaaa, (0, 1, 2, 3), t2.aaaa, (4, 5, 6, 7), (0, 4, 5, 1, 6, 2, 7, 3))
    t4_aaaaaaaa += einsum(t2.aaaa, (0, 1, 2, 3), t2.aaaa, (4, 5, 6, 7), (0, 4, 5, 1, 6, 7, 2, 3)) * -1.0
    t4_aaaaaaaa += einsum(t2.aaaa, (0, 1, 2, 3), t2.aaaa, (4, 5, 6, 7), (0, 1, 4, 5, 2, 6, 3, 7))
    t4_aaaaaaaa += einsum(t2.aaaa, (0, 1, 2, 3), t2.aaaa, (4, 5, 6, 7), (0, 1, 4, 5, 2, 6, 7, 3)) * -1.0
    t4_aaaaaaaa += einsum(t2.aaaa, (0, 1, 2, 3), t2.aaaa, (4, 5, 6, 7), (0, 1, 4, 5, 2, 3, 6, 7)) * -1.0
    t4_aaaaaaaa += einsum(t2.aaaa, (0, 1, 2, 3), t2.aaaa, (4, 5, 6, 7), (0, 1, 4, 5, 6, 2, 3, 7)) * -1.0
    t4_aaaaaaaa += einsum(t2.aaaa, (0, 1, 2, 3), t2.aaaa, (4, 5, 6, 7), (0, 1, 4, 5, 6, 2, 7, 3))
    t4_aaaaaaaa += einsum(t2.aaaa, (0, 1, 2, 3), t2.aaaa, (4, 5, 6, 7), (0, 1, 4, 5, 6, 7, 2, 3)) * -1.0
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.aaaa, (4, 5, 6, 7), (0, 2, 4, 5, 1, 3, 6, 7)) * -1.0
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.aaaa, (4, 5, 6, 7), (0, 2, 4, 5, 1, 6, 3, 7))
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.aaaa, (4, 5, 6, 7), (0, 2, 4, 5, 1, 6, 7, 3)) * -1.0
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.aaaa, (4, 5, 6, 7), (0, 2, 4, 5, 3, 1, 6, 7))
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.aaaa, (4, 5, 6, 7), (0, 2, 4, 5, 6, 1, 3, 7)) * -1.0
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.aaaa, (4, 5, 6, 7), (0, 2, 4, 5, 6, 1, 7, 3))
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.aaaa, (4, 5, 6, 7), (0, 2, 4, 5, 3, 6, 1, 7)) * -1.0
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.aaaa, (4, 5, 6, 7), (0, 2, 4, 5, 6, 3, 1, 7))
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.aaaa, (4, 5, 6, 7), (0, 2, 4, 5, 6, 7, 1, 3)) * -1.0
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.aaaa, (4, 5, 6, 7), (0, 2, 4, 5, 3, 6, 7, 1))
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.aaaa, (4, 5, 6, 7), (0, 2, 4, 5, 6, 3, 7, 1)) * -1.0
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.aaaa, (4, 5, 6, 7), (0, 2, 4, 5, 6, 7, 3, 1))
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.aaaa, (4, 5, 6, 7), (0, 4, 2, 5, 1, 3, 6, 7))
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.aaaa, (4, 5, 6, 7), (0, 4, 2, 5, 1, 6, 3, 7)) * -1.0
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.aaaa, (4, 5, 6, 7), (0, 4, 2, 5, 1, 6, 7, 3))
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.aaaa, (4, 5, 6, 7), (0, 4, 2, 5, 3, 1, 6, 7)) * -1.0
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.aaaa, (4, 5, 6, 7), (0, 4, 2, 5, 6, 1, 3, 7))
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.aaaa, (4, 5, 6, 7), (0, 4, 2, 5, 6, 1, 7, 3)) * -1.0
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.aaaa, (4, 5, 6, 7), (0, 4, 2, 5, 3, 6, 1, 7))
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.aaaa, (4, 5, 6, 7), (0, 4, 2, 5, 6, 3, 1, 7)) * -1.0
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.aaaa, (4, 5, 6, 7), (0, 4, 2, 5, 6, 7, 1, 3))
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.aaaa, (4, 5, 6, 7), (0, 4, 2, 5, 3, 6, 7, 1)) * -1.0
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.aaaa, (4, 5, 6, 7), (0, 4, 2, 5, 6, 3, 7, 1))
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.aaaa, (4, 5, 6, 7), (0, 4, 2, 5, 6, 7, 3, 1)) * -1.0
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.aaaa, (4, 5, 6, 7), (0, 4, 5, 2, 1, 3, 6, 7)) * -1.0
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.aaaa, (4, 5, 6, 7), (0, 4, 5, 2, 1, 6, 3, 7))
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.aaaa, (4, 5, 6, 7), (0, 4, 5, 2, 1, 6, 7, 3)) * -1.0
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.aaaa, (4, 5, 6, 7), (0, 4, 5, 2, 3, 1, 6, 7))
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.aaaa, (4, 5, 6, 7), (0, 4, 5, 2, 6, 1, 3, 7)) * -1.0
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.aaaa, (4, 5, 6, 7), (0, 4, 5, 2, 6, 1, 7, 3))
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.aaaa, (4, 5, 6, 7), (0, 4, 5, 2, 3, 6, 1, 7)) * -1.0
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.aaaa, (4, 5, 6, 7), (0, 4, 5, 2, 6, 3, 1, 7))
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.aaaa, (4, 5, 6, 7), (0, 4, 5, 2, 6, 7, 1, 3)) * -1.0
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.aaaa, (4, 5, 6, 7), (0, 4, 5, 2, 3, 6, 7, 1))
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.aaaa, (4, 5, 6, 7), (0, 4, 5, 2, 6, 3, 7, 1)) * -1.0
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.aaaa, (4, 5, 6, 7), (0, 4, 5, 2, 6, 7, 3, 1))
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.aaaa, (4, 5, 6, 7), (4, 0, 2, 5, 1, 3, 6, 7)) * -1.0
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.aaaa, (4, 5, 6, 7), (4, 0, 2, 5, 1, 6, 3, 7))
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.aaaa, (4, 5, 6, 7), (4, 0, 2, 5, 1, 6, 7, 3)) * -1.0
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.aaaa, (4, 5, 6, 7), (4, 0, 2, 5, 3, 1, 6, 7))
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.aaaa, (4, 5, 6, 7), (4, 0, 2, 5, 6, 1, 3, 7)) * -1.0
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.aaaa, (4, 5, 6, 7), (4, 0, 2, 5, 6, 1, 7, 3))
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.aaaa, (4, 5, 6, 7), (4, 0, 2, 5, 3, 6, 1, 7)) * -1.0
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.aaaa, (4, 5, 6, 7), (4, 0, 2, 5, 6, 3, 1, 7))
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.aaaa, (4, 5, 6, 7), (4, 0, 2, 5, 6, 7, 1, 3)) * -1.0
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.aaaa, (4, 5, 6, 7), (4, 0, 2, 5, 3, 6, 7, 1))
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.aaaa, (4, 5, 6, 7), (4, 0, 2, 5, 6, 3, 7, 1)) * -1.0
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.aaaa, (4, 5, 6, 7), (4, 0, 2, 5, 6, 7, 3, 1))
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.aaaa, (4, 5, 6, 7), (4, 0, 5, 2, 1, 3, 6, 7))
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.aaaa, (4, 5, 6, 7), (4, 0, 5, 2, 1, 6, 3, 7)) * -1.0
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.aaaa, (4, 5, 6, 7), (4, 0, 5, 2, 1, 6, 7, 3))
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.aaaa, (4, 5, 6, 7), (4, 0, 5, 2, 3, 1, 6, 7)) * -1.0
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.aaaa, (4, 5, 6, 7), (4, 0, 5, 2, 6, 1, 3, 7))
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.aaaa, (4, 5, 6, 7), (4, 0, 5, 2, 6, 1, 7, 3)) * -1.0
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.aaaa, (4, 5, 6, 7), (4, 0, 5, 2, 3, 6, 1, 7))
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.aaaa, (4, 5, 6, 7), (4, 0, 5, 2, 6, 3, 1, 7)) * -1.0
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.aaaa, (4, 5, 6, 7), (4, 0, 5, 2, 6, 7, 1, 3))
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.aaaa, (4, 5, 6, 7), (4, 0, 5, 2, 3, 6, 7, 1)) * -1.0
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.aaaa, (4, 5, 6, 7), (4, 0, 5, 2, 6, 3, 7, 1))
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.aaaa, (4, 5, 6, 7), (4, 0, 5, 2, 6, 7, 3, 1)) * -1.0
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.aaaa, (4, 5, 6, 7), (4, 5, 0, 2, 1, 3, 6, 7)) * -1.0
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.aaaa, (4, 5, 6, 7), (4, 5, 0, 2, 1, 6, 3, 7))
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.aaaa, (4, 5, 6, 7), (4, 5, 0, 2, 1, 6, 7, 3)) * -1.0
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.aaaa, (4, 5, 6, 7), (4, 5, 0, 2, 3, 1, 6, 7))
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.aaaa, (4, 5, 6, 7), (4, 5, 0, 2, 6, 1, 3, 7)) * -1.0
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.aaaa, (4, 5, 6, 7), (4, 5, 0, 2, 6, 1, 7, 3))
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.aaaa, (4, 5, 6, 7), (4, 5, 0, 2, 3, 6, 1, 7)) * -1.0
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.aaaa, (4, 5, 6, 7), (4, 5, 0, 2, 6, 3, 1, 7))
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.aaaa, (4, 5, 6, 7), (4, 5, 0, 2, 6, 7, 1, 3)) * -1.0
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.aaaa, (4, 5, 6, 7), (4, 5, 0, 2, 3, 6, 7, 1))
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.aaaa, (4, 5, 6, 7), (4, 5, 0, 2, 6, 3, 7, 1)) * -1.0
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.aaaa, (4, 5, 6, 7), (4, 5, 0, 2, 6, 7, 3, 1))
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t1.aa, (4, 5), t1.aa, (6, 7), (0, 2, 4, 6, 1, 3, 5, 7)) * -1.0
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t1.aa, (4, 5), t1.aa, (6, 7), (0, 2, 4, 6, 1, 3, 7, 5))
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t1.aa, (4, 5), t1.aa, (6, 7), (0, 2, 4, 6, 1, 5, 3, 7))
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t1.aa, (4, 5), t1.aa, (6, 7), (0, 2, 4, 6, 1, 7, 3, 5)) * -1.0
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t1.aa, (4, 5), t1.aa, (6, 7), (0, 2, 4, 6, 1, 5, 7, 3)) * -1.0
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t1.aa, (4, 5), t1.aa, (6, 7), (0, 2, 4, 6, 1, 7, 5, 3))
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t1.aa, (4, 5), t1.aa, (6, 7), (0, 2, 4, 6, 3, 1, 5, 7))
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t1.aa, (4, 5), t1.aa, (6, 7), (0, 2, 4, 6, 3, 1, 7, 5)) * -1.0
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t1.aa, (4, 5), t1.aa, (6, 7), (0, 2, 4, 6, 5, 1, 3, 7)) * -1.0
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t1.aa, (4, 5), t1.aa, (6, 7), (0, 2, 4, 6, 7, 1, 3, 5))
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t1.aa, (4, 5), t1.aa, (6, 7), (0, 2, 4, 6, 5, 1, 7, 3))
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t1.aa, (4, 5), t1.aa, (6, 7), (0, 2, 4, 6, 7, 1, 5, 3)) * -1.0
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t1.aa, (4, 5), t1.aa, (6, 7), (0, 2, 4, 6, 3, 5, 1, 7)) * -1.0
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t1.aa, (4, 5), t1.aa, (6, 7), (0, 2, 4, 6, 3, 7, 1, 5))
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t1.aa, (4, 5), t1.aa, (6, 7), (0, 2, 4, 6, 5, 3, 1, 7))
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t1.aa, (4, 5), t1.aa, (6, 7), (0, 2, 4, 6, 7, 3, 1, 5)) * -1.0
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t1.aa, (4, 5), t1.aa, (6, 7), (0, 2, 4, 6, 5, 7, 1, 3)) * -1.0
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t1.aa, (4, 5), t1.aa, (6, 7), (0, 2, 4, 6, 7, 5, 1, 3))
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t1.aa, (4, 5), t1.aa, (6, 7), (0, 2, 4, 6, 3, 5, 7, 1))
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t1.aa, (4, 5), t1.aa, (6, 7), (0, 2, 4, 6, 3, 7, 5, 1)) * -1.0
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t1.aa, (4, 5), t1.aa, (6, 7), (0, 2, 4, 6, 5, 3, 7, 1)) * -1.0
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t1.aa, (4, 5), t1.aa, (6, 7), (0, 2, 4, 6, 7, 3, 5, 1))
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t1.aa, (4, 5), t1.aa, (6, 7), (0, 2, 4, 6, 5, 7, 3, 1))
    t4_aaaaaaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t1.aa, (4, 5), t1.aa, (6, 7), (0, 2, 4, 6, 7, 5, 3, 1)) * -1.0
    return t4_aaaaaaaa

def t4_uhf_aaabaaab(c4=None, t1=None, t2=None, t3=None, nocc=None, nvir=None):
    t4_aaabaaab = np.zeros((nocc[0], nocc[0], nocc[0], nocc[1], nvir[0], nvir[0], nvir[0], nvir[1]), dtype=np.float64)
    t4_aaabaaab += einsum(c4.aaabaaab, (0, 1, 2, 3, 4, 5, 6, 7), (0, 1, 2, 3, 4, 5, 6, 7))
    t4_aaabaaab += einsum(t1.aa, (0, 1), t3.abaaba, (2, 3, 4, 5, 6, 7), (0, 2, 4, 3, 1, 5, 7, 6)) * -1.0
    t4_aaabaaab += einsum(t1.aa, (0, 1), t3.abaaba, (2, 3, 4, 5, 6, 7), (0, 2, 4, 3, 5, 1, 7, 6))
    t4_aaabaaab += einsum(t1.aa, (0, 1), t3.abaaba, (2, 3, 4, 5, 6, 7), (0, 2, 4, 3, 5, 7, 1, 6)) * -1.0
    t4_aaabaaab += einsum(t1.aa, (0, 1), t3.abaaba, (2, 3, 4, 5, 6, 7), (2, 0, 4, 3, 1, 5, 7, 6))
    t4_aaabaaab += einsum(t1.aa, (0, 1), t3.abaaba, (2, 3, 4, 5, 6, 7), (2, 0, 4, 3, 5, 1, 7, 6)) * -1.0
    t4_aaabaaab += einsum(t1.aa, (0, 1), t3.abaaba, (2, 3, 4, 5, 6, 7), (2, 0, 4, 3, 5, 7, 1, 6))
    t4_aaabaaab += einsum(t1.aa, (0, 1), t3.abaaba, (2, 3, 4, 5, 6, 7), (2, 4, 0, 3, 1, 5, 7, 6)) * -1.0
    t4_aaabaaab += einsum(t1.aa, (0, 1), t3.abaaba, (2, 3, 4, 5, 6, 7), (2, 4, 0, 3, 5, 1, 7, 6))
    t4_aaabaaab += einsum(t1.aa, (0, 1), t3.abaaba, (2, 3, 4, 5, 6, 7), (2, 4, 0, 3, 5, 7, 1, 6)) * -1.0
    t4_aaabaaab += einsum(t1.bb, (0, 1), t3.aaaaaa, (2, 3, 4, 5, 6, 7), (2, 3, 4, 0, 5, 6, 7, 1)) * -1.0
    t4_aaabaaab += einsum(t2.abab, (0, 1, 2, 3), t2.aaaa, (4, 5, 6, 7), (0, 4, 5, 1, 2, 6, 7, 3)) * -1.0
    t4_aaabaaab += einsum(t2.abab, (0, 1, 2, 3), t2.aaaa, (4, 5, 6, 7), (0, 4, 5, 1, 6, 2, 7, 3))
    t4_aaabaaab += einsum(t2.abab, (0, 1, 2, 3), t2.aaaa, (4, 5, 6, 7), (0, 4, 5, 1, 6, 7, 2, 3)) * -1.0
    t4_aaabaaab += einsum(t2.abab, (0, 1, 2, 3), t2.aaaa, (4, 5, 6, 7), (4, 0, 5, 1, 2, 6, 7, 3))
    t4_aaabaaab += einsum(t2.abab, (0, 1, 2, 3), t2.aaaa, (4, 5, 6, 7), (4, 0, 5, 1, 6, 2, 7, 3)) * -1.0
    t4_aaabaaab += einsum(t2.abab, (0, 1, 2, 3), t2.aaaa, (4, 5, 6, 7), (4, 0, 5, 1, 6, 7, 2, 3))
    t4_aaabaaab += einsum(t2.abab, (0, 1, 2, 3), t2.aaaa, (4, 5, 6, 7), (4, 5, 0, 1, 2, 6, 7, 3)) * -1.0
    t4_aaabaaab += einsum(t2.abab, (0, 1, 2, 3), t2.aaaa, (4, 5, 6, 7), (4, 5, 0, 1, 6, 2, 7, 3))
    t4_aaabaaab += einsum(t2.abab, (0, 1, 2, 3), t2.aaaa, (4, 5, 6, 7), (4, 5, 0, 1, 6, 7, 2, 3)) * -1.0
    t4_aaabaaab += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.abab, (4, 5, 6, 7), (0, 2, 4, 5, 1, 3, 6, 7)) * -1.0
    t4_aaabaaab += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.abab, (4, 5, 6, 7), (0, 2, 4, 5, 1, 6, 3, 7))
    t4_aaabaaab += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.abab, (4, 5, 6, 7), (0, 2, 4, 5, 3, 1, 6, 7))
    t4_aaabaaab += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.abab, (4, 5, 6, 7), (0, 2, 4, 5, 6, 1, 3, 7)) * -1.0
    t4_aaabaaab += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.abab, (4, 5, 6, 7), (0, 2, 4, 5, 3, 6, 1, 7)) * -1.0
    t4_aaabaaab += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.abab, (4, 5, 6, 7), (0, 2, 4, 5, 6, 3, 1, 7))
    t4_aaabaaab += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.abab, (4, 5, 6, 7), (0, 4, 2, 5, 1, 3, 6, 7))
    t4_aaabaaab += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.abab, (4, 5, 6, 7), (0, 4, 2, 5, 1, 6, 3, 7)) * -1.0
    t4_aaabaaab += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.abab, (4, 5, 6, 7), (0, 4, 2, 5, 3, 1, 6, 7)) * -1.0
    t4_aaabaaab += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.abab, (4, 5, 6, 7), (0, 4, 2, 5, 6, 1, 3, 7))
    t4_aaabaaab += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.abab, (4, 5, 6, 7), (0, 4, 2, 5, 3, 6, 1, 7))
    t4_aaabaaab += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.abab, (4, 5, 6, 7), (0, 4, 2, 5, 6, 3, 1, 7)) * -1.0
    t4_aaabaaab += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.abab, (4, 5, 6, 7), (4, 0, 2, 5, 1, 3, 6, 7)) * -1.0
    t4_aaabaaab += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.abab, (4, 5, 6, 7), (4, 0, 2, 5, 1, 6, 3, 7))
    t4_aaabaaab += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.abab, (4, 5, 6, 7), (4, 0, 2, 5, 3, 1, 6, 7))
    t4_aaabaaab += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.abab, (4, 5, 6, 7), (4, 0, 2, 5, 6, 1, 3, 7)) * -1.0
    t4_aaabaaab += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.abab, (4, 5, 6, 7), (4, 0, 2, 5, 3, 6, 1, 7)) * -1.0
    t4_aaabaaab += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.abab, (4, 5, 6, 7), (4, 0, 2, 5, 6, 3, 1, 7))
    t4_aaabaaab += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.aaaa, (4, 5, 6, 7), (0, 4, 5, 2, 1, 6, 7, 3)) * -1.0
    t4_aaabaaab += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.aaaa, (4, 5, 6, 7), (0, 4, 5, 2, 6, 1, 7, 3))
    t4_aaabaaab += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.aaaa, (4, 5, 6, 7), (0, 4, 5, 2, 6, 7, 1, 3)) * -1.0
    t4_aaabaaab += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.aaaa, (4, 5, 6, 7), (4, 0, 5, 2, 1, 6, 7, 3))
    t4_aaabaaab += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.aaaa, (4, 5, 6, 7), (4, 0, 5, 2, 6, 1, 7, 3)) * -1.0
    t4_aaabaaab += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.aaaa, (4, 5, 6, 7), (4, 0, 5, 2, 6, 7, 1, 3))
    t4_aaabaaab += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.aaaa, (4, 5, 6, 7), (4, 5, 0, 2, 1, 6, 7, 3)) * -1.0
    t4_aaabaaab += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.aaaa, (4, 5, 6, 7), (4, 5, 0, 2, 6, 1, 7, 3))
    t4_aaabaaab += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.aaaa, (4, 5, 6, 7), (4, 5, 0, 2, 6, 7, 1, 3)) * -1.0
    t4_aaabaaab += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t1.aa, (4, 5), t1.bb, (6, 7), (0, 2, 4, 6, 1, 3, 5, 7)) * -1.0
    t4_aaabaaab += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t1.aa, (4, 5), t1.bb, (6, 7), (0, 2, 4, 6, 1, 5, 3, 7))
    t4_aaabaaab += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t1.aa, (4, 5), t1.bb, (6, 7), (0, 2, 4, 6, 3, 1, 5, 7))
    t4_aaabaaab += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t1.aa, (4, 5), t1.bb, (6, 7), (0, 2, 4, 6, 5, 1, 3, 7)) * -1.0
    t4_aaabaaab += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t1.aa, (4, 5), t1.bb, (6, 7), (0, 2, 4, 6, 3, 5, 1, 7)) * -1.0
    t4_aaabaaab += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t1.aa, (4, 5), t1.bb, (6, 7), (0, 2, 4, 6, 5, 3, 1, 7))
    return t4_aaabaaab

def t4_uhf_aabaaaba(c4=None, t1=None, t2=None, t3=None, nocc=None, nvir=None):
    t4_aabaaaba = np.zeros((nocc[0], nocc[0], nocc[1], nocc[0], nvir[0], nvir[0], nvir[1], nvir[0]), dtype=np.float64)
    t4_aabaaaba += einsum(c4.aabaaaba, (0, 1, 2, 3, 4, 5, 6, 7), (0, 1, 2, 3, 4, 5, 6, 7))
    t4_aabaaaba += einsum(t1.aa, (0, 1), t3.abaaba, (2, 3, 4, 5, 6, 7), (0, 2, 3, 4, 1, 5, 6, 7)) * -1.0
    t4_aabaaaba += einsum(t1.aa, (0, 1), t3.abaaba, (2, 3, 4, 5, 6, 7), (0, 2, 3, 4, 5, 1, 6, 7))
    t4_aabaaaba += einsum(t1.aa, (0, 1), t3.abaaba, (2, 3, 4, 5, 6, 7), (0, 2, 3, 4, 5, 7, 6, 1)) * -1.0
    t4_aabaaaba += einsum(t1.aa, (0, 1), t3.abaaba, (2, 3, 4, 5, 6, 7), (2, 0, 3, 4, 1, 5, 6, 7))
    t4_aabaaaba += einsum(t1.aa, (0, 1), t3.abaaba, (2, 3, 4, 5, 6, 7), (2, 0, 3, 4, 5, 1, 6, 7)) * -1.0
    t4_aabaaaba += einsum(t1.aa, (0, 1), t3.abaaba, (2, 3, 4, 5, 6, 7), (2, 0, 3, 4, 5, 7, 6, 1))
    t4_aabaaaba += einsum(t1.aa, (0, 1), t3.abaaba, (2, 3, 4, 5, 6, 7), (2, 4, 3, 0, 1, 5, 6, 7)) * -1.0
    t4_aabaaaba += einsum(t1.aa, (0, 1), t3.abaaba, (2, 3, 4, 5, 6, 7), (2, 4, 3, 0, 5, 1, 6, 7))
    t4_aabaaaba += einsum(t1.aa, (0, 1), t3.abaaba, (2, 3, 4, 5, 6, 7), (2, 4, 3, 0, 5, 7, 6, 1)) * -1.0
    t4_aabaaaba += einsum(t1.bb, (0, 1), t3.aaaaaa, (2, 3, 4, 5, 6, 7), (2, 3, 0, 4, 5, 6, 1, 7)) * -1.0
    t4_aabaaaba += einsum(t2.abab, (0, 1, 2, 3), t2.aaaa, (4, 5, 6, 7), (0, 4, 1, 5, 2, 6, 3, 7)) * -1.0
    t4_aabaaaba += einsum(t2.abab, (0, 1, 2, 3), t2.aaaa, (4, 5, 6, 7), (0, 4, 1, 5, 6, 2, 3, 7))
    t4_aabaaaba += einsum(t2.abab, (0, 1, 2, 3), t2.aaaa, (4, 5, 6, 7), (0, 4, 1, 5, 6, 7, 3, 2)) * -1.0
    t4_aabaaaba += einsum(t2.abab, (0, 1, 2, 3), t2.aaaa, (4, 5, 6, 7), (4, 0, 1, 5, 2, 6, 3, 7))
    t4_aabaaaba += einsum(t2.abab, (0, 1, 2, 3), t2.aaaa, (4, 5, 6, 7), (4, 0, 1, 5, 6, 2, 3, 7)) * -1.0
    t4_aabaaaba += einsum(t2.abab, (0, 1, 2, 3), t2.aaaa, (4, 5, 6, 7), (4, 0, 1, 5, 6, 7, 3, 2))
    t4_aabaaaba += einsum(t2.abab, (0, 1, 2, 3), t2.aaaa, (4, 5, 6, 7), (4, 5, 1, 0, 2, 6, 3, 7)) * -1.0
    t4_aabaaaba += einsum(t2.abab, (0, 1, 2, 3), t2.aaaa, (4, 5, 6, 7), (4, 5, 1, 0, 6, 2, 3, 7))
    t4_aabaaaba += einsum(t2.abab, (0, 1, 2, 3), t2.aaaa, (4, 5, 6, 7), (4, 5, 1, 0, 6, 7, 3, 2)) * -1.0
    t4_aabaaaba += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.abab, (4, 5, 6, 7), (0, 2, 5, 4, 1, 3, 7, 6)) * -1.0
    t4_aabaaaba += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.abab, (4, 5, 6, 7), (0, 2, 5, 4, 1, 6, 7, 3))
    t4_aabaaaba += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.abab, (4, 5, 6, 7), (0, 2, 5, 4, 3, 1, 7, 6))
    t4_aabaaaba += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.abab, (4, 5, 6, 7), (0, 2, 5, 4, 6, 1, 7, 3)) * -1.0
    t4_aabaaaba += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.abab, (4, 5, 6, 7), (0, 2, 5, 4, 3, 6, 7, 1)) * -1.0
    t4_aabaaaba += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.abab, (4, 5, 6, 7), (0, 2, 5, 4, 6, 3, 7, 1))
    t4_aabaaaba += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.abab, (4, 5, 6, 7), (0, 4, 5, 2, 1, 3, 7, 6))
    t4_aabaaaba += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.abab, (4, 5, 6, 7), (0, 4, 5, 2, 1, 6, 7, 3)) * -1.0
    t4_aabaaaba += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.abab, (4, 5, 6, 7), (0, 4, 5, 2, 3, 1, 7, 6)) * -1.0
    t4_aabaaaba += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.abab, (4, 5, 6, 7), (0, 4, 5, 2, 6, 1, 7, 3))
    t4_aabaaaba += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.abab, (4, 5, 6, 7), (0, 4, 5, 2, 3, 6, 7, 1))
    t4_aabaaaba += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.abab, (4, 5, 6, 7), (0, 4, 5, 2, 6, 3, 7, 1)) * -1.0
    t4_aabaaaba += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.abab, (4, 5, 6, 7), (4, 0, 5, 2, 1, 3, 7, 6)) * -1.0
    t4_aabaaaba += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.abab, (4, 5, 6, 7), (4, 0, 5, 2, 1, 6, 7, 3))
    t4_aabaaaba += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.abab, (4, 5, 6, 7), (4, 0, 5, 2, 3, 1, 7, 6))
    t4_aabaaaba += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.abab, (4, 5, 6, 7), (4, 0, 5, 2, 6, 1, 7, 3)) * -1.0
    t4_aabaaaba += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.abab, (4, 5, 6, 7), (4, 0, 5, 2, 3, 6, 7, 1)) * -1.0
    t4_aabaaaba += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.abab, (4, 5, 6, 7), (4, 0, 5, 2, 6, 3, 7, 1))
    t4_aabaaaba += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.aaaa, (4, 5, 6, 7), (0, 4, 2, 5, 1, 6, 3, 7)) * -1.0
    t4_aabaaaba += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.aaaa, (4, 5, 6, 7), (0, 4, 2, 5, 6, 1, 3, 7))
    t4_aabaaaba += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.aaaa, (4, 5, 6, 7), (0, 4, 2, 5, 6, 7, 3, 1)) * -1.0
    t4_aabaaaba += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.aaaa, (4, 5, 6, 7), (4, 0, 2, 5, 1, 6, 3, 7))
    t4_aabaaaba += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.aaaa, (4, 5, 6, 7), (4, 0, 2, 5, 6, 1, 3, 7)) * -1.0
    t4_aabaaaba += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.aaaa, (4, 5, 6, 7), (4, 0, 2, 5, 6, 7, 3, 1))
    t4_aabaaaba += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.aaaa, (4, 5, 6, 7), (4, 5, 2, 0, 1, 6, 3, 7)) * -1.0
    t4_aabaaaba += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.aaaa, (4, 5, 6, 7), (4, 5, 2, 0, 6, 1, 3, 7))
    t4_aabaaaba += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.aaaa, (4, 5, 6, 7), (4, 5, 2, 0, 6, 7, 3, 1)) * -1.0
    t4_aabaaaba += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t1.aa, (4, 5), t1.bb, (6, 7), (0, 2, 6, 4, 1, 3, 7, 5)) * -1.0
    t4_aabaaaba += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t1.aa, (4, 5), t1.bb, (6, 7), (0, 2, 6, 4, 1, 5, 7, 3))
    t4_aabaaaba += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t1.aa, (4, 5), t1.bb, (6, 7), (0, 2, 6, 4, 3, 1, 7, 5))
    t4_aabaaaba += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t1.aa, (4, 5), t1.bb, (6, 7), (0, 2, 6, 4, 5, 1, 7, 3)) * -1.0
    t4_aabaaaba += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t1.aa, (4, 5), t1.bb, (6, 7), (0, 2, 6, 4, 3, 5, 7, 1)) * -1.0
    t4_aabaaaba += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t1.aa, (4, 5), t1.bb, (6, 7), (0, 2, 6, 4, 5, 3, 7, 1))
    return t4_aabaaaba

def t4_uhf_aabbaabb(c4=None, t1=None, t2=None, t3=None, nocc=None, nvir=None):
    t4_aabbaabb = np.zeros((nocc[0], nocc[0], nocc[1], nocc[1], nvir[0], nvir[0], nvir[1], nvir[1]), dtype=np.float64)
    t4_aabbaabb += einsum(c4.abababab, (0, 1, 2, 3, 4, 5, 6, 7), (0, 2, 1, 3, 4, 6, 5, 7))
    t4_aabbaabb += einsum(t1.aa, (0, 1), t3.abbabb, (2, 3, 4, 5, 6, 7), (0, 2, 3, 4, 1, 5, 6, 7)) * -1.0
    t4_aabbaabb += einsum(t1.aa, (0, 1), t3.abbabb, (2, 3, 4, 5, 6, 7), (0, 2, 3, 4, 5, 1, 6, 7))
    t4_aabbaabb += einsum(t1.aa, (0, 1), t3.abbabb, (2, 3, 4, 5, 6, 7), (2, 0, 3, 4, 1, 5, 6, 7))
    t4_aabbaabb += einsum(t1.aa, (0, 1), t3.abbabb, (2, 3, 4, 5, 6, 7), (2, 0, 3, 4, 5, 1, 6, 7)) * -1.0
    t4_aabbaabb += einsum(t1.bb, (0, 1), t3.abaaba, (2, 3, 4, 5, 6, 7), (2, 4, 0, 3, 5, 7, 1, 6)) * -1.0
    t4_aabbaabb += einsum(t1.bb, (0, 1), t3.abaaba, (2, 3, 4, 5, 6, 7), (2, 4, 0, 3, 5, 7, 6, 1))
    t4_aabbaabb += einsum(t1.bb, (0, 1), t3.abaaba, (2, 3, 4, 5, 6, 7), (2, 4, 3, 0, 5, 7, 1, 6))
    t4_aabbaabb += einsum(t1.bb, (0, 1), t3.abaaba, (2, 3, 4, 5, 6, 7), (2, 4, 3, 0, 5, 7, 6, 1)) * -1.0
    t4_aabbaabb += einsum(t2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 6, 7), (0, 4, 1, 5, 2, 6, 3, 7)) * -1.0
    t4_aabbaabb += einsum(t2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 6, 7), (0, 4, 1, 5, 2, 6, 7, 3))
    t4_aabbaabb += einsum(t2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 6, 7), (0, 4, 1, 5, 6, 2, 3, 7))
    t4_aabbaabb += einsum(t2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 6, 7), (0, 4, 1, 5, 6, 2, 7, 3)) * -1.0
    t4_aabbaabb += einsum(t2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 6, 7), (0, 4, 5, 1, 2, 6, 3, 7))
    t4_aabbaabb += einsum(t2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 6, 7), (0, 4, 5, 1, 2, 6, 7, 3)) * -1.0
    t4_aabbaabb += einsum(t2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 6, 7), (0, 4, 5, 1, 6, 2, 3, 7)) * -1.0
    t4_aabbaabb += einsum(t2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 6, 7), (0, 4, 5, 1, 6, 2, 7, 3))
    t4_aabbaabb += einsum(t2.aaaa, (0, 1, 2, 3), t2.bbbb, (4, 5, 6, 7), (0, 1, 4, 5, 2, 3, 6, 7)) * -1.0
    t4_aabbaabb += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.bbbb, (4, 5, 6, 7), (0, 2, 4, 5, 1, 3, 6, 7)) * -1.0
    t4_aabbaabb += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.bbbb, (4, 5, 6, 7), (0, 2, 4, 5, 3, 1, 6, 7))
    t4_aabbaabb += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (0, 4, 2, 5, 1, 6, 3, 7)) * -1.0
    t4_aabbaabb += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (0, 4, 2, 5, 1, 6, 7, 3))
    t4_aabbaabb += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (0, 4, 2, 5, 6, 1, 3, 7))
    t4_aabbaabb += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (0, 4, 2, 5, 6, 1, 7, 3)) * -1.0
    t4_aabbaabb += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (0, 4, 5, 2, 1, 6, 3, 7))
    t4_aabbaabb += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (0, 4, 5, 2, 1, 6, 7, 3)) * -1.0
    t4_aabbaabb += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (0, 4, 5, 2, 6, 1, 3, 7)) * -1.0
    t4_aabbaabb += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (0, 4, 5, 2, 6, 1, 7, 3))
    t4_aabbaabb += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (4, 0, 2, 5, 1, 6, 3, 7))
    t4_aabbaabb += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (4, 0, 2, 5, 1, 6, 7, 3)) * -1.0
    t4_aabbaabb += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (4, 0, 2, 5, 6, 1, 3, 7)) * -1.0
    t4_aabbaabb += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (4, 0, 2, 5, 6, 1, 7, 3))
    t4_aabbaabb += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (4, 0, 5, 2, 1, 6, 3, 7)) * -1.0
    t4_aabbaabb += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (4, 0, 5, 2, 1, 6, 7, 3))
    t4_aabbaabb += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (4, 0, 5, 2, 6, 1, 3, 7))
    t4_aabbaabb += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (4, 0, 5, 2, 6, 1, 7, 3)) * -1.0
    t4_aabbaabb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.aaaa, (4, 5, 6, 7), (4, 5, 0, 2, 6, 7, 1, 3)) * -1.0
    t4_aabbaabb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.aaaa, (4, 5, 6, 7), (4, 5, 0, 2, 6, 7, 3, 1))
    t4_aabbaabb += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t1.bb, (4, 5), t1.bb, (6, 7), (0, 2, 4, 6, 1, 3, 5, 7)) * -1.0
    t4_aabbaabb += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t1.bb, (4, 5), t1.bb, (6, 7), (0, 2, 4, 6, 1, 3, 7, 5))
    t4_aabbaabb += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t1.bb, (4, 5), t1.bb, (6, 7), (0, 2, 4, 6, 3, 1, 5, 7))
    t4_aabbaabb += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t1.bb, (4, 5), t1.bb, (6, 7), (0, 2, 4, 6, 3, 1, 7, 5)) * -1.0
    return t4_aabbaabb

def t4_uhf_abaaabaa(c4=None, t1=None, t2=None, t3=None, nocc=None, nvir=None):
    t4_abaaabaa = np.zeros((nocc[0], nocc[1], nocc[0], nocc[0], nvir[0], nvir[1], nvir[0], nvir[0]), dtype=np.float64)
    t4_abaaabaa += einsum(c4.abaaabaa, (0, 1, 2, 3, 4, 5, 6, 7), (0, 1, 2, 3, 4, 5, 6, 7))
    t4_abaaabaa += einsum(t1.aa, (0, 1), t3.abaaba, (2, 3, 4, 5, 6, 7), (0, 3, 2, 4, 1, 6, 5, 7)) * -1.0
    t4_abaaabaa += einsum(t1.aa, (0, 1), t3.abaaba, (2, 3, 4, 5, 6, 7), (0, 3, 2, 4, 5, 6, 1, 7))
    t4_abaaabaa += einsum(t1.aa, (0, 1), t3.abaaba, (2, 3, 4, 5, 6, 7), (0, 3, 2, 4, 5, 6, 7, 1)) * -1.0
    t4_abaaabaa += einsum(t1.aa, (0, 1), t3.abaaba, (2, 3, 4, 5, 6, 7), (2, 3, 0, 4, 1, 6, 5, 7))
    t4_abaaabaa += einsum(t1.aa, (0, 1), t3.abaaba, (2, 3, 4, 5, 6, 7), (2, 3, 0, 4, 5, 6, 1, 7)) * -1.0
    t4_abaaabaa += einsum(t1.aa, (0, 1), t3.abaaba, (2, 3, 4, 5, 6, 7), (2, 3, 0, 4, 5, 6, 7, 1))
    t4_abaaabaa += einsum(t1.aa, (0, 1), t3.abaaba, (2, 3, 4, 5, 6, 7), (2, 3, 4, 0, 1, 6, 5, 7)) * -1.0
    t4_abaaabaa += einsum(t1.aa, (0, 1), t3.abaaba, (2, 3, 4, 5, 6, 7), (2, 3, 4, 0, 5, 6, 1, 7))
    t4_abaaabaa += einsum(t1.aa, (0, 1), t3.abaaba, (2, 3, 4, 5, 6, 7), (2, 3, 4, 0, 5, 6, 7, 1)) * -1.0
    t4_abaaabaa += einsum(t1.bb, (0, 1), t3.aaaaaa, (2, 3, 4, 5, 6, 7), (2, 0, 3, 4, 5, 1, 6, 7)) * -1.0
    t4_abaaabaa += einsum(t2.abab, (0, 1, 2, 3), t2.aaaa, (4, 5, 6, 7), (0, 1, 4, 5, 2, 3, 6, 7)) * -1.0
    t4_abaaabaa += einsum(t2.abab, (0, 1, 2, 3), t2.aaaa, (4, 5, 6, 7), (0, 1, 4, 5, 6, 3, 2, 7))
    t4_abaaabaa += einsum(t2.abab, (0, 1, 2, 3), t2.aaaa, (4, 5, 6, 7), (0, 1, 4, 5, 6, 3, 7, 2)) * -1.0
    t4_abaaabaa += einsum(t2.abab, (0, 1, 2, 3), t2.aaaa, (4, 5, 6, 7), (4, 1, 0, 5, 2, 3, 6, 7))
    t4_abaaabaa += einsum(t2.abab, (0, 1, 2, 3), t2.aaaa, (4, 5, 6, 7), (4, 1, 0, 5, 6, 3, 2, 7)) * -1.0
    t4_abaaabaa += einsum(t2.abab, (0, 1, 2, 3), t2.aaaa, (4, 5, 6, 7), (4, 1, 0, 5, 6, 3, 7, 2))
    t4_abaaabaa += einsum(t2.abab, (0, 1, 2, 3), t2.aaaa, (4, 5, 6, 7), (4, 1, 5, 0, 2, 3, 6, 7)) * -1.0
    t4_abaaabaa += einsum(t2.abab, (0, 1, 2, 3), t2.aaaa, (4, 5, 6, 7), (4, 1, 5, 0, 6, 3, 2, 7))
    t4_abaaabaa += einsum(t2.abab, (0, 1, 2, 3), t2.aaaa, (4, 5, 6, 7), (4, 1, 5, 0, 6, 3, 7, 2)) * -1.0
    t4_abaaabaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.abab, (4, 5, 6, 7), (0, 5, 2, 4, 1, 7, 3, 6)) * -1.0
    t4_abaaabaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.abab, (4, 5, 6, 7), (0, 5, 2, 4, 1, 7, 6, 3))
    t4_abaaabaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.abab, (4, 5, 6, 7), (0, 5, 2, 4, 3, 7, 1, 6))
    t4_abaaabaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.abab, (4, 5, 6, 7), (0, 5, 2, 4, 6, 7, 1, 3)) * -1.0
    t4_abaaabaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.abab, (4, 5, 6, 7), (0, 5, 2, 4, 3, 7, 6, 1)) * -1.0
    t4_abaaabaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.abab, (4, 5, 6, 7), (0, 5, 2, 4, 6, 7, 3, 1))
    t4_abaaabaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.abab, (4, 5, 6, 7), (0, 5, 4, 2, 1, 7, 3, 6))
    t4_abaaabaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.abab, (4, 5, 6, 7), (0, 5, 4, 2, 1, 7, 6, 3)) * -1.0
    t4_abaaabaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.abab, (4, 5, 6, 7), (0, 5, 4, 2, 3, 7, 1, 6)) * -1.0
    t4_abaaabaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.abab, (4, 5, 6, 7), (0, 5, 4, 2, 6, 7, 1, 3))
    t4_abaaabaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.abab, (4, 5, 6, 7), (0, 5, 4, 2, 3, 7, 6, 1))
    t4_abaaabaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.abab, (4, 5, 6, 7), (0, 5, 4, 2, 6, 7, 3, 1)) * -1.0
    t4_abaaabaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.abab, (4, 5, 6, 7), (4, 5, 0, 2, 1, 7, 3, 6)) * -1.0
    t4_abaaabaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.abab, (4, 5, 6, 7), (4, 5, 0, 2, 1, 7, 6, 3))
    t4_abaaabaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.abab, (4, 5, 6, 7), (4, 5, 0, 2, 3, 7, 1, 6))
    t4_abaaabaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.abab, (4, 5, 6, 7), (4, 5, 0, 2, 6, 7, 1, 3)) * -1.0
    t4_abaaabaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.abab, (4, 5, 6, 7), (4, 5, 0, 2, 3, 7, 6, 1)) * -1.0
    t4_abaaabaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.abab, (4, 5, 6, 7), (4, 5, 0, 2, 6, 7, 3, 1))
    t4_abaaabaa += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.aaaa, (4, 5, 6, 7), (0, 2, 4, 5, 1, 3, 6, 7)) * -1.0
    t4_abaaabaa += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.aaaa, (4, 5, 6, 7), (0, 2, 4, 5, 6, 3, 1, 7))
    t4_abaaabaa += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.aaaa, (4, 5, 6, 7), (0, 2, 4, 5, 6, 3, 7, 1)) * -1.0
    t4_abaaabaa += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.aaaa, (4, 5, 6, 7), (4, 2, 0, 5, 1, 3, 6, 7))
    t4_abaaabaa += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.aaaa, (4, 5, 6, 7), (4, 2, 0, 5, 6, 3, 1, 7)) * -1.0
    t4_abaaabaa += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.aaaa, (4, 5, 6, 7), (4, 2, 0, 5, 6, 3, 7, 1))
    t4_abaaabaa += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.aaaa, (4, 5, 6, 7), (4, 2, 5, 0, 1, 3, 6, 7)) * -1.0
    t4_abaaabaa += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.aaaa, (4, 5, 6, 7), (4, 2, 5, 0, 6, 3, 1, 7))
    t4_abaaabaa += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.aaaa, (4, 5, 6, 7), (4, 2, 5, 0, 6, 3, 7, 1)) * -1.0
    t4_abaaabaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t1.aa, (4, 5), t1.bb, (6, 7), (0, 6, 2, 4, 1, 7, 3, 5)) * -1.0
    t4_abaaabaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t1.aa, (4, 5), t1.bb, (6, 7), (0, 6, 2, 4, 1, 7, 5, 3))
    t4_abaaabaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t1.aa, (4, 5), t1.bb, (6, 7), (0, 6, 2, 4, 3, 7, 1, 5))
    t4_abaaabaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t1.aa, (4, 5), t1.bb, (6, 7), (0, 6, 2, 4, 5, 7, 1, 3)) * -1.0
    t4_abaaabaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t1.aa, (4, 5), t1.bb, (6, 7), (0, 6, 2, 4, 3, 7, 5, 1)) * -1.0
    t4_abaaabaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t1.aa, (4, 5), t1.bb, (6, 7), (0, 6, 2, 4, 5, 7, 3, 1))
    return t4_abaaabaa

def t4_uhf_abababab(c4=None, t1=None, t2=None, t3=None, nocc=None, nvir=None):
    t4_abababab = np.zeros((nocc[0], nocc[1], nocc[0], nocc[1], nvir[0], nvir[1], nvir[0], nvir[1]), dtype=np.float64)
    t4_abababab += einsum(c4.abababab, (0, 1, 2, 3, 4, 5, 6, 7), (0, 1, 2, 3, 4, 5, 6, 7))
    t4_abababab += einsum(t1.aa, (0, 1), t3.babbab, (2, 3, 4, 5, 6, 7), (0, 2, 3, 4, 1, 5, 6, 7)) * -1.0
    t4_abababab += einsum(t1.aa, (0, 1), t3.babbab, (2, 3, 4, 5, 6, 7), (0, 2, 3, 4, 6, 5, 1, 7))
    t4_abababab += einsum(t1.aa, (0, 1), t3.abbabb, (2, 3, 4, 5, 6, 7), (2, 3, 0, 4, 1, 6, 5, 7))
    t4_abababab += einsum(t1.aa, (0, 1), t3.abbabb, (2, 3, 4, 5, 6, 7), (2, 3, 0, 4, 5, 6, 1, 7)) * -1.0
    t4_abababab += einsum(t1.bb, (0, 1), t3.abaaba, (2, 3, 4, 5, 6, 7), (2, 0, 4, 3, 5, 1, 7, 6)) * -1.0
    t4_abababab += einsum(t1.bb, (0, 1), t3.abaaba, (2, 3, 4, 5, 6, 7), (2, 0, 4, 3, 5, 6, 7, 1))
    t4_abababab += einsum(t1.bb, (0, 1), t3.abaaba, (2, 3, 4, 5, 6, 7), (2, 3, 4, 0, 5, 1, 7, 6))
    t4_abababab += einsum(t1.bb, (0, 1), t3.abaaba, (2, 3, 4, 5, 6, 7), (2, 3, 4, 0, 5, 6, 7, 1)) * -1.0
    t4_abababab += einsum(t2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 6, 7), (0, 1, 4, 5, 2, 3, 6, 7)) * -1.0
    t4_abababab += einsum(t2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 6, 7), (0, 1, 4, 5, 2, 7, 6, 3))
    t4_abababab += einsum(t2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 6, 7), (0, 1, 4, 5, 6, 3, 2, 7))
    t4_abababab += einsum(t2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 6, 7), (0, 1, 4, 5, 6, 7, 2, 3)) * -1.0
    t4_abababab += einsum(t2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 6, 7), (0, 5, 4, 1, 2, 3, 6, 7))
    t4_abababab += einsum(t2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 6, 7), (0, 5, 4, 1, 2, 7, 6, 3)) * -1.0
    t4_abababab += einsum(t2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 6, 7), (0, 5, 4, 1, 6, 3, 2, 7)) * -1.0
    t4_abababab += einsum(t2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 6, 7), (0, 5, 4, 1, 6, 7, 2, 3))
    t4_abababab += einsum(t2.aaaa, (0, 1, 2, 3), t2.bbbb, (4, 5, 6, 7), (0, 4, 1, 5, 2, 6, 3, 7)) * -1.0
    t4_abababab += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.bbbb, (4, 5, 6, 7), (0, 4, 2, 5, 1, 6, 3, 7)) * -1.0
    t4_abababab += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.bbbb, (4, 5, 6, 7), (0, 4, 2, 5, 3, 6, 1, 7))
    t4_abababab += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (0, 2, 4, 5, 1, 3, 6, 7)) * -1.0
    t4_abababab += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (0, 2, 4, 5, 1, 7, 6, 3))
    t4_abababab += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (0, 2, 4, 5, 6, 3, 1, 7))
    t4_abababab += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (0, 2, 4, 5, 6, 7, 1, 3)) * -1.0
    t4_abababab += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (0, 5, 4, 2, 1, 3, 6, 7))
    t4_abababab += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (0, 5, 4, 2, 1, 7, 6, 3)) * -1.0
    t4_abababab += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (0, 5, 4, 2, 6, 3, 1, 7)) * -1.0
    t4_abababab += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (0, 5, 4, 2, 6, 7, 1, 3))
    t4_abababab += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (4, 2, 0, 5, 1, 3, 6, 7))
    t4_abababab += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (4, 2, 0, 5, 1, 7, 6, 3)) * -1.0
    t4_abababab += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (4, 2, 0, 5, 6, 3, 1, 7)) * -1.0
    t4_abababab += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (4, 2, 0, 5, 6, 7, 1, 3))
    t4_abababab += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (4, 5, 0, 2, 1, 3, 6, 7)) * -1.0
    t4_abababab += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (4, 5, 0, 2, 1, 7, 6, 3))
    t4_abababab += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (4, 5, 0, 2, 6, 3, 1, 7))
    t4_abababab += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (4, 5, 0, 2, 6, 7, 1, 3)) * -1.0
    t4_abababab += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.aaaa, (4, 5, 6, 7), (4, 0, 5, 2, 6, 1, 7, 3)) * -1.0
    t4_abababab += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.aaaa, (4, 5, 6, 7), (4, 0, 5, 2, 6, 3, 7, 1))
    t4_abababab += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t1.bb, (4, 5), t1.bb, (6, 7), (0, 4, 2, 6, 1, 5, 3, 7)) * -1.0
    t4_abababab += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t1.bb, (4, 5), t1.bb, (6, 7), (0, 4, 2, 6, 1, 7, 3, 5))
    t4_abababab += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t1.bb, (4, 5), t1.bb, (6, 7), (0, 4, 2, 6, 3, 5, 1, 7))
    t4_abababab += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t1.bb, (4, 5), t1.bb, (6, 7), (0, 4, 2, 6, 3, 7, 1, 5)) * -1.0
    return t4_abababab

def t4_uhf_abbaabba(c4=None, t1=None, t2=None, t3=None, nocc=None, nvir=None):
    t4_abbaabba = np.zeros((nocc[0], nocc[1], nocc[1], nocc[0], nvir[0], nvir[1], nvir[1], nvir[0]), dtype=np.float64)
    t4_abbaabba += einsum(c4.abababab, (0, 1, 2, 3, 4, 5, 6, 7), (0, 1, 3, 2, 4, 5, 7, 6))
    t4_abbaabba += einsum(t1.aa, (0, 1), t3.bbabba, (2, 3, 4, 5, 6, 7), (0, 2, 3, 4, 1, 5, 6, 7)) * -1.0
    t4_abbaabba += einsum(t1.aa, (0, 1), t3.bbabba, (2, 3, 4, 5, 6, 7), (0, 2, 3, 4, 7, 5, 6, 1))
    t4_abbaabba += einsum(t1.aa, (0, 1), t3.abbabb, (2, 3, 4, 5, 6, 7), (2, 3, 4, 0, 1, 6, 7, 5))
    t4_abbaabba += einsum(t1.aa, (0, 1), t3.abbabb, (2, 3, 4, 5, 6, 7), (2, 3, 4, 0, 5, 6, 7, 1)) * -1.0
    t4_abbaabba += einsum(t1.bb, (0, 1), t3.abaaba, (2, 3, 4, 5, 6, 7), (2, 0, 3, 4, 5, 1, 6, 7)) * -1.0
    t4_abbaabba += einsum(t1.bb, (0, 1), t3.abaaba, (2, 3, 4, 5, 6, 7), (2, 0, 3, 4, 5, 6, 1, 7))
    t4_abbaabba += einsum(t1.bb, (0, 1), t3.abaaba, (2, 3, 4, 5, 6, 7), (2, 3, 0, 4, 5, 1, 6, 7))
    t4_abbaabba += einsum(t1.bb, (0, 1), t3.abaaba, (2, 3, 4, 5, 6, 7), (2, 3, 0, 4, 5, 6, 1, 7)) * -1.0
    t4_abbaabba += einsum(t2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 6, 7), (0, 1, 5, 4, 2, 3, 7, 6)) * -1.0
    t4_abbaabba += einsum(t2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 6, 7), (0, 1, 5, 4, 2, 7, 3, 6))
    t4_abbaabba += einsum(t2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 6, 7), (0, 1, 5, 4, 6, 3, 7, 2))
    t4_abbaabba += einsum(t2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 6, 7), (0, 1, 5, 4, 6, 7, 3, 2)) * -1.0
    t4_abbaabba += einsum(t2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 6, 7), (0, 5, 1, 4, 2, 3, 7, 6))
    t4_abbaabba += einsum(t2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 6, 7), (0, 5, 1, 4, 2, 7, 3, 6)) * -1.0
    t4_abbaabba += einsum(t2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 6, 7), (0, 5, 1, 4, 6, 3, 7, 2)) * -1.0
    t4_abbaabba += einsum(t2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 6, 7), (0, 5, 1, 4, 6, 7, 3, 2))
    t4_abbaabba += einsum(t2.aaaa, (0, 1, 2, 3), t2.bbbb, (4, 5, 6, 7), (0, 4, 5, 1, 2, 6, 7, 3)) * -1.0
    t4_abbaabba += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.bbbb, (4, 5, 6, 7), (0, 4, 5, 2, 1, 6, 7, 3)) * -1.0
    t4_abbaabba += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.bbbb, (4, 5, 6, 7), (0, 4, 5, 2, 3, 6, 7, 1))
    t4_abbaabba += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (0, 2, 5, 4, 1, 3, 7, 6)) * -1.0
    t4_abbaabba += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (0, 2, 5, 4, 1, 7, 3, 6))
    t4_abbaabba += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (0, 2, 5, 4, 6, 3, 7, 1))
    t4_abbaabba += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (0, 2, 5, 4, 6, 7, 3, 1)) * -1.0
    t4_abbaabba += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (0, 5, 2, 4, 1, 3, 7, 6))
    t4_abbaabba += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (0, 5, 2, 4, 1, 7, 3, 6)) * -1.0
    t4_abbaabba += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (0, 5, 2, 4, 6, 3, 7, 1)) * -1.0
    t4_abbaabba += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (0, 5, 2, 4, 6, 7, 3, 1))
    t4_abbaabba += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (4, 2, 5, 0, 1, 3, 7, 6))
    t4_abbaabba += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (4, 2, 5, 0, 1, 7, 3, 6)) * -1.0
    t4_abbaabba += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (4, 2, 5, 0, 6, 3, 7, 1)) * -1.0
    t4_abbaabba += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (4, 2, 5, 0, 6, 7, 3, 1))
    t4_abbaabba += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (4, 5, 2, 0, 1, 3, 7, 6)) * -1.0
    t4_abbaabba += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (4, 5, 2, 0, 1, 7, 3, 6))
    t4_abbaabba += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (4, 5, 2, 0, 6, 3, 7, 1))
    t4_abbaabba += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (4, 5, 2, 0, 6, 7, 3, 1)) * -1.0
    t4_abbaabba += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.aaaa, (4, 5, 6, 7), (4, 0, 2, 5, 6, 1, 3, 7)) * -1.0
    t4_abbaabba += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.aaaa, (4, 5, 6, 7), (4, 0, 2, 5, 6, 3, 1, 7))
    t4_abbaabba += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t1.bb, (4, 5), t1.bb, (6, 7), (0, 4, 6, 2, 1, 5, 7, 3)) * -1.0
    t4_abbaabba += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t1.bb, (4, 5), t1.bb, (6, 7), (0, 4, 6, 2, 1, 7, 5, 3))
    t4_abbaabba += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t1.bb, (4, 5), t1.bb, (6, 7), (0, 4, 6, 2, 3, 5, 7, 1))
    t4_abbaabba += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t1.bb, (4, 5), t1.bb, (6, 7), (0, 4, 6, 2, 3, 7, 5, 1)) * -1.0
    return t4_abbaabba

def t4_uhf_abbbabbb(c4=None, t1=None, t2=None, t3=None, nocc=None, nvir=None):
    t4_abbbabbb = np.zeros((nocc[0], nocc[1], nocc[1], nocc[1], nvir[0], nvir[1], nvir[1], nvir[1]), dtype=np.float64)
    t4_abbbabbb += einsum(c4.abbbabbb, (0, 1, 2, 3, 4, 5, 6, 7), (0, 1, 2, 3, 4, 5, 6, 7))
    t4_abbbabbb += einsum(t1.aa, (0, 1), t3.bbbbbb, (2, 3, 4, 5, 6, 7), (0, 2, 3, 4, 1, 5, 6, 7)) * -1.0
    t4_abbbabbb += einsum(t1.bb, (0, 1), t3.abbabb, (2, 3, 4, 5, 6, 7), (2, 0, 3, 4, 5, 1, 6, 7)) * -1.0
    t4_abbbabbb += einsum(t1.bb, (0, 1), t3.abbabb, (2, 3, 4, 5, 6, 7), (2, 0, 3, 4, 5, 6, 1, 7))
    t4_abbbabbb += einsum(t1.bb, (0, 1), t3.abbabb, (2, 3, 4, 5, 6, 7), (2, 0, 3, 4, 5, 6, 7, 1)) * -1.0
    t4_abbbabbb += einsum(t1.bb, (0, 1), t3.abbabb, (2, 3, 4, 5, 6, 7), (2, 3, 0, 4, 5, 1, 6, 7))
    t4_abbbabbb += einsum(t1.bb, (0, 1), t3.abbabb, (2, 3, 4, 5, 6, 7), (2, 3, 0, 4, 5, 6, 1, 7)) * -1.0
    t4_abbbabbb += einsum(t1.bb, (0, 1), t3.abbabb, (2, 3, 4, 5, 6, 7), (2, 3, 0, 4, 5, 6, 7, 1))
    t4_abbbabbb += einsum(t1.bb, (0, 1), t3.abbabb, (2, 3, 4, 5, 6, 7), (2, 3, 4, 0, 5, 1, 6, 7)) * -1.0
    t4_abbbabbb += einsum(t1.bb, (0, 1), t3.abbabb, (2, 3, 4, 5, 6, 7), (2, 3, 4, 0, 5, 6, 1, 7))
    t4_abbbabbb += einsum(t1.bb, (0, 1), t3.abbabb, (2, 3, 4, 5, 6, 7), (2, 3, 4, 0, 5, 6, 7, 1)) * -1.0
    t4_abbbabbb += einsum(t2.abab, (0, 1, 2, 3), t2.bbbb, (4, 5, 6, 7), (0, 4, 1, 5, 2, 6, 3, 7)) * -1.0
    t4_abbbabbb += einsum(t2.abab, (0, 1, 2, 3), t2.bbbb, (4, 5, 6, 7), (0, 4, 1, 5, 2, 6, 7, 3))
    t4_abbbabbb += einsum(t2.abab, (0, 1, 2, 3), t2.bbbb, (4, 5, 6, 7), (0, 4, 1, 5, 2, 3, 6, 7))
    t4_abbbabbb += einsum(t2.abab, (0, 1, 2, 3), t2.bbbb, (4, 5, 6, 7), (0, 4, 5, 1, 2, 6, 3, 7))
    t4_abbbabbb += einsum(t2.abab, (0, 1, 2, 3), t2.bbbb, (4, 5, 6, 7), (0, 4, 5, 1, 2, 6, 7, 3)) * -1.0
    t4_abbbabbb += einsum(t2.abab, (0, 1, 2, 3), t2.bbbb, (4, 5, 6, 7), (0, 4, 5, 1, 2, 3, 6, 7)) * -1.0
    t4_abbbabbb += einsum(t2.abab, (0, 1, 2, 3), t2.bbbb, (4, 5, 6, 7), (0, 1, 4, 5, 2, 6, 3, 7))
    t4_abbbabbb += einsum(t2.abab, (0, 1, 2, 3), t2.bbbb, (4, 5, 6, 7), (0, 1, 4, 5, 2, 6, 7, 3)) * -1.0
    t4_abbbabbb += einsum(t2.abab, (0, 1, 2, 3), t2.bbbb, (4, 5, 6, 7), (0, 1, 4, 5, 2, 3, 6, 7)) * -1.0
    t4_abbbabbb += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (0, 2, 4, 5, 1, 3, 6, 7)) * -1.0
    t4_abbbabbb += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (0, 2, 4, 5, 1, 6, 3, 7))
    t4_abbbabbb += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (0, 2, 4, 5, 1, 6, 7, 3)) * -1.0
    t4_abbbabbb += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (0, 4, 2, 5, 1, 3, 6, 7))
    t4_abbbabbb += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (0, 4, 2, 5, 1, 6, 3, 7)) * -1.0
    t4_abbbabbb += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (0, 4, 2, 5, 1, 6, 7, 3))
    t4_abbbabbb += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (0, 4, 5, 2, 1, 3, 6, 7)) * -1.0
    t4_abbbabbb += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (0, 4, 5, 2, 1, 6, 3, 7))
    t4_abbbabbb += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (0, 4, 5, 2, 1, 6, 7, 3)) * -1.0
    t4_abbbabbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (4, 0, 2, 5, 6, 1, 3, 7)) * -1.0
    t4_abbbabbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (4, 0, 2, 5, 6, 1, 7, 3))
    t4_abbbabbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (4, 0, 2, 5, 6, 3, 1, 7))
    t4_abbbabbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (4, 0, 2, 5, 6, 7, 1, 3)) * -1.0
    t4_abbbabbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (4, 0, 2, 5, 6, 3, 7, 1)) * -1.0
    t4_abbbabbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (4, 0, 2, 5, 6, 7, 3, 1))
    t4_abbbabbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (4, 0, 5, 2, 6, 1, 3, 7))
    t4_abbbabbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (4, 0, 5, 2, 6, 1, 7, 3)) * -1.0
    t4_abbbabbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (4, 0, 5, 2, 6, 3, 1, 7)) * -1.0
    t4_abbbabbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (4, 0, 5, 2, 6, 7, 1, 3))
    t4_abbbabbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (4, 0, 5, 2, 6, 3, 7, 1))
    t4_abbbabbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (4, 0, 5, 2, 6, 7, 3, 1)) * -1.0
    t4_abbbabbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (4, 5, 0, 2, 6, 1, 3, 7)) * -1.0
    t4_abbbabbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (4, 5, 0, 2, 6, 1, 7, 3))
    t4_abbbabbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (4, 5, 0, 2, 6, 3, 1, 7))
    t4_abbbabbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (4, 5, 0, 2, 6, 7, 1, 3)) * -1.0
    t4_abbbabbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (4, 5, 0, 2, 6, 3, 7, 1)) * -1.0
    t4_abbbabbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (4, 5, 0, 2, 6, 7, 3, 1))
    t4_abbbabbb += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t1.bb, (4, 5), t1.bb, (6, 7), (0, 2, 4, 6, 1, 3, 5, 7)) * -1.0
    t4_abbbabbb += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t1.bb, (4, 5), t1.bb, (6, 7), (0, 2, 4, 6, 1, 3, 7, 5))
    t4_abbbabbb += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t1.bb, (4, 5), t1.bb, (6, 7), (0, 2, 4, 6, 1, 5, 3, 7))
    t4_abbbabbb += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t1.bb, (4, 5), t1.bb, (6, 7), (0, 2, 4, 6, 1, 7, 3, 5)) * -1.0
    t4_abbbabbb += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t1.bb, (4, 5), t1.bb, (6, 7), (0, 2, 4, 6, 1, 5, 7, 3)) * -1.0
    t4_abbbabbb += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t1.bb, (4, 5), t1.bb, (6, 7), (0, 2, 4, 6, 1, 7, 5, 3))
    return t4_abbbabbb

def t4_uhf_baaabaaa(c4=None, t1=None, t2=None, t3=None, nocc=None, nvir=None):
    t4_baaabaaa = np.zeros((nocc[1], nocc[0], nocc[0], nocc[0], nvir[1], nvir[0], nvir[0], nvir[0]), dtype=np.float64)
    t4_baaabaaa += einsum(c4.baaabaaa, (0, 1, 2, 3, 4, 5, 6, 7), (0, 1, 2, 3, 4, 5, 6, 7))
    t4_baaabaaa += einsum(t1.aa, (0, 1), t3.abaaba, (2, 3, 4, 5, 6, 7), (3, 0, 2, 4, 6, 1, 5, 7)) * -1.0
    t4_baaabaaa += einsum(t1.aa, (0, 1), t3.abaaba, (2, 3, 4, 5, 6, 7), (3, 0, 2, 4, 6, 5, 1, 7))
    t4_baaabaaa += einsum(t1.aa, (0, 1), t3.abaaba, (2, 3, 4, 5, 6, 7), (3, 0, 2, 4, 6, 5, 7, 1)) * -1.0
    t4_baaabaaa += einsum(t1.aa, (0, 1), t3.abaaba, (2, 3, 4, 5, 6, 7), (3, 2, 0, 4, 6, 1, 5, 7))
    t4_baaabaaa += einsum(t1.aa, (0, 1), t3.abaaba, (2, 3, 4, 5, 6, 7), (3, 2, 0, 4, 6, 5, 1, 7)) * -1.0
    t4_baaabaaa += einsum(t1.aa, (0, 1), t3.abaaba, (2, 3, 4, 5, 6, 7), (3, 2, 0, 4, 6, 5, 7, 1))
    t4_baaabaaa += einsum(t1.aa, (0, 1), t3.abaaba, (2, 3, 4, 5, 6, 7), (3, 2, 4, 0, 6, 1, 5, 7)) * -1.0
    t4_baaabaaa += einsum(t1.aa, (0, 1), t3.abaaba, (2, 3, 4, 5, 6, 7), (3, 2, 4, 0, 6, 5, 1, 7))
    t4_baaabaaa += einsum(t1.aa, (0, 1), t3.abaaba, (2, 3, 4, 5, 6, 7), (3, 2, 4, 0, 6, 5, 7, 1)) * -1.0
    t4_baaabaaa += einsum(t1.bb, (0, 1), t3.aaaaaa, (2, 3, 4, 5, 6, 7), (0, 2, 3, 4, 1, 5, 6, 7)) * -1.0
    t4_baaabaaa += einsum(t2.abab, (0, 1, 2, 3), t2.aaaa, (4, 5, 6, 7), (1, 0, 4, 5, 3, 2, 6, 7)) * -1.0
    t4_baaabaaa += einsum(t2.abab, (0, 1, 2, 3), t2.aaaa, (4, 5, 6, 7), (1, 0, 4, 5, 3, 6, 2, 7))
    t4_baaabaaa += einsum(t2.abab, (0, 1, 2, 3), t2.aaaa, (4, 5, 6, 7), (1, 0, 4, 5, 3, 6, 7, 2)) * -1.0
    t4_baaabaaa += einsum(t2.abab, (0, 1, 2, 3), t2.aaaa, (4, 5, 6, 7), (1, 4, 0, 5, 3, 2, 6, 7))
    t4_baaabaaa += einsum(t2.abab, (0, 1, 2, 3), t2.aaaa, (4, 5, 6, 7), (1, 4, 0, 5, 3, 6, 2, 7)) * -1.0
    t4_baaabaaa += einsum(t2.abab, (0, 1, 2, 3), t2.aaaa, (4, 5, 6, 7), (1, 4, 0, 5, 3, 6, 7, 2))
    t4_baaabaaa += einsum(t2.abab, (0, 1, 2, 3), t2.aaaa, (4, 5, 6, 7), (1, 4, 5, 0, 3, 2, 6, 7)) * -1.0
    t4_baaabaaa += einsum(t2.abab, (0, 1, 2, 3), t2.aaaa, (4, 5, 6, 7), (1, 4, 5, 0, 3, 6, 2, 7))
    t4_baaabaaa += einsum(t2.abab, (0, 1, 2, 3), t2.aaaa, (4, 5, 6, 7), (1, 4, 5, 0, 3, 6, 7, 2)) * -1.0
    t4_baaabaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.abab, (4, 5, 6, 7), (5, 0, 2, 4, 7, 1, 3, 6)) * -1.0
    t4_baaabaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.abab, (4, 5, 6, 7), (5, 0, 2, 4, 7, 1, 6, 3))
    t4_baaabaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.abab, (4, 5, 6, 7), (5, 0, 2, 4, 7, 3, 1, 6))
    t4_baaabaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.abab, (4, 5, 6, 7), (5, 0, 2, 4, 7, 6, 1, 3)) * -1.0
    t4_baaabaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.abab, (4, 5, 6, 7), (5, 0, 2, 4, 7, 3, 6, 1)) * -1.0
    t4_baaabaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.abab, (4, 5, 6, 7), (5, 0, 2, 4, 7, 6, 3, 1))
    t4_baaabaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.abab, (4, 5, 6, 7), (5, 0, 4, 2, 7, 1, 3, 6))
    t4_baaabaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.abab, (4, 5, 6, 7), (5, 0, 4, 2, 7, 1, 6, 3)) * -1.0
    t4_baaabaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.abab, (4, 5, 6, 7), (5, 0, 4, 2, 7, 3, 1, 6)) * -1.0
    t4_baaabaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.abab, (4, 5, 6, 7), (5, 0, 4, 2, 7, 6, 1, 3))
    t4_baaabaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.abab, (4, 5, 6, 7), (5, 0, 4, 2, 7, 3, 6, 1))
    t4_baaabaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.abab, (4, 5, 6, 7), (5, 0, 4, 2, 7, 6, 3, 1)) * -1.0
    t4_baaabaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.abab, (4, 5, 6, 7), (5, 4, 0, 2, 7, 1, 3, 6)) * -1.0
    t4_baaabaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.abab, (4, 5, 6, 7), (5, 4, 0, 2, 7, 1, 6, 3))
    t4_baaabaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.abab, (4, 5, 6, 7), (5, 4, 0, 2, 7, 3, 1, 6))
    t4_baaabaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.abab, (4, 5, 6, 7), (5, 4, 0, 2, 7, 6, 1, 3)) * -1.0
    t4_baaabaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.abab, (4, 5, 6, 7), (5, 4, 0, 2, 7, 3, 6, 1)) * -1.0
    t4_baaabaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.abab, (4, 5, 6, 7), (5, 4, 0, 2, 7, 6, 3, 1))
    t4_baaabaaa += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.aaaa, (4, 5, 6, 7), (2, 0, 4, 5, 3, 1, 6, 7)) * -1.0
    t4_baaabaaa += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.aaaa, (4, 5, 6, 7), (2, 0, 4, 5, 3, 6, 1, 7))
    t4_baaabaaa += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.aaaa, (4, 5, 6, 7), (2, 0, 4, 5, 3, 6, 7, 1)) * -1.0
    t4_baaabaaa += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.aaaa, (4, 5, 6, 7), (2, 4, 0, 5, 3, 1, 6, 7))
    t4_baaabaaa += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.aaaa, (4, 5, 6, 7), (2, 4, 0, 5, 3, 6, 1, 7)) * -1.0
    t4_baaabaaa += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.aaaa, (4, 5, 6, 7), (2, 4, 0, 5, 3, 6, 7, 1))
    t4_baaabaaa += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.aaaa, (4, 5, 6, 7), (2, 4, 5, 0, 3, 1, 6, 7)) * -1.0
    t4_baaabaaa += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.aaaa, (4, 5, 6, 7), (2, 4, 5, 0, 3, 6, 1, 7))
    t4_baaabaaa += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.aaaa, (4, 5, 6, 7), (2, 4, 5, 0, 3, 6, 7, 1)) * -1.0
    t4_baaabaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t1.aa, (4, 5), t1.bb, (6, 7), (6, 0, 2, 4, 7, 1, 3, 5)) * -1.0
    t4_baaabaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t1.aa, (4, 5), t1.bb, (6, 7), (6, 0, 2, 4, 7, 1, 5, 3))
    t4_baaabaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t1.aa, (4, 5), t1.bb, (6, 7), (6, 0, 2, 4, 7, 3, 1, 5))
    t4_baaabaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t1.aa, (4, 5), t1.bb, (6, 7), (6, 0, 2, 4, 7, 5, 1, 3)) * -1.0
    t4_baaabaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t1.aa, (4, 5), t1.bb, (6, 7), (6, 0, 2, 4, 7, 3, 5, 1)) * -1.0
    t4_baaabaaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t1.aa, (4, 5), t1.bb, (6, 7), (6, 0, 2, 4, 7, 5, 3, 1))
    return t4_baaabaaa

def t4_uhf_baabbaab(c4=None, t1=None, t2=None, t3=None, nocc=None, nvir=None):
    t4_baabbaab = np.zeros((nocc[1], nocc[0], nocc[0], nocc[1], nvir[1], nvir[0], nvir[0], nvir[1]), dtype=np.float64)
    t4_baabbaab += einsum(c4.abababab, (0, 1, 2, 3, 4, 5, 6, 7), (1, 0, 2, 3, 5, 4, 6, 7))
    t4_baabbaab += einsum(t1.aa, (0, 1), t3.babbab, (2, 3, 4, 5, 6, 7), (2, 0, 3, 4, 5, 1, 6, 7)) * -1.0
    t4_baabbaab += einsum(t1.aa, (0, 1), t3.babbab, (2, 3, 4, 5, 6, 7), (2, 0, 3, 4, 5, 6, 1, 7))
    t4_baabbaab += einsum(t1.aa, (0, 1), t3.babbab, (2, 3, 4, 5, 6, 7), (2, 3, 0, 4, 5, 1, 6, 7))
    t4_baabbaab += einsum(t1.aa, (0, 1), t3.babbab, (2, 3, 4, 5, 6, 7), (2, 3, 0, 4, 5, 6, 1, 7)) * -1.0
    t4_baabbaab += einsum(t1.bb, (0, 1), t3.abaaba, (2, 3, 4, 5, 6, 7), (0, 2, 4, 3, 1, 5, 7, 6)) * -1.0
    t4_baabbaab += einsum(t1.bb, (0, 1), t3.abaaba, (2, 3, 4, 5, 6, 7), (0, 2, 4, 3, 6, 5, 7, 1))
    t4_baabbaab += einsum(t1.bb, (0, 1), t3.abaaba, (2, 3, 4, 5, 6, 7), (3, 2, 4, 0, 1, 5, 7, 6))
    t4_baabbaab += einsum(t1.bb, (0, 1), t3.abaaba, (2, 3, 4, 5, 6, 7), (3, 2, 4, 0, 6, 5, 7, 1)) * -1.0
    t4_baabbaab += einsum(t2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 6, 7), (1, 0, 4, 5, 3, 2, 6, 7)) * -1.0
    t4_baabbaab += einsum(t2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 6, 7), (1, 0, 4, 5, 7, 2, 6, 3))
    t4_baabbaab += einsum(t2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 6, 7), (1, 0, 4, 5, 3, 6, 2, 7))
    t4_baabbaab += einsum(t2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 6, 7), (1, 0, 4, 5, 7, 6, 2, 3)) * -1.0
    t4_baabbaab += einsum(t2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 6, 7), (5, 0, 4, 1, 3, 2, 6, 7))
    t4_baabbaab += einsum(t2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 6, 7), (5, 0, 4, 1, 7, 2, 6, 3)) * -1.0
    t4_baabbaab += einsum(t2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 6, 7), (5, 0, 4, 1, 3, 6, 2, 7)) * -1.0
    t4_baabbaab += einsum(t2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 6, 7), (5, 0, 4, 1, 7, 6, 2, 3))
    t4_baabbaab += einsum(t2.bbbb, (0, 1, 2, 3), t2.aaaa, (4, 5, 6, 7), (0, 4, 5, 1, 2, 6, 7, 3)) * -1.0
    t4_baabbaab += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.bbbb, (4, 5, 6, 7), (4, 0, 2, 5, 6, 1, 3, 7)) * -1.0
    t4_baabbaab += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.bbbb, (4, 5, 6, 7), (4, 0, 2, 5, 6, 3, 1, 7))
    t4_baabbaab += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (2, 0, 4, 5, 3, 1, 6, 7)) * -1.0
    t4_baabbaab += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (2, 0, 4, 5, 7, 1, 6, 3))
    t4_baabbaab += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (2, 0, 4, 5, 3, 6, 1, 7))
    t4_baabbaab += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (2, 0, 4, 5, 7, 6, 1, 3)) * -1.0
    t4_baabbaab += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (5, 0, 4, 2, 3, 1, 6, 7))
    t4_baabbaab += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (5, 0, 4, 2, 7, 1, 6, 3)) * -1.0
    t4_baabbaab += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (5, 0, 4, 2, 3, 6, 1, 7)) * -1.0
    t4_baabbaab += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (5, 0, 4, 2, 7, 6, 1, 3))
    t4_baabbaab += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (2, 4, 0, 5, 3, 1, 6, 7))
    t4_baabbaab += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (2, 4, 0, 5, 7, 1, 6, 3)) * -1.0
    t4_baabbaab += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (2, 4, 0, 5, 3, 6, 1, 7)) * -1.0
    t4_baabbaab += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (2, 4, 0, 5, 7, 6, 1, 3))
    t4_baabbaab += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (5, 4, 0, 2, 3, 1, 6, 7)) * -1.0
    t4_baabbaab += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (5, 4, 0, 2, 7, 1, 6, 3))
    t4_baabbaab += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (5, 4, 0, 2, 3, 6, 1, 7))
    t4_baabbaab += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (5, 4, 0, 2, 7, 6, 1, 3)) * -1.0
    t4_baabbaab += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.aaaa, (4, 5, 6, 7), (0, 4, 5, 2, 1, 6, 7, 3)) * -1.0
    t4_baabbaab += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.aaaa, (4, 5, 6, 7), (0, 4, 5, 2, 3, 6, 7, 1))
    t4_baabbaab += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t1.bb, (4, 5), t1.bb, (6, 7), (4, 0, 2, 6, 5, 1, 3, 7)) * -1.0
    t4_baabbaab += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t1.bb, (4, 5), t1.bb, (6, 7), (4, 0, 2, 6, 7, 1, 3, 5))
    t4_baabbaab += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t1.bb, (4, 5), t1.bb, (6, 7), (4, 0, 2, 6, 5, 3, 1, 7))
    t4_baabbaab += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t1.bb, (4, 5), t1.bb, (6, 7), (4, 0, 2, 6, 7, 3, 1, 5)) * -1.0
    return t4_baabbaab

def t4_uhf_babababa(c4=None, t1=None, t2=None, t3=None, nocc=None, nvir=None):
    t4_babababa = np.zeros((nocc[1], nocc[0], nocc[1], nocc[0], nvir[1], nvir[0], nvir[1], nvir[0]), dtype=np.float64)
    t4_babababa += einsum(c4.abababab, (0, 1, 2, 3, 4, 5, 6, 7), (1, 0, 3, 2, 5, 4, 7, 6))
    t4_babababa += einsum(t1.aa, (0, 1), t3.bbabba, (2, 3, 4, 5, 6, 7), (2, 0, 3, 4, 5, 1, 6, 7)) * -1.0
    t4_babababa += einsum(t1.aa, (0, 1), t3.bbabba, (2, 3, 4, 5, 6, 7), (2, 0, 3, 4, 5, 7, 6, 1))
    t4_babababa += einsum(t1.aa, (0, 1), t3.babbab, (2, 3, 4, 5, 6, 7), (2, 3, 4, 0, 5, 1, 7, 6))
    t4_babababa += einsum(t1.aa, (0, 1), t3.babbab, (2, 3, 4, 5, 6, 7), (2, 3, 4, 0, 5, 6, 7, 1)) * -1.0
    t4_babababa += einsum(t1.bb, (0, 1), t3.abaaba, (2, 3, 4, 5, 6, 7), (0, 2, 3, 4, 1, 5, 6, 7)) * -1.0
    t4_babababa += einsum(t1.bb, (0, 1), t3.abaaba, (2, 3, 4, 5, 6, 7), (0, 2, 3, 4, 6, 5, 1, 7))
    t4_babababa += einsum(t1.bb, (0, 1), t3.abaaba, (2, 3, 4, 5, 6, 7), (3, 2, 0, 4, 1, 5, 6, 7))
    t4_babababa += einsum(t1.bb, (0, 1), t3.abaaba, (2, 3, 4, 5, 6, 7), (3, 2, 0, 4, 6, 5, 1, 7)) * -1.0
    t4_babababa += einsum(t2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 6, 7), (1, 0, 5, 4, 3, 2, 7, 6)) * -1.0
    t4_babababa += einsum(t2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 6, 7), (1, 0, 5, 4, 7, 2, 3, 6))
    t4_babababa += einsum(t2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 6, 7), (1, 0, 5, 4, 3, 6, 7, 2))
    t4_babababa += einsum(t2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 6, 7), (1, 0, 5, 4, 7, 6, 3, 2)) * -1.0
    t4_babababa += einsum(t2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 6, 7), (5, 0, 1, 4, 3, 2, 7, 6))
    t4_babababa += einsum(t2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 6, 7), (5, 0, 1, 4, 7, 2, 3, 6)) * -1.0
    t4_babababa += einsum(t2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 6, 7), (5, 0, 1, 4, 3, 6, 7, 2)) * -1.0
    t4_babababa += einsum(t2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 6, 7), (5, 0, 1, 4, 7, 6, 3, 2))
    t4_babababa += einsum(t2.bbbb, (0, 1, 2, 3), t2.aaaa, (4, 5, 6, 7), (0, 4, 1, 5, 2, 6, 3, 7)) * -1.0
    t4_babababa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.bbbb, (4, 5, 6, 7), (4, 0, 5, 2, 6, 1, 7, 3)) * -1.0
    t4_babababa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.bbbb, (4, 5, 6, 7), (4, 0, 5, 2, 6, 3, 7, 1))
    t4_babababa += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (2, 0, 5, 4, 3, 1, 7, 6)) * -1.0
    t4_babababa += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (2, 0, 5, 4, 7, 1, 3, 6))
    t4_babababa += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (2, 0, 5, 4, 3, 6, 7, 1))
    t4_babababa += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (2, 0, 5, 4, 7, 6, 3, 1)) * -1.0
    t4_babababa += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (5, 0, 2, 4, 3, 1, 7, 6))
    t4_babababa += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (5, 0, 2, 4, 7, 1, 3, 6)) * -1.0
    t4_babababa += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (5, 0, 2, 4, 3, 6, 7, 1)) * -1.0
    t4_babababa += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (5, 0, 2, 4, 7, 6, 3, 1))
    t4_babababa += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (2, 4, 5, 0, 3, 1, 7, 6))
    t4_babababa += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (2, 4, 5, 0, 7, 1, 3, 6)) * -1.0
    t4_babababa += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (2, 4, 5, 0, 3, 6, 7, 1)) * -1.0
    t4_babababa += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (2, 4, 5, 0, 7, 6, 3, 1))
    t4_babababa += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (5, 4, 2, 0, 3, 1, 7, 6)) * -1.0
    t4_babababa += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (5, 4, 2, 0, 7, 1, 3, 6))
    t4_babababa += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (5, 4, 2, 0, 3, 6, 7, 1))
    t4_babababa += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (5, 4, 2, 0, 7, 6, 3, 1)) * -1.0
    t4_babababa += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.aaaa, (4, 5, 6, 7), (0, 4, 2, 5, 1, 6, 3, 7)) * -1.0
    t4_babababa += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.aaaa, (4, 5, 6, 7), (0, 4, 2, 5, 3, 6, 1, 7))
    t4_babababa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t1.bb, (4, 5), t1.bb, (6, 7), (4, 0, 6, 2, 5, 1, 7, 3)) * -1.0
    t4_babababa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t1.bb, (4, 5), t1.bb, (6, 7), (4, 0, 6, 2, 7, 1, 5, 3))
    t4_babababa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t1.bb, (4, 5), t1.bb, (6, 7), (4, 0, 6, 2, 5, 3, 7, 1))
    t4_babababa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t1.bb, (4, 5), t1.bb, (6, 7), (4, 0, 6, 2, 7, 3, 5, 1)) * -1.0
    return t4_babababa

def t4_uhf_babbbabb(c4=None, t1=None, t2=None, t3=None, nocc=None, nvir=None):
    t4_babbbabb = np.zeros((nocc[1], nocc[0], nocc[1], nocc[1], nvir[1], nvir[0], nvir[1], nvir[1]), dtype=np.float64)
    t4_babbbabb += einsum(c4.babbbabb, (0, 1, 2, 3, 4, 5, 6, 7), (0, 1, 2, 3, 4, 5, 6, 7))
    t4_babbbabb += einsum(t1.aa, (0, 1), t3.bbbbbb, (2, 3, 4, 5, 6, 7), (2, 0, 3, 4, 5, 1, 6, 7)) * -1.0
    t4_babbbabb += einsum(t1.bb, (0, 1), t3.abbabb, (2, 3, 4, 5, 6, 7), (0, 2, 3, 4, 1, 5, 6, 7)) * -1.0
    t4_babbbabb += einsum(t1.bb, (0, 1), t3.abbabb, (2, 3, 4, 5, 6, 7), (0, 2, 3, 4, 6, 5, 1, 7))
    t4_babbbabb += einsum(t1.bb, (0, 1), t3.abbabb, (2, 3, 4, 5, 6, 7), (0, 2, 3, 4, 6, 5, 7, 1)) * -1.0
    t4_babbbabb += einsum(t1.bb, (0, 1), t3.babbab, (2, 3, 4, 5, 6, 7), (2, 3, 0, 4, 1, 6, 5, 7))
    t4_babbbabb += einsum(t1.bb, (0, 1), t3.babbab, (2, 3, 4, 5, 6, 7), (2, 3, 0, 4, 5, 6, 1, 7)) * -1.0
    t4_babbbabb += einsum(t1.bb, (0, 1), t3.babbab, (2, 3, 4, 5, 6, 7), (2, 3, 0, 4, 5, 6, 7, 1))
    t4_babbbabb += einsum(t1.bb, (0, 1), t3.babbab, (2, 3, 4, 5, 6, 7), (2, 3, 4, 0, 1, 6, 5, 7)) * -1.0
    t4_babbbabb += einsum(t1.bb, (0, 1), t3.babbab, (2, 3, 4, 5, 6, 7), (2, 3, 4, 0, 5, 6, 1, 7))
    t4_babbbabb += einsum(t1.bb, (0, 1), t3.babbab, (2, 3, 4, 5, 6, 7), (2, 3, 4, 0, 5, 6, 7, 1)) * -1.0
    t4_babbbabb += einsum(t2.abab, (0, 1, 2, 3), t2.bbbb, (4, 5, 6, 7), (4, 0, 1, 5, 6, 2, 3, 7)) * -1.0
    t4_babbbabb += einsum(t2.abab, (0, 1, 2, 3), t2.bbbb, (4, 5, 6, 7), (4, 0, 1, 5, 6, 2, 7, 3))
    t4_babbbabb += einsum(t2.abab, (0, 1, 2, 3), t2.bbbb, (4, 5, 6, 7), (4, 0, 1, 5, 3, 2, 6, 7))
    t4_babbbabb += einsum(t2.abab, (0, 1, 2, 3), t2.bbbb, (4, 5, 6, 7), (4, 0, 5, 1, 6, 2, 3, 7))
    t4_babbbabb += einsum(t2.abab, (0, 1, 2, 3), t2.bbbb, (4, 5, 6, 7), (4, 0, 5, 1, 6, 2, 7, 3)) * -1.0
    t4_babbbabb += einsum(t2.abab, (0, 1, 2, 3), t2.bbbb, (4, 5, 6, 7), (4, 0, 5, 1, 3, 2, 6, 7)) * -1.0
    t4_babbbabb += einsum(t2.abab, (0, 1, 2, 3), t2.bbbb, (4, 5, 6, 7), (1, 0, 4, 5, 6, 2, 3, 7))
    t4_babbbabb += einsum(t2.abab, (0, 1, 2, 3), t2.bbbb, (4, 5, 6, 7), (1, 0, 4, 5, 6, 2, 7, 3)) * -1.0
    t4_babbbabb += einsum(t2.abab, (0, 1, 2, 3), t2.bbbb, (4, 5, 6, 7), (1, 0, 4, 5, 3, 2, 6, 7)) * -1.0
    t4_babbbabb += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (2, 0, 4, 5, 3, 1, 6, 7)) * -1.0
    t4_babbbabb += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (2, 0, 4, 5, 6, 1, 3, 7))
    t4_babbbabb += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (2, 0, 4, 5, 6, 1, 7, 3)) * -1.0
    t4_babbbabb += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (4, 0, 2, 5, 3, 1, 6, 7))
    t4_babbbabb += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (4, 0, 2, 5, 6, 1, 3, 7)) * -1.0
    t4_babbbabb += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (4, 0, 2, 5, 6, 1, 7, 3))
    t4_babbbabb += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (4, 0, 5, 2, 3, 1, 6, 7)) * -1.0
    t4_babbbabb += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (4, 0, 5, 2, 6, 1, 3, 7))
    t4_babbbabb += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (4, 0, 5, 2, 6, 1, 7, 3)) * -1.0
    t4_babbbabb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (0, 4, 2, 5, 1, 6, 3, 7)) * -1.0
    t4_babbbabb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (0, 4, 2, 5, 1, 6, 7, 3))
    t4_babbbabb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (0, 4, 2, 5, 3, 6, 1, 7))
    t4_babbbabb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (0, 4, 2, 5, 7, 6, 1, 3)) * -1.0
    t4_babbbabb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (0, 4, 2, 5, 3, 6, 7, 1)) * -1.0
    t4_babbbabb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (0, 4, 2, 5, 7, 6, 3, 1))
    t4_babbbabb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (0, 4, 5, 2, 1, 6, 3, 7))
    t4_babbbabb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (0, 4, 5, 2, 1, 6, 7, 3)) * -1.0
    t4_babbbabb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (0, 4, 5, 2, 3, 6, 1, 7)) * -1.0
    t4_babbbabb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (0, 4, 5, 2, 7, 6, 1, 3))
    t4_babbbabb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (0, 4, 5, 2, 3, 6, 7, 1))
    t4_babbbabb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (0, 4, 5, 2, 7, 6, 3, 1)) * -1.0
    t4_babbbabb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (5, 4, 0, 2, 1, 6, 3, 7)) * -1.0
    t4_babbbabb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (5, 4, 0, 2, 1, 6, 7, 3))
    t4_babbbabb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (5, 4, 0, 2, 3, 6, 1, 7))
    t4_babbbabb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (5, 4, 0, 2, 7, 6, 1, 3)) * -1.0
    t4_babbbabb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (5, 4, 0, 2, 3, 6, 7, 1)) * -1.0
    t4_babbbabb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (5, 4, 0, 2, 7, 6, 3, 1))
    t4_babbbabb += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t1.bb, (4, 5), t1.bb, (6, 7), (2, 0, 4, 6, 3, 1, 5, 7)) * -1.0
    t4_babbbabb += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t1.bb, (4, 5), t1.bb, (6, 7), (2, 0, 4, 6, 3, 1, 7, 5))
    t4_babbbabb += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t1.bb, (4, 5), t1.bb, (6, 7), (2, 0, 4, 6, 5, 1, 3, 7))
    t4_babbbabb += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t1.bb, (4, 5), t1.bb, (6, 7), (2, 0, 4, 6, 7, 1, 3, 5)) * -1.0
    t4_babbbabb += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t1.bb, (4, 5), t1.bb, (6, 7), (2, 0, 4, 6, 5, 1, 7, 3)) * -1.0
    t4_babbbabb += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t1.bb, (4, 5), t1.bb, (6, 7), (2, 0, 4, 6, 7, 1, 5, 3))
    return t4_babbbabb

def t4_uhf_bbaabbaa(c4=None, t1=None, t2=None, t3=None, nocc=None, nvir=None):
    t4_bbaabbaa = np.zeros((nocc[1], nocc[1], nocc[0], nocc[0], nvir[1], nvir[1], nvir[0], nvir[0]), dtype=np.float64)
    t4_bbaabbaa += einsum(c4.abababab, (0, 1, 2, 3, 4, 5, 6, 7), (1, 3, 0, 2, 5, 7, 4, 6))
    t4_bbaabbaa += einsum(t1.aa, (0, 1), t3.bbabba, (2, 3, 4, 5, 6, 7), (2, 3, 0, 4, 5, 6, 1, 7)) * -1.0
    t4_bbaabbaa += einsum(t1.aa, (0, 1), t3.bbabba, (2, 3, 4, 5, 6, 7), (2, 3, 0, 4, 5, 6, 7, 1))
    t4_bbaabbaa += einsum(t1.aa, (0, 1), t3.bbabba, (2, 3, 4, 5, 6, 7), (2, 3, 4, 0, 5, 6, 1, 7))
    t4_bbaabbaa += einsum(t1.aa, (0, 1), t3.bbabba, (2, 3, 4, 5, 6, 7), (2, 3, 4, 0, 5, 6, 7, 1)) * -1.0
    t4_bbaabbaa += einsum(t1.bb, (0, 1), t3.abaaba, (2, 3, 4, 5, 6, 7), (0, 3, 2, 4, 1, 6, 5, 7)) * -1.0
    t4_bbaabbaa += einsum(t1.bb, (0, 1), t3.abaaba, (2, 3, 4, 5, 6, 7), (0, 3, 2, 4, 6, 1, 5, 7))
    t4_bbaabbaa += einsum(t1.bb, (0, 1), t3.abaaba, (2, 3, 4, 5, 6, 7), (3, 0, 2, 4, 1, 6, 5, 7))
    t4_bbaabbaa += einsum(t1.bb, (0, 1), t3.abaaba, (2, 3, 4, 5, 6, 7), (3, 0, 2, 4, 6, 1, 5, 7)) * -1.0
    t4_bbaabbaa += einsum(t2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 6, 7), (1, 5, 0, 4, 3, 7, 2, 6)) * -1.0
    t4_bbaabbaa += einsum(t2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 6, 7), (1, 5, 0, 4, 7, 3, 2, 6))
    t4_bbaabbaa += einsum(t2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 6, 7), (1, 5, 0, 4, 3, 7, 6, 2))
    t4_bbaabbaa += einsum(t2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 6, 7), (1, 5, 0, 4, 7, 3, 6, 2)) * -1.0
    t4_bbaabbaa += einsum(t2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 6, 7), (5, 1, 0, 4, 3, 7, 2, 6))
    t4_bbaabbaa += einsum(t2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 6, 7), (5, 1, 0, 4, 7, 3, 2, 6)) * -1.0
    t4_bbaabbaa += einsum(t2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 6, 7), (5, 1, 0, 4, 3, 7, 6, 2)) * -1.0
    t4_bbaabbaa += einsum(t2.abab, (0, 1, 2, 3), t2.abab, (4, 5, 6, 7), (5, 1, 0, 4, 7, 3, 6, 2))
    t4_bbaabbaa += einsum(t2.bbbb, (0, 1, 2, 3), t2.aaaa, (4, 5, 6, 7), (0, 1, 4, 5, 2, 3, 6, 7)) * -1.0
    t4_bbaabbaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.bbbb, (4, 5, 6, 7), (4, 5, 0, 2, 6, 7, 1, 3)) * -1.0
    t4_bbaabbaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t2.bbbb, (4, 5, 6, 7), (4, 5, 0, 2, 6, 7, 3, 1))
    t4_bbaabbaa += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (2, 5, 0, 4, 3, 7, 1, 6)) * -1.0
    t4_bbaabbaa += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (2, 5, 0, 4, 7, 3, 1, 6))
    t4_bbaabbaa += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (2, 5, 0, 4, 3, 7, 6, 1))
    t4_bbaabbaa += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (2, 5, 0, 4, 7, 3, 6, 1)) * -1.0
    t4_bbaabbaa += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (5, 2, 0, 4, 3, 7, 1, 6))
    t4_bbaabbaa += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (5, 2, 0, 4, 7, 3, 1, 6)) * -1.0
    t4_bbaabbaa += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (5, 2, 0, 4, 3, 7, 6, 1)) * -1.0
    t4_bbaabbaa += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (5, 2, 0, 4, 7, 3, 6, 1))
    t4_bbaabbaa += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (2, 5, 4, 0, 3, 7, 1, 6))
    t4_bbaabbaa += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (2, 5, 4, 0, 7, 3, 1, 6)) * -1.0
    t4_bbaabbaa += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (2, 5, 4, 0, 3, 7, 6, 1)) * -1.0
    t4_bbaabbaa += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (2, 5, 4, 0, 7, 3, 6, 1))
    t4_bbaabbaa += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (5, 2, 4, 0, 3, 7, 1, 6)) * -1.0
    t4_bbaabbaa += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (5, 2, 4, 0, 7, 3, 1, 6))
    t4_bbaabbaa += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (5, 2, 4, 0, 3, 7, 6, 1))
    t4_bbaabbaa += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (5, 2, 4, 0, 7, 3, 6, 1)) * -1.0
    t4_bbaabbaa += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.aaaa, (4, 5, 6, 7), (0, 2, 4, 5, 1, 3, 6, 7)) * -1.0
    t4_bbaabbaa += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.aaaa, (4, 5, 6, 7), (0, 2, 4, 5, 3, 1, 6, 7))
    t4_bbaabbaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t1.bb, (4, 5), t1.bb, (6, 7), (4, 6, 0, 2, 5, 7, 1, 3)) * -1.0
    t4_bbaabbaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t1.bb, (4, 5), t1.bb, (6, 7), (4, 6, 0, 2, 7, 5, 1, 3))
    t4_bbaabbaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t1.bb, (4, 5), t1.bb, (6, 7), (4, 6, 0, 2, 5, 7, 3, 1))
    t4_bbaabbaa += einsum(t1.aa, (0, 1), t1.aa, (2, 3), t1.bb, (4, 5), t1.bb, (6, 7), (4, 6, 0, 2, 7, 5, 3, 1)) * -1.0
    return t4_bbaabbaa

def t4_uhf_bbabbbab(c4=None, t1=None, t2=None, t3=None, nocc=None, nvir=None):
    t4_bbabbbab = np.zeros((nocc[1], nocc[1], nocc[0], nocc[1], nvir[1], nvir[1], nvir[0], nvir[1]), dtype=np.float64)
    t4_bbabbbab += einsum(c4.bbabbbab, (0, 1, 2, 3, 4, 5, 6, 7), (0, 1, 2, 3, 4, 5, 6, 7))
    t4_bbabbbab += einsum(t1.aa, (0, 1), t3.bbbbbb, (2, 3, 4, 5, 6, 7), (2, 3, 0, 4, 5, 6, 1, 7)) * -1.0
    t4_bbabbbab += einsum(t1.bb, (0, 1), t3.babbab, (2, 3, 4, 5, 6, 7), (0, 2, 3, 4, 1, 5, 6, 7)) * -1.0
    t4_bbabbbab += einsum(t1.bb, (0, 1), t3.babbab, (2, 3, 4, 5, 6, 7), (0, 2, 3, 4, 5, 1, 6, 7))
    t4_bbabbbab += einsum(t1.bb, (0, 1), t3.babbab, (2, 3, 4, 5, 6, 7), (0, 2, 3, 4, 5, 7, 6, 1)) * -1.0
    t4_bbabbbab += einsum(t1.bb, (0, 1), t3.babbab, (2, 3, 4, 5, 6, 7), (2, 0, 3, 4, 1, 5, 6, 7))
    t4_bbabbbab += einsum(t1.bb, (0, 1), t3.babbab, (2, 3, 4, 5, 6, 7), (2, 0, 3, 4, 5, 1, 6, 7)) * -1.0
    t4_bbabbbab += einsum(t1.bb, (0, 1), t3.babbab, (2, 3, 4, 5, 6, 7), (2, 0, 3, 4, 5, 7, 6, 1))
    t4_bbabbbab += einsum(t1.bb, (0, 1), t3.bbabba, (2, 3, 4, 5, 6, 7), (2, 3, 4, 0, 1, 5, 7, 6)) * -1.0
    t4_bbabbbab += einsum(t1.bb, (0, 1), t3.bbabba, (2, 3, 4, 5, 6, 7), (2, 3, 4, 0, 5, 1, 7, 6))
    t4_bbabbbab += einsum(t1.bb, (0, 1), t3.bbabba, (2, 3, 4, 5, 6, 7), (2, 3, 4, 0, 5, 6, 7, 1)) * -1.0
    t4_bbabbbab += einsum(t2.abab, (0, 1, 2, 3), t2.bbbb, (4, 5, 6, 7), (4, 1, 0, 5, 6, 3, 2, 7)) * -1.0
    t4_bbabbbab += einsum(t2.abab, (0, 1, 2, 3), t2.bbbb, (4, 5, 6, 7), (4, 1, 0, 5, 6, 7, 2, 3))
    t4_bbabbbab += einsum(t2.abab, (0, 1, 2, 3), t2.bbbb, (4, 5, 6, 7), (4, 1, 0, 5, 3, 6, 2, 7))
    t4_bbabbbab += einsum(t2.abab, (0, 1, 2, 3), t2.bbbb, (4, 5, 6, 7), (4, 5, 0, 1, 6, 3, 2, 7))
    t4_bbabbbab += einsum(t2.abab, (0, 1, 2, 3), t2.bbbb, (4, 5, 6, 7), (4, 5, 0, 1, 6, 7, 2, 3)) * -1.0
    t4_bbabbbab += einsum(t2.abab, (0, 1, 2, 3), t2.bbbb, (4, 5, 6, 7), (4, 5, 0, 1, 3, 6, 2, 7)) * -1.0
    t4_bbabbbab += einsum(t2.abab, (0, 1, 2, 3), t2.bbbb, (4, 5, 6, 7), (1, 4, 0, 5, 6, 3, 2, 7))
    t4_bbabbbab += einsum(t2.abab, (0, 1, 2, 3), t2.bbbb, (4, 5, 6, 7), (1, 4, 0, 5, 6, 7, 2, 3)) * -1.0
    t4_bbabbbab += einsum(t2.abab, (0, 1, 2, 3), t2.bbbb, (4, 5, 6, 7), (1, 4, 0, 5, 3, 6, 2, 7)) * -1.0
    t4_bbabbbab += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (2, 4, 0, 5, 3, 6, 1, 7)) * -1.0
    t4_bbabbbab += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (2, 4, 0, 5, 6, 3, 1, 7))
    t4_bbabbbab += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (2, 4, 0, 5, 6, 7, 1, 3)) * -1.0
    t4_bbabbbab += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (4, 2, 0, 5, 3, 6, 1, 7))
    t4_bbabbbab += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (4, 2, 0, 5, 6, 3, 1, 7)) * -1.0
    t4_bbabbbab += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (4, 2, 0, 5, 6, 7, 1, 3))
    t4_bbabbbab += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (4, 5, 0, 2, 3, 6, 1, 7)) * -1.0
    t4_bbabbbab += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (4, 5, 0, 2, 6, 3, 1, 7))
    t4_bbabbbab += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (4, 5, 0, 2, 6, 7, 1, 3)) * -1.0
    t4_bbabbbab += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (0, 2, 4, 5, 1, 3, 6, 7)) * -1.0
    t4_bbabbbab += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (0, 2, 4, 5, 1, 7, 6, 3))
    t4_bbabbbab += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (0, 2, 4, 5, 3, 1, 6, 7))
    t4_bbabbbab += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (0, 2, 4, 5, 7, 1, 6, 3)) * -1.0
    t4_bbabbbab += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (0, 2, 4, 5, 3, 7, 6, 1)) * -1.0
    t4_bbabbbab += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (0, 2, 4, 5, 7, 3, 6, 1))
    t4_bbabbbab += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (0, 5, 4, 2, 1, 3, 6, 7))
    t4_bbabbbab += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (0, 5, 4, 2, 1, 7, 6, 3)) * -1.0
    t4_bbabbbab += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (0, 5, 4, 2, 3, 1, 6, 7)) * -1.0
    t4_bbabbbab += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (0, 5, 4, 2, 7, 1, 6, 3))
    t4_bbabbbab += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (0, 5, 4, 2, 3, 7, 6, 1))
    t4_bbabbbab += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (0, 5, 4, 2, 7, 3, 6, 1)) * -1.0
    t4_bbabbbab += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (5, 0, 4, 2, 1, 3, 6, 7)) * -1.0
    t4_bbabbbab += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (5, 0, 4, 2, 1, 7, 6, 3))
    t4_bbabbbab += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (5, 0, 4, 2, 3, 1, 6, 7))
    t4_bbabbbab += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (5, 0, 4, 2, 7, 1, 6, 3)) * -1.0
    t4_bbabbbab += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (5, 0, 4, 2, 3, 7, 6, 1)) * -1.0
    t4_bbabbbab += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (5, 0, 4, 2, 7, 3, 6, 1))
    t4_bbabbbab += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t1.bb, (4, 5), t1.bb, (6, 7), (2, 4, 0, 6, 3, 5, 1, 7)) * -1.0
    t4_bbabbbab += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t1.bb, (4, 5), t1.bb, (6, 7), (2, 4, 0, 6, 3, 7, 1, 5))
    t4_bbabbbab += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t1.bb, (4, 5), t1.bb, (6, 7), (2, 4, 0, 6, 5, 3, 1, 7))
    t4_bbabbbab += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t1.bb, (4, 5), t1.bb, (6, 7), (2, 4, 0, 6, 7, 3, 1, 5)) * -1.0
    t4_bbabbbab += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t1.bb, (4, 5), t1.bb, (6, 7), (2, 4, 0, 6, 5, 7, 1, 3)) * -1.0
    t4_bbabbbab += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t1.bb, (4, 5), t1.bb, (6, 7), (2, 4, 0, 6, 7, 5, 1, 3))
    return t4_bbabbbab

def t4_uhf_bbbabbba(c4=None, t1=None, t2=None, t3=None, nocc=None, nvir=None):
    t4_bbbabbba = np.zeros((nocc[1], nocc[1], nocc[1], nocc[0], nvir[1], nvir[1], nvir[1], nvir[0]), dtype=np.float64)
    t4_bbbabbba += einsum(c4.bbbabbba, (0, 1, 2, 3, 4, 5, 6, 7), (0, 1, 2, 3, 4, 5, 6, 7))
    t4_bbbabbba += einsum(t1.aa, (0, 1), t3.bbbbbb, (2, 3, 4, 5, 6, 7), (2, 3, 4, 0, 5, 6, 7, 1)) * -1.0
    t4_bbbabbba += einsum(t1.bb, (0, 1), t3.bbabba, (2, 3, 4, 5, 6, 7), (0, 2, 3, 4, 1, 5, 6, 7)) * -1.0
    t4_bbbabbba += einsum(t1.bb, (0, 1), t3.bbabba, (2, 3, 4, 5, 6, 7), (0, 2, 3, 4, 5, 1, 6, 7))
    t4_bbbabbba += einsum(t1.bb, (0, 1), t3.bbabba, (2, 3, 4, 5, 6, 7), (0, 2, 3, 4, 5, 6, 1, 7)) * -1.0
    t4_bbbabbba += einsum(t1.bb, (0, 1), t3.bbabba, (2, 3, 4, 5, 6, 7), (2, 0, 3, 4, 1, 5, 6, 7))
    t4_bbbabbba += einsum(t1.bb, (0, 1), t3.bbabba, (2, 3, 4, 5, 6, 7), (2, 0, 3, 4, 5, 1, 6, 7)) * -1.0
    t4_bbbabbba += einsum(t1.bb, (0, 1), t3.bbabba, (2, 3, 4, 5, 6, 7), (2, 0, 3, 4, 5, 6, 1, 7))
    t4_bbbabbba += einsum(t1.bb, (0, 1), t3.bbabba, (2, 3, 4, 5, 6, 7), (2, 3, 0, 4, 1, 5, 6, 7)) * -1.0
    t4_bbbabbba += einsum(t1.bb, (0, 1), t3.bbabba, (2, 3, 4, 5, 6, 7), (2, 3, 0, 4, 5, 1, 6, 7))
    t4_bbbabbba += einsum(t1.bb, (0, 1), t3.bbabba, (2, 3, 4, 5, 6, 7), (2, 3, 0, 4, 5, 6, 1, 7)) * -1.0
    t4_bbbabbba += einsum(t2.abab, (0, 1, 2, 3), t2.bbbb, (4, 5, 6, 7), (4, 1, 5, 0, 6, 3, 7, 2)) * -1.0
    t4_bbbabbba += einsum(t2.abab, (0, 1, 2, 3), t2.bbbb, (4, 5, 6, 7), (4, 1, 5, 0, 6, 7, 3, 2))
    t4_bbbabbba += einsum(t2.abab, (0, 1, 2, 3), t2.bbbb, (4, 5, 6, 7), (4, 1, 5, 0, 3, 6, 7, 2))
    t4_bbbabbba += einsum(t2.abab, (0, 1, 2, 3), t2.bbbb, (4, 5, 6, 7), (4, 5, 1, 0, 6, 3, 7, 2))
    t4_bbbabbba += einsum(t2.abab, (0, 1, 2, 3), t2.bbbb, (4, 5, 6, 7), (4, 5, 1, 0, 6, 7, 3, 2)) * -1.0
    t4_bbbabbba += einsum(t2.abab, (0, 1, 2, 3), t2.bbbb, (4, 5, 6, 7), (4, 5, 1, 0, 3, 6, 7, 2)) * -1.0
    t4_bbbabbba += einsum(t2.abab, (0, 1, 2, 3), t2.bbbb, (4, 5, 6, 7), (1, 4, 5, 0, 6, 3, 7, 2))
    t4_bbbabbba += einsum(t2.abab, (0, 1, 2, 3), t2.bbbb, (4, 5, 6, 7), (1, 4, 5, 0, 6, 7, 3, 2)) * -1.0
    t4_bbbabbba += einsum(t2.abab, (0, 1, 2, 3), t2.bbbb, (4, 5, 6, 7), (1, 4, 5, 0, 3, 6, 7, 2)) * -1.0
    t4_bbbabbba += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (2, 4, 5, 0, 3, 6, 7, 1)) * -1.0
    t4_bbbabbba += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (2, 4, 5, 0, 6, 3, 7, 1))
    t4_bbbabbba += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (2, 4, 5, 0, 6, 7, 3, 1)) * -1.0
    t4_bbbabbba += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (4, 2, 5, 0, 3, 6, 7, 1))
    t4_bbbabbba += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (4, 2, 5, 0, 6, 3, 7, 1)) * -1.0
    t4_bbbabbba += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (4, 2, 5, 0, 6, 7, 3, 1))
    t4_bbbabbba += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (4, 5, 2, 0, 3, 6, 7, 1)) * -1.0
    t4_bbbabbba += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (4, 5, 2, 0, 6, 3, 7, 1))
    t4_bbbabbba += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (4, 5, 2, 0, 6, 7, 3, 1)) * -1.0
    t4_bbbabbba += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (0, 2, 5, 4, 1, 3, 7, 6)) * -1.0
    t4_bbbabbba += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (0, 2, 5, 4, 1, 7, 3, 6))
    t4_bbbabbba += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (0, 2, 5, 4, 3, 1, 7, 6))
    t4_bbbabbba += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (0, 2, 5, 4, 7, 1, 3, 6)) * -1.0
    t4_bbbabbba += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (0, 2, 5, 4, 3, 7, 1, 6)) * -1.0
    t4_bbbabbba += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (0, 2, 5, 4, 7, 3, 1, 6))
    t4_bbbabbba += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (0, 5, 2, 4, 1, 3, 7, 6))
    t4_bbbabbba += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (0, 5, 2, 4, 1, 7, 3, 6)) * -1.0
    t4_bbbabbba += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (0, 5, 2, 4, 3, 1, 7, 6)) * -1.0
    t4_bbbabbba += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (0, 5, 2, 4, 7, 1, 3, 6))
    t4_bbbabbba += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (0, 5, 2, 4, 3, 7, 1, 6))
    t4_bbbabbba += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (0, 5, 2, 4, 7, 3, 1, 6)) * -1.0
    t4_bbbabbba += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (5, 0, 2, 4, 1, 3, 7, 6)) * -1.0
    t4_bbbabbba += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (5, 0, 2, 4, 1, 7, 3, 6))
    t4_bbbabbba += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (5, 0, 2, 4, 3, 1, 7, 6))
    t4_bbbabbba += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (5, 0, 2, 4, 7, 1, 3, 6)) * -1.0
    t4_bbbabbba += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (5, 0, 2, 4, 3, 7, 1, 6)) * -1.0
    t4_bbbabbba += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.abab, (4, 5, 6, 7), (5, 0, 2, 4, 7, 3, 1, 6))
    t4_bbbabbba += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t1.bb, (4, 5), t1.bb, (6, 7), (2, 4, 6, 0, 3, 5, 7, 1)) * -1.0
    t4_bbbabbba += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t1.bb, (4, 5), t1.bb, (6, 7), (2, 4, 6, 0, 3, 7, 5, 1))
    t4_bbbabbba += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t1.bb, (4, 5), t1.bb, (6, 7), (2, 4, 6, 0, 5, 3, 7, 1))
    t4_bbbabbba += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t1.bb, (4, 5), t1.bb, (6, 7), (2, 4, 6, 0, 7, 3, 5, 1)) * -1.0
    t4_bbbabbba += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t1.bb, (4, 5), t1.bb, (6, 7), (2, 4, 6, 0, 5, 7, 3, 1)) * -1.0
    t4_bbbabbba += einsum(t1.aa, (0, 1), t1.bb, (2, 3), t1.bb, (4, 5), t1.bb, (6, 7), (2, 4, 6, 0, 7, 5, 3, 1))
    return t4_bbbabbba

def t4_uhf_bbbbbbbb(c4=None, t1=None, t2=None, t3=None, nocc=None, nvir=None):
    t4_bbbbbbbb = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1], nvir[1]), dtype=np.float64)
    t4_bbbbbbbb += einsum(c4.bbbbbbbb, (0, 1, 2, 3, 4, 5, 6, 7), (0, 1, 2, 3, 4, 5, 6, 7))
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t3.bbbbbb, (2, 3, 4, 5, 6, 7), (0, 2, 3, 4, 1, 5, 6, 7)) * -1.0
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t3.bbbbbb, (2, 3, 4, 5, 6, 7), (0, 2, 3, 4, 5, 1, 6, 7))
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t3.bbbbbb, (2, 3, 4, 5, 6, 7), (0, 2, 3, 4, 5, 6, 1, 7)) * -1.0
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t3.bbbbbb, (2, 3, 4, 5, 6, 7), (0, 2, 3, 4, 5, 6, 7, 1))
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t3.bbbbbb, (2, 3, 4, 5, 6, 7), (2, 0, 3, 4, 1, 5, 6, 7))
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t3.bbbbbb, (2, 3, 4, 5, 6, 7), (2, 0, 3, 4, 5, 1, 6, 7)) * -1.0
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t3.bbbbbb, (2, 3, 4, 5, 6, 7), (2, 0, 3, 4, 5, 6, 1, 7))
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t3.bbbbbb, (2, 3, 4, 5, 6, 7), (2, 0, 3, 4, 5, 6, 7, 1)) * -1.0
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t3.bbbbbb, (2, 3, 4, 5, 6, 7), (2, 3, 0, 4, 1, 5, 6, 7)) * -1.0
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t3.bbbbbb, (2, 3, 4, 5, 6, 7), (2, 3, 0, 4, 5, 1, 6, 7))
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t3.bbbbbb, (2, 3, 4, 5, 6, 7), (2, 3, 0, 4, 5, 6, 1, 7)) * -1.0
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t3.bbbbbb, (2, 3, 4, 5, 6, 7), (2, 3, 0, 4, 5, 6, 7, 1))
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t3.bbbbbb, (2, 3, 4, 5, 6, 7), (2, 3, 4, 0, 1, 5, 6, 7))
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t3.bbbbbb, (2, 3, 4, 5, 6, 7), (2, 3, 4, 0, 5, 1, 6, 7)) * -1.0
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t3.bbbbbb, (2, 3, 4, 5, 6, 7), (2, 3, 4, 0, 5, 6, 1, 7))
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t3.bbbbbb, (2, 3, 4, 5, 6, 7), (2, 3, 4, 0, 5, 6, 7, 1)) * -1.0
    t4_bbbbbbbb += einsum(t2.bbbb, (0, 1, 2, 3), t2.bbbb, (4, 5, 6, 7), (0, 4, 1, 5, 2, 6, 3, 7)) * -1.0
    t4_bbbbbbbb += einsum(t2.bbbb, (0, 1, 2, 3), t2.bbbb, (4, 5, 6, 7), (0, 4, 1, 5, 2, 6, 7, 3))
    t4_bbbbbbbb += einsum(t2.bbbb, (0, 1, 2, 3), t2.bbbb, (4, 5, 6, 7), (0, 4, 1, 5, 2, 3, 6, 7))
    t4_bbbbbbbb += einsum(t2.bbbb, (0, 1, 2, 3), t2.bbbb, (4, 5, 6, 7), (0, 4, 1, 5, 6, 2, 3, 7))
    t4_bbbbbbbb += einsum(t2.bbbb, (0, 1, 2, 3), t2.bbbb, (4, 5, 6, 7), (0, 4, 1, 5, 6, 2, 7, 3)) * -1.0
    t4_bbbbbbbb += einsum(t2.bbbb, (0, 1, 2, 3), t2.bbbb, (4, 5, 6, 7), (0, 4, 1, 5, 6, 7, 2, 3))
    t4_bbbbbbbb += einsum(t2.bbbb, (0, 1, 2, 3), t2.bbbb, (4, 5, 6, 7), (0, 4, 5, 1, 2, 6, 3, 7))
    t4_bbbbbbbb += einsum(t2.bbbb, (0, 1, 2, 3), t2.bbbb, (4, 5, 6, 7), (0, 4, 5, 1, 2, 6, 7, 3)) * -1.0
    t4_bbbbbbbb += einsum(t2.bbbb, (0, 1, 2, 3), t2.bbbb, (4, 5, 6, 7), (0, 4, 5, 1, 2, 3, 6, 7)) * -1.0
    t4_bbbbbbbb += einsum(t2.bbbb, (0, 1, 2, 3), t2.bbbb, (4, 5, 6, 7), (0, 4, 5, 1, 6, 2, 3, 7)) * -1.0
    t4_bbbbbbbb += einsum(t2.bbbb, (0, 1, 2, 3), t2.bbbb, (4, 5, 6, 7), (0, 4, 5, 1, 6, 2, 7, 3))
    t4_bbbbbbbb += einsum(t2.bbbb, (0, 1, 2, 3), t2.bbbb, (4, 5, 6, 7), (0, 4, 5, 1, 6, 7, 2, 3)) * -1.0
    t4_bbbbbbbb += einsum(t2.bbbb, (0, 1, 2, 3), t2.bbbb, (4, 5, 6, 7), (0, 1, 4, 5, 2, 6, 3, 7))
    t4_bbbbbbbb += einsum(t2.bbbb, (0, 1, 2, 3), t2.bbbb, (4, 5, 6, 7), (0, 1, 4, 5, 2, 6, 7, 3)) * -1.0
    t4_bbbbbbbb += einsum(t2.bbbb, (0, 1, 2, 3), t2.bbbb, (4, 5, 6, 7), (0, 1, 4, 5, 2, 3, 6, 7)) * -1.0
    t4_bbbbbbbb += einsum(t2.bbbb, (0, 1, 2, 3), t2.bbbb, (4, 5, 6, 7), (0, 1, 4, 5, 6, 2, 3, 7)) * -1.0
    t4_bbbbbbbb += einsum(t2.bbbb, (0, 1, 2, 3), t2.bbbb, (4, 5, 6, 7), (0, 1, 4, 5, 6, 2, 7, 3))
    t4_bbbbbbbb += einsum(t2.bbbb, (0, 1, 2, 3), t2.bbbb, (4, 5, 6, 7), (0, 1, 4, 5, 6, 7, 2, 3)) * -1.0
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (0, 2, 4, 5, 1, 3, 6, 7)) * -1.0
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (0, 2, 4, 5, 1, 6, 3, 7))
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (0, 2, 4, 5, 1, 6, 7, 3)) * -1.0
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (0, 2, 4, 5, 3, 1, 6, 7))
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (0, 2, 4, 5, 6, 1, 3, 7)) * -1.0
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (0, 2, 4, 5, 6, 1, 7, 3))
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (0, 2, 4, 5, 3, 6, 1, 7)) * -1.0
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (0, 2, 4, 5, 6, 3, 1, 7))
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (0, 2, 4, 5, 6, 7, 1, 3)) * -1.0
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (0, 2, 4, 5, 3, 6, 7, 1))
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (0, 2, 4, 5, 6, 3, 7, 1)) * -1.0
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (0, 2, 4, 5, 6, 7, 3, 1))
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (0, 4, 2, 5, 1, 3, 6, 7))
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (0, 4, 2, 5, 1, 6, 3, 7)) * -1.0
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (0, 4, 2, 5, 1, 6, 7, 3))
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (0, 4, 2, 5, 3, 1, 6, 7)) * -1.0
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (0, 4, 2, 5, 6, 1, 3, 7))
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (0, 4, 2, 5, 6, 1, 7, 3)) * -1.0
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (0, 4, 2, 5, 3, 6, 1, 7))
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (0, 4, 2, 5, 6, 3, 1, 7)) * -1.0
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (0, 4, 2, 5, 6, 7, 1, 3))
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (0, 4, 2, 5, 3, 6, 7, 1)) * -1.0
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (0, 4, 2, 5, 6, 3, 7, 1))
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (0, 4, 2, 5, 6, 7, 3, 1)) * -1.0
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (0, 4, 5, 2, 1, 3, 6, 7)) * -1.0
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (0, 4, 5, 2, 1, 6, 3, 7))
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (0, 4, 5, 2, 1, 6, 7, 3)) * -1.0
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (0, 4, 5, 2, 3, 1, 6, 7))
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (0, 4, 5, 2, 6, 1, 3, 7)) * -1.0
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (0, 4, 5, 2, 6, 1, 7, 3))
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (0, 4, 5, 2, 3, 6, 1, 7)) * -1.0
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (0, 4, 5, 2, 6, 3, 1, 7))
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (0, 4, 5, 2, 6, 7, 1, 3)) * -1.0
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (0, 4, 5, 2, 3, 6, 7, 1))
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (0, 4, 5, 2, 6, 3, 7, 1)) * -1.0
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (0, 4, 5, 2, 6, 7, 3, 1))
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (4, 0, 2, 5, 1, 3, 6, 7)) * -1.0
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (4, 0, 2, 5, 1, 6, 3, 7))
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (4, 0, 2, 5, 1, 6, 7, 3)) * -1.0
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (4, 0, 2, 5, 3, 1, 6, 7))
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (4, 0, 2, 5, 6, 1, 3, 7)) * -1.0
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (4, 0, 2, 5, 6, 1, 7, 3))
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (4, 0, 2, 5, 3, 6, 1, 7)) * -1.0
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (4, 0, 2, 5, 6, 3, 1, 7))
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (4, 0, 2, 5, 6, 7, 1, 3)) * -1.0
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (4, 0, 2, 5, 3, 6, 7, 1))
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (4, 0, 2, 5, 6, 3, 7, 1)) * -1.0
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (4, 0, 2, 5, 6, 7, 3, 1))
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (4, 0, 5, 2, 1, 3, 6, 7))
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (4, 0, 5, 2, 1, 6, 3, 7)) * -1.0
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (4, 0, 5, 2, 1, 6, 7, 3))
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (4, 0, 5, 2, 3, 1, 6, 7)) * -1.0
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (4, 0, 5, 2, 6, 1, 3, 7))
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (4, 0, 5, 2, 6, 1, 7, 3)) * -1.0
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (4, 0, 5, 2, 3, 6, 1, 7))
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (4, 0, 5, 2, 6, 3, 1, 7)) * -1.0
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (4, 0, 5, 2, 6, 7, 1, 3))
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (4, 0, 5, 2, 3, 6, 7, 1)) * -1.0
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (4, 0, 5, 2, 6, 3, 7, 1))
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (4, 0, 5, 2, 6, 7, 3, 1)) * -1.0
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (4, 5, 0, 2, 1, 3, 6, 7)) * -1.0
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (4, 5, 0, 2, 1, 6, 3, 7))
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (4, 5, 0, 2, 1, 6, 7, 3)) * -1.0
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (4, 5, 0, 2, 3, 1, 6, 7))
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (4, 5, 0, 2, 6, 1, 3, 7)) * -1.0
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (4, 5, 0, 2, 6, 1, 7, 3))
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (4, 5, 0, 2, 3, 6, 1, 7)) * -1.0
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (4, 5, 0, 2, 6, 3, 1, 7))
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (4, 5, 0, 2, 6, 7, 1, 3)) * -1.0
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (4, 5, 0, 2, 3, 6, 7, 1))
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (4, 5, 0, 2, 6, 3, 7, 1)) * -1.0
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t2.bbbb, (4, 5, 6, 7), (4, 5, 0, 2, 6, 7, 3, 1))
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t1.bb, (4, 5), t1.bb, (6, 7), (0, 2, 4, 6, 1, 3, 5, 7)) * -1.0
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t1.bb, (4, 5), t1.bb, (6, 7), (0, 2, 4, 6, 1, 3, 7, 5))
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t1.bb, (4, 5), t1.bb, (6, 7), (0, 2, 4, 6, 1, 5, 3, 7))
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t1.bb, (4, 5), t1.bb, (6, 7), (0, 2, 4, 6, 1, 7, 3, 5)) * -1.0
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t1.bb, (4, 5), t1.bb, (6, 7), (0, 2, 4, 6, 1, 5, 7, 3)) * -1.0
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t1.bb, (4, 5), t1.bb, (6, 7), (0, 2, 4, 6, 1, 7, 5, 3))
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t1.bb, (4, 5), t1.bb, (6, 7), (0, 2, 4, 6, 3, 1, 5, 7))
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t1.bb, (4, 5), t1.bb, (6, 7), (0, 2, 4, 6, 3, 1, 7, 5)) * -1.0
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t1.bb, (4, 5), t1.bb, (6, 7), (0, 2, 4, 6, 5, 1, 3, 7)) * -1.0
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t1.bb, (4, 5), t1.bb, (6, 7), (0, 2, 4, 6, 7, 1, 3, 5))
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t1.bb, (4, 5), t1.bb, (6, 7), (0, 2, 4, 6, 5, 1, 7, 3))
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t1.bb, (4, 5), t1.bb, (6, 7), (0, 2, 4, 6, 7, 1, 5, 3)) * -1.0
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t1.bb, (4, 5), t1.bb, (6, 7), (0, 2, 4, 6, 3, 5, 1, 7)) * -1.0
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t1.bb, (4, 5), t1.bb, (6, 7), (0, 2, 4, 6, 3, 7, 1, 5))
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t1.bb, (4, 5), t1.bb, (6, 7), (0, 2, 4, 6, 5, 3, 1, 7))
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t1.bb, (4, 5), t1.bb, (6, 7), (0, 2, 4, 6, 7, 3, 1, 5)) * -1.0
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t1.bb, (4, 5), t1.bb, (6, 7), (0, 2, 4, 6, 5, 7, 1, 3)) * -1.0
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t1.bb, (4, 5), t1.bb, (6, 7), (0, 2, 4, 6, 7, 5, 1, 3))
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t1.bb, (4, 5), t1.bb, (6, 7), (0, 2, 4, 6, 3, 5, 7, 1))
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t1.bb, (4, 5), t1.bb, (6, 7), (0, 2, 4, 6, 3, 7, 5, 1)) * -1.0
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t1.bb, (4, 5), t1.bb, (6, 7), (0, 2, 4, 6, 5, 3, 7, 1)) * -1.0
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t1.bb, (4, 5), t1.bb, (6, 7), (0, 2, 4, 6, 7, 3, 5, 1))
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t1.bb, (4, 5), t1.bb, (6, 7), (0, 2, 4, 6, 5, 7, 3, 1))
    t4_bbbbbbbb += einsum(t1.bb, (0, 1), t1.bb, (2, 3), t1.bb, (4, 5), t1.bb, (6, 7), (0, 2, 4, 6, 7, 5, 3, 1)) * -1.0
    return t4_bbbbbbbb

def t4_rhf(c4=None, t1=None, t2=None, t3=None, nocc=None, nvir=None):
    t4 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir, nvir, nvir), dtype=np.float64)
    t4 += einsum(c4, (0, 1, 2, 3, 4, 5, 6, 7), (0, 1, 2, 3, 4, 5, 6, 7))
    t4 += einsum(t1, (0, 1), t3, (2, 3, 4, 5, 6, 7), (0, 2, 3, 4, 1, 5, 6, 7)) * -1.0
    t4 += einsum(t1, (0, 1), t3, (2, 3, 4, 5, 6, 7), (0, 2, 3, 4, 6, 5, 1, 7))
    t4 += einsum(t1, (0, 1), t3, (2, 3, 4, 5, 6, 7), (2, 0, 4, 3, 5, 1, 7, 6)) * -1.0
    t4 += einsum(t1, (0, 1), t3, (2, 3, 4, 5, 6, 7), (2, 0, 4, 3, 5, 6, 7, 1))
    t4 += einsum(t1, (0, 1), t3, (2, 3, 4, 5, 6, 7), (3, 2, 0, 4, 1, 5, 6, 7))
    t4 += einsum(t1, (0, 1), t3, (2, 3, 4, 5, 6, 7), (3, 2, 0, 4, 6, 5, 1, 7)) * -1.0
    t4 += einsum(t1, (0, 1), t3, (2, 3, 4, 5, 6, 7), (2, 3, 4, 0, 5, 1, 7, 6))
    t4 += einsum(t1, (0, 1), t3, (2, 3, 4, 5, 6, 7), (2, 3, 4, 0, 5, 6, 7, 1)) * -1.0
    t4 += einsum(t2, (0, 1, 2, 3), t2, (4, 5, 6, 7), (0, 4, 1, 5, 2, 6, 3, 7)) * -1.0
    t4 += einsum(t2, (0, 1, 2, 3), t2, (4, 5, 6, 7), (0, 4, 1, 5, 2, 7, 3, 6))
    t4 += einsum(t2, (0, 1, 2, 3), t2, (4, 5, 6, 7), (0, 4, 1, 5, 3, 6, 2, 7))
    t4 += einsum(t2, (0, 1, 2, 3), t2, (4, 5, 6, 7), (0, 4, 1, 5, 3, 7, 2, 6)) * -1.0
    t4 += einsum(t2, (0, 1, 2, 3), t2, (4, 5, 6, 7), (0, 1, 4, 5, 2, 3, 6, 7)) * -1.0
    t4 += einsum(t2, (0, 1, 2, 3), t2, (4, 5, 6, 7), (0, 1, 4, 5, 2, 7, 6, 3))
    t4 += einsum(t2, (0, 1, 2, 3), t2, (4, 5, 6, 7), (0, 1, 4, 5, 6, 3, 2, 7))
    t4 += einsum(t2, (0, 1, 2, 3), t2, (4, 5, 6, 7), (0, 1, 4, 5, 6, 7, 2, 3)) * -1.0
    t4 += einsum(t2, (0, 1, 2, 3), t2, (4, 5, 6, 7), (0, 5, 4, 1, 2, 3, 6, 7))
    t4 += einsum(t2, (0, 1, 2, 3), t2, (4, 5, 6, 7), (0, 5, 4, 1, 2, 7, 6, 3)) * -1.0
    t4 += einsum(t2, (0, 1, 2, 3), t2, (4, 5, 6, 7), (0, 5, 4, 1, 6, 3, 2, 7)) * -1.0
    t4 += einsum(t2, (0, 1, 2, 3), t2, (4, 5, 6, 7), (0, 5, 4, 1, 6, 7, 2, 3))
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (0, 2, 4, 5, 1, 3, 6, 7)) * -1.0
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (0, 2, 4, 5, 1, 7, 6, 3))
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (0, 2, 4, 5, 6, 3, 1, 7))
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (0, 2, 4, 5, 6, 7, 1, 3)) * -1.0
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (0, 4, 2, 5, 1, 6, 3, 7)) * -1.0
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (0, 4, 2, 5, 1, 7, 3, 6))
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (0, 4, 2, 5, 3, 6, 1, 7))
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (0, 4, 2, 5, 3, 7, 1, 6)) * -1.0
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (0, 5, 4, 2, 1, 3, 6, 7))
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (0, 5, 4, 2, 1, 7, 6, 3)) * -1.0
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (0, 5, 4, 2, 6, 3, 1, 7)) * -1.0
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (0, 5, 4, 2, 6, 7, 1, 3))
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (4, 0, 2, 5, 3, 1, 6, 7))
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (4, 0, 2, 5, 6, 1, 3, 7)) * -1.0
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (4, 0, 2, 5, 3, 7, 6, 1)) * -1.0
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (4, 0, 2, 5, 6, 7, 3, 1))
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (4, 0, 5, 2, 6, 1, 7, 3)) * -1.0
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (4, 0, 5, 2, 7, 1, 6, 3))
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (4, 0, 5, 2, 6, 3, 7, 1))
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (4, 0, 5, 2, 7, 3, 6, 1)) * -1.0
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (4, 5, 0, 2, 1, 3, 6, 7)) * -1.0
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (4, 5, 0, 2, 1, 7, 6, 3))
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (4, 5, 0, 2, 6, 3, 1, 7))
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (4, 5, 0, 2, 6, 7, 1, 3)) * -1.0
    t4 += einsum(t1, (0, 1), t1, (2, 3), t1, (4, 5), t1, (6, 7), (0, 2, 4, 6, 1, 3, 5, 7)) * -1.0
    t4 += einsum(t1, (0, 1), t1, (2, 3), t1, (4, 5), t1, (6, 7), (0, 2, 4, 6, 1, 7, 5, 3))
    t4 += einsum(t1, (0, 1), t1, (2, 3), t1, (4, 5), t1, (6, 7), (0, 2, 4, 6, 5, 3, 1, 7))
    t4 += einsum(t1, (0, 1), t1, (2, 3), t1, (4, 5), t1, (6, 7), (0, 2, 4, 6, 5, 7, 1, 3)) * -1.0
    return t4

def t4_ghf(c4=None, t1=None, t2=None, t3=None, nocc=None, nvir=None):
    t4 = np.zeros((nocc, nocc, nocc, nocc, nvir, nvir, nvir, nvir), dtype=np.float64)
    t4 += einsum(c4, (0, 1, 2, 3, 4, 5, 6, 7), (0, 1, 2, 3, 4, 5, 6, 7))
    t4 += einsum(t1, (0, 1), t3, (2, 3, 4, 5, 6, 7), (0, 2, 3, 4, 1, 5, 6, 7)) * -1.0
    t4 += einsum(t1, (0, 1), t3, (2, 3, 4, 5, 6, 7), (0, 2, 3, 4, 5, 1, 6, 7))
    t4 += einsum(t1, (0, 1), t3, (2, 3, 4, 5, 6, 7), (0, 2, 3, 4, 5, 6, 1, 7)) * -1.0
    t4 += einsum(t1, (0, 1), t3, (2, 3, 4, 5, 6, 7), (0, 2, 3, 4, 5, 6, 7, 1))
    t4 += einsum(t1, (0, 1), t3, (2, 3, 4, 5, 6, 7), (2, 0, 3, 4, 1, 5, 6, 7))
    t4 += einsum(t1, (0, 1), t3, (2, 3, 4, 5, 6, 7), (2, 0, 3, 4, 5, 1, 6, 7)) * -1.0
    t4 += einsum(t1, (0, 1), t3, (2, 3, 4, 5, 6, 7), (2, 0, 3, 4, 5, 6, 1, 7))
    t4 += einsum(t1, (0, 1), t3, (2, 3, 4, 5, 6, 7), (2, 0, 3, 4, 5, 6, 7, 1)) * -1.0
    t4 += einsum(t1, (0, 1), t3, (2, 3, 4, 5, 6, 7), (2, 3, 0, 4, 1, 5, 6, 7)) * -1.0
    t4 += einsum(t1, (0, 1), t3, (2, 3, 4, 5, 6, 7), (2, 3, 0, 4, 5, 1, 6, 7))
    t4 += einsum(t1, (0, 1), t3, (2, 3, 4, 5, 6, 7), (2, 3, 0, 4, 5, 6, 1, 7)) * -1.0
    t4 += einsum(t1, (0, 1), t3, (2, 3, 4, 5, 6, 7), (2, 3, 0, 4, 5, 6, 7, 1))
    t4 += einsum(t1, (0, 1), t3, (2, 3, 4, 5, 6, 7), (2, 3, 4, 0, 1, 5, 6, 7))
    t4 += einsum(t1, (0, 1), t3, (2, 3, 4, 5, 6, 7), (2, 3, 4, 0, 5, 1, 6, 7)) * -1.0
    t4 += einsum(t1, (0, 1), t3, (2, 3, 4, 5, 6, 7), (2, 3, 4, 0, 5, 6, 1, 7))
    t4 += einsum(t1, (0, 1), t3, (2, 3, 4, 5, 6, 7), (2, 3, 4, 0, 5, 6, 7, 1)) * -1.0
    t4 += einsum(t2, (0, 1, 2, 3), t2, (4, 5, 6, 7), (0, 4, 1, 5, 2, 6, 3, 7)) * -1.0
    t4 += einsum(t2, (0, 1, 2, 3), t2, (4, 5, 6, 7), (0, 4, 1, 5, 2, 6, 7, 3))
    t4 += einsum(t2, (0, 1, 2, 3), t2, (4, 5, 6, 7), (0, 4, 1, 5, 2, 3, 6, 7))
    t4 += einsum(t2, (0, 1, 2, 3), t2, (4, 5, 6, 7), (0, 4, 1, 5, 6, 2, 3, 7))
    t4 += einsum(t2, (0, 1, 2, 3), t2, (4, 5, 6, 7), (0, 4, 1, 5, 6, 2, 7, 3)) * -1.0
    t4 += einsum(t2, (0, 1, 2, 3), t2, (4, 5, 6, 7), (0, 4, 1, 5, 6, 7, 2, 3))
    t4 += einsum(t2, (0, 1, 2, 3), t2, (4, 5, 6, 7), (0, 4, 5, 1, 2, 6, 3, 7))
    t4 += einsum(t2, (0, 1, 2, 3), t2, (4, 5, 6, 7), (0, 4, 5, 1, 2, 6, 7, 3)) * -1.0
    t4 += einsum(t2, (0, 1, 2, 3), t2, (4, 5, 6, 7), (0, 4, 5, 1, 2, 3, 6, 7)) * -1.0
    t4 += einsum(t2, (0, 1, 2, 3), t2, (4, 5, 6, 7), (0, 4, 5, 1, 6, 2, 3, 7)) * -1.0
    t4 += einsum(t2, (0, 1, 2, 3), t2, (4, 5, 6, 7), (0, 4, 5, 1, 6, 2, 7, 3))
    t4 += einsum(t2, (0, 1, 2, 3), t2, (4, 5, 6, 7), (0, 4, 5, 1, 6, 7, 2, 3)) * -1.0
    t4 += einsum(t2, (0, 1, 2, 3), t2, (4, 5, 6, 7), (0, 1, 4, 5, 2, 6, 3, 7))
    t4 += einsum(t2, (0, 1, 2, 3), t2, (4, 5, 6, 7), (0, 1, 4, 5, 2, 6, 7, 3)) * -1.0
    t4 += einsum(t2, (0, 1, 2, 3), t2, (4, 5, 6, 7), (0, 1, 4, 5, 2, 3, 6, 7)) * -1.0
    t4 += einsum(t2, (0, 1, 2, 3), t2, (4, 5, 6, 7), (0, 1, 4, 5, 6, 2, 3, 7)) * -1.0
    t4 += einsum(t2, (0, 1, 2, 3), t2, (4, 5, 6, 7), (0, 1, 4, 5, 6, 2, 7, 3))
    t4 += einsum(t2, (0, 1, 2, 3), t2, (4, 5, 6, 7), (0, 1, 4, 5, 6, 7, 2, 3)) * -1.0
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (0, 2, 4, 5, 1, 3, 6, 7)) * -1.0
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (0, 2, 4, 5, 1, 6, 3, 7))
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (0, 2, 4, 5, 1, 6, 7, 3)) * -1.0
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (0, 2, 4, 5, 3, 1, 6, 7))
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (0, 2, 4, 5, 6, 1, 3, 7)) * -1.0
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (0, 2, 4, 5, 6, 1, 7, 3))
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (0, 2, 4, 5, 3, 6, 1, 7)) * -1.0
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (0, 2, 4, 5, 6, 3, 1, 7))
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (0, 2, 4, 5, 6, 7, 1, 3)) * -1.0
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (0, 2, 4, 5, 3, 6, 7, 1))
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (0, 2, 4, 5, 6, 3, 7, 1)) * -1.0
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (0, 2, 4, 5, 6, 7, 3, 1))
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (0, 4, 2, 5, 1, 3, 6, 7))
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (0, 4, 2, 5, 1, 6, 3, 7)) * -1.0
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (0, 4, 2, 5, 1, 6, 7, 3))
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (0, 4, 2, 5, 3, 1, 6, 7)) * -1.0
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (0, 4, 2, 5, 6, 1, 3, 7))
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (0, 4, 2, 5, 6, 1, 7, 3)) * -1.0
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (0, 4, 2, 5, 3, 6, 1, 7))
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (0, 4, 2, 5, 6, 3, 1, 7)) * -1.0
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (0, 4, 2, 5, 6, 7, 1, 3))
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (0, 4, 2, 5, 3, 6, 7, 1)) * -1.0
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (0, 4, 2, 5, 6, 3, 7, 1))
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (0, 4, 2, 5, 6, 7, 3, 1)) * -1.0
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (0, 4, 5, 2, 1, 3, 6, 7)) * -1.0
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (0, 4, 5, 2, 1, 6, 3, 7))
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (0, 4, 5, 2, 1, 6, 7, 3)) * -1.0
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (0, 4, 5, 2, 3, 1, 6, 7))
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (0, 4, 5, 2, 6, 1, 3, 7)) * -1.0
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (0, 4, 5, 2, 6, 1, 7, 3))
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (0, 4, 5, 2, 3, 6, 1, 7)) * -1.0
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (0, 4, 5, 2, 6, 3, 1, 7))
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (0, 4, 5, 2, 6, 7, 1, 3)) * -1.0
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (0, 4, 5, 2, 3, 6, 7, 1))
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (0, 4, 5, 2, 6, 3, 7, 1)) * -1.0
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (0, 4, 5, 2, 6, 7, 3, 1))
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (4, 0, 2, 5, 1, 3, 6, 7)) * -1.0
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (4, 0, 2, 5, 1, 6, 3, 7))
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (4, 0, 2, 5, 1, 6, 7, 3)) * -1.0
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (4, 0, 2, 5, 3, 1, 6, 7))
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (4, 0, 2, 5, 6, 1, 3, 7)) * -1.0
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (4, 0, 2, 5, 6, 1, 7, 3))
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (4, 0, 2, 5, 3, 6, 1, 7)) * -1.0
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (4, 0, 2, 5, 6, 3, 1, 7))
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (4, 0, 2, 5, 6, 7, 1, 3)) * -1.0
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (4, 0, 2, 5, 3, 6, 7, 1))
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (4, 0, 2, 5, 6, 3, 7, 1)) * -1.0
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (4, 0, 2, 5, 6, 7, 3, 1))
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (4, 0, 5, 2, 1, 3, 6, 7))
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (4, 0, 5, 2, 1, 6, 3, 7)) * -1.0
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (4, 0, 5, 2, 1, 6, 7, 3))
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (4, 0, 5, 2, 3, 1, 6, 7)) * -1.0
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (4, 0, 5, 2, 6, 1, 3, 7))
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (4, 0, 5, 2, 6, 1, 7, 3)) * -1.0
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (4, 0, 5, 2, 3, 6, 1, 7))
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (4, 0, 5, 2, 6, 3, 1, 7)) * -1.0
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (4, 0, 5, 2, 6, 7, 1, 3))
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (4, 0, 5, 2, 3, 6, 7, 1)) * -1.0
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (4, 0, 5, 2, 6, 3, 7, 1))
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (4, 0, 5, 2, 6, 7, 3, 1)) * -1.0
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (4, 5, 0, 2, 1, 3, 6, 7)) * -1.0
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (4, 5, 0, 2, 1, 6, 3, 7))
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (4, 5, 0, 2, 1, 6, 7, 3)) * -1.0
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (4, 5, 0, 2, 3, 1, 6, 7))
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (4, 5, 0, 2, 6, 1, 3, 7)) * -1.0
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (4, 5, 0, 2, 6, 1, 7, 3))
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (4, 5, 0, 2, 3, 6, 1, 7)) * -1.0
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (4, 5, 0, 2, 6, 3, 1, 7))
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (4, 5, 0, 2, 6, 7, 1, 3)) * -1.0
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (4, 5, 0, 2, 3, 6, 7, 1))
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (4, 5, 0, 2, 6, 3, 7, 1)) * -1.0
    t4 += einsum(t1, (0, 1), t1, (2, 3), t2, (4, 5, 6, 7), (4, 5, 0, 2, 6, 7, 3, 1))
    t4 += einsum(t1, (0, 1), t1, (2, 3), t1, (4, 5), t1, (6, 7), (0, 2, 4, 6, 1, 3, 5, 7)) * -1.0
    t4 += einsum(t1, (0, 1), t1, (2, 3), t1, (4, 5), t1, (6, 7), (0, 2, 4, 6, 1, 3, 7, 5))
    t4 += einsum(t1, (0, 1), t1, (2, 3), t1, (4, 5), t1, (6, 7), (0, 2, 4, 6, 1, 5, 3, 7))
    t4 += einsum(t1, (0, 1), t1, (2, 3), t1, (4, 5), t1, (6, 7), (0, 2, 4, 6, 1, 7, 3, 5)) * -1.0
    t4 += einsum(t1, (0, 1), t1, (2, 3), t1, (4, 5), t1, (6, 7), (0, 2, 4, 6, 1, 5, 7, 3)) * -1.0
    t4 += einsum(t1, (0, 1), t1, (2, 3), t1, (4, 5), t1, (6, 7), (0, 2, 4, 6, 1, 7, 5, 3))
    t4 += einsum(t1, (0, 1), t1, (2, 3), t1, (4, 5), t1, (6, 7), (0, 2, 4, 6, 3, 1, 5, 7))
    t4 += einsum(t1, (0, 1), t1, (2, 3), t1, (4, 5), t1, (6, 7), (0, 2, 4, 6, 3, 1, 7, 5)) * -1.0
    t4 += einsum(t1, (0, 1), t1, (2, 3), t1, (4, 5), t1, (6, 7), (0, 2, 4, 6, 5, 1, 3, 7)) * -1.0
    t4 += einsum(t1, (0, 1), t1, (2, 3), t1, (4, 5), t1, (6, 7), (0, 2, 4, 6, 7, 1, 3, 5))
    t4 += einsum(t1, (0, 1), t1, (2, 3), t1, (4, 5), t1, (6, 7), (0, 2, 4, 6, 5, 1, 7, 3))
    t4 += einsum(t1, (0, 1), t1, (2, 3), t1, (4, 5), t1, (6, 7), (0, 2, 4, 6, 7, 1, 5, 3)) * -1.0
    t4 += einsum(t1, (0, 1), t1, (2, 3), t1, (4, 5), t1, (6, 7), (0, 2, 4, 6, 3, 5, 1, 7)) * -1.0
    t4 += einsum(t1, (0, 1), t1, (2, 3), t1, (4, 5), t1, (6, 7), (0, 2, 4, 6, 3, 7, 1, 5))
    t4 += einsum(t1, (0, 1), t1, (2, 3), t1, (4, 5), t1, (6, 7), (0, 2, 4, 6, 5, 3, 1, 7))
    t4 += einsum(t1, (0, 1), t1, (2, 3), t1, (4, 5), t1, (6, 7), (0, 2, 4, 6, 7, 3, 1, 5)) * -1.0
    t4 += einsum(t1, (0, 1), t1, (2, 3), t1, (4, 5), t1, (6, 7), (0, 2, 4, 6, 5, 7, 1, 3)) * -1.0
    t4 += einsum(t1, (0, 1), t1, (2, 3), t1, (4, 5), t1, (6, 7), (0, 2, 4, 6, 7, 5, 1, 3))
    t4 += einsum(t1, (0, 1), t1, (2, 3), t1, (4, 5), t1, (6, 7), (0, 2, 4, 6, 3, 5, 7, 1))
    t4 += einsum(t1, (0, 1), t1, (2, 3), t1, (4, 5), t1, (6, 7), (0, 2, 4, 6, 3, 7, 5, 1)) * -1.0
    t4 += einsum(t1, (0, 1), t1, (2, 3), t1, (4, 5), t1, (6, 7), (0, 2, 4, 6, 5, 3, 7, 1)) * -1.0
    t4 += einsum(t1, (0, 1), t1, (2, 3), t1, (4, 5), t1, (6, 7), (0, 2, 4, 6, 7, 3, 5, 1))
    t4 += einsum(t1, (0, 1), t1, (2, 3), t1, (4, 5), t1, (6, 7), (0, 2, 4, 6, 5, 7, 3, 1))
    t4 += einsum(t1, (0, 1), t1, (2, 3), t1, (4, 5), t1, (6, 7), (0, 2, 4, 6, 7, 5, 3, 1)) * -1.0
    return t4

