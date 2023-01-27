import numpy as np
import vayesta
from vayesta.core.util import *
from vayesta.core.types import wf as wf_types


def CISDTQ_WaveFunction(mo, *args, **kwargs):
    if mo.nspin == 1:
        cls = RCISDTQ_WaveFunction
    elif mo.nspin == 2:
        cls = UCISDTQ_WaveFunction
    return cls(mo, *args, **kwargs)


class RCISDTQ_WaveFunction(wf_types.WaveFunction):

    def __init__(self, mo, c0, c1, c2, c3, c4):
        super().__init__(mo)
        self.c0 = c0
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.c4 = c4
        if not (isinstance(c4, tuple) and len(c4) == 2):
            raise ValueError("c4 definition in RCISDTQ wfn requires tuple of (abaa, abab) spin signatures")

    def as_ccsdtq(self):
        t1 = self.c1/self.c0
        t2 = self.c2/self.c0 - einsum('ia,jb->ijab', t1, t1)
        # see also THE JOURNAL OF CHEMICAL PHYSICS 147, 154105 (2017)

        # === t3 ===
        t3 = self.c3/self.c0
        # As a useful intermediate, compute t2_aa from t2 (_ab), which is 
        # just the antisymmetric permutation.
        t2aa = t2 - t2.transpose(1,0,2,3)

        t1t2 = einsum('ia, jkbc -> ijkabc', t1, t2)
        t1t1t1 = einsum('ia,jb,kc -> ijkabc', t1, t1, t1)

        t3 -= t1t2                                          
        t3 += t1t2.transpose(0,1,2,5,4,3)
        t3 += t1t2.transpose(2,1,0,3,4,5)                                      
        t3 -= t1t2.transpose(2,1,0,5,4,3)                                      
        t3 -= einsum('jb,ikac -> ijkabc', t1, t2aa) 
        t3 -= t1t1t1
        t3 += t1t1t1.transpose(0,1,2,5,4,3)

        # === t4_abaa === (Note that we construct both the abaa and abab spin signatures)
        # Unpack c4 array into spin signatures
        c4_abaa, c4_abab = self.c4
        # Construct the abaa first
        t4_abaa = c4_abaa/self.c0

        # A useful intermediate is the t3_aaa. Construct this from t3 (_aba) mixed spin.
        t3_aaa = t3 - t3.transpose(0,2,1,3,4,5) - t3.transpose(1,0,2,3,4,5)

        # (t1 t3) terms + permutations
        t1t3a = einsum('ia,kjlcbd -> ijklabcd', t1, t3) 
        t1t3b = np.einsum('kd,ijlabc -> ijklabcd', t1, t3) 

        t4_abaa -= t1t3a 
        t4_abaa += t1t3b 
        t4_abaa += t1t3a.transpose(0,1,2,3,6,5,4,7)   
        t4_abaa -= t1t3a.transpose(0,1,3,2,7,5,6,4)  
        t4_abaa -= einsum('jb, iklacd -> ijklabcd', t1, t3_aaa)
        t4_abaa += t1t3a.transpose(2,1,0,3,4,5,6,7) 
        t4_abaa -= t1t3a.transpose(2,1,0,3,6,5,4,7) 
        t4_abaa -= t1t3a.transpose(3,1,2,0,4,5,7,6) 
        t4_abaa += t1t3b.transpose(0,1,3,2,4,5,7,6) 
        t4_abaa -= t1t3b.transpose(0,1,3,2,4,5,6,7)

        # (t2 t2) terms + permutations
        t2t2a = einsum('ijab, klcd -> ijklabcd', t2, t2aa)
        t2t2b = einsum('ljcb, kida -> ijklabcd', t2, t2aa)

        t4_abaa -= t2t2a
        t4_abaa += t2t2b
        t4_abaa += t2t2a.transpose(0,1,2,3,6,5,4,7)
        t4_abaa -= t2t2a.transpose(0,1,3,2,7,5,6,4)
        t4_abaa -= t2t2a.transpose(3,1,2,0,7,5,6,4)
        t4_abaa -= t2t2a.transpose(3,1,2,0,4,5,7,6)
        t4_abaa += t2t2b.transpose(0,1,3,2,4,5,7,6)
        t4_abaa -= t2t2a.transpose(2,1,0,3,6,5,4,7)
        t4_abaa += t2t2a.transpose(2,1,0,3,4,5,6,7) 

        # (t1 t1 t2) terms + permutations
        t1t1t2a = einsum('ia, jb, klcd -> ijklabcd', t1, t1, t2aa)
        t1t1t2b = einsum('kd, jb, ilac -> ijklabcd', t1, t1, t2aa)
        t1t1t2c = einsum('ia, kc, ljdb -> ijklabcd', t1, t1, t2)
        t1t1t2d = einsum('ic, ld, kjab -> ijklabcd', t1, t1, t2)

        t4_abaa -= t1t1t2a
        t4_abaa += t1t1t2a.transpose(0,1,2,3,6,5,4,7)
        t4_abaa -= t1t1t2a.transpose(0,1,3,2,7,5,6,4)
        t4_abaa += t1t1t2a.transpose(2,1,0,3,4,5,6,7)
        t4_abaa -= t1t1t2a.transpose(2,1,0,3,6,5,4,7)
        t4_abaa += t1t1t2b 
        t4_abaa -= t1t1t2a.transpose(3,1,2,0,4,5,7,6)
        t4_abaa += t1t1t2b.transpose(0,1,3,2,4,5,7,6)
        t4_abaa -= t1t1t2b.transpose(0,1,3,2,4,5,6,7)
        t4_abaa -= t1t1t2c 
        t4_abaa += t1t1t2c.transpose(0,1,2,3,4,5,7,6)
        t4_abaa += t1t1t2c.transpose(0,1,3,2,4,5,6,7)
        t4_abaa -= t1t1t2c.transpose(0,1,3,2,4,5,7,6)
        t4_abaa += t1t1t2c.transpose(0,1,2,3,6,5,4,7)
        t4_abaa -= t1t1t2c.transpose(2,1,0,3,7,5,6,4)
        t4_abaa -= t1t1t2c.transpose(0,1,3,2,6,5,4,7)
        t4_abaa += t1t1t2d 
        t4_abaa -= t1t1t2c.transpose(2,1,0,3,4,5,7,6)
        t4_abaa += t1t1t2c.transpose(0,1,2,3,7,5,6,4)
        t4_abaa += t1t1t2d.transpose(3,1,2,0,6,5,4,7)
        t4_abaa -= t1t1t2d.transpose(0,1,2,3,4,5,7,6)
        t4_abaa -= t1t1t2c.transpose(3,1,2,0,6,5,4,7)
        t4_abaa += t1t1t2d.transpose(2,1,0,3,6,5,4,7)
        t4_abaa += t1t1t2c.transpose(3,1,2,0,4,5,6,7)
        t4_abaa -= t1t1t2d.transpose(2,1,0,3,4,5,6,7)
        t4_abaa -= t1t1t2c.transpose(3,1,2,0,4,5,7,6)
        t4_abaa += t1t1t2d.transpose(2,1,0,3,4,5,7,6) 

        # (t1 t1 t1 t1) terms + permutations
        t1t1t1t1 = einsum('ia, jb, kc, ld -> ijklabcd', t1, t1, t1, t1) 

        t4_abaa -= t1t1t1t1
        t4_abaa += t1t1t1t1.transpose(0,1,2,3,4,5,7,6)
        t4_abaa += t1t1t1t1.transpose(0,1,2,3,6,5,4,7)
        t4_abaa -= t1t1t1t1.transpose(0,1,3,2,6,5,4,7)
        t4_abaa -= t1t1t1t1.transpose(0,1,3,2,7,5,6,4)
        t4_abaa += t1t1t1t1.transpose(0,1,2,3,7,5,6,4)

        # Now construct t4 with spin signature (abab -> abab)
        t4_abab = c4_abab/self.c0

        # (t1 t3) terms + permutations
        t1t3a = einsum('ia, jklbcd -> ijklabcd', t1, t3)
        t1t3b = einsum('jd, ilkabc -> ijklabcd', t1, t3)

        t4_abab -= t1t3a
        t4_abab += t1t3a.transpose(0,1,2,3,6,5,4,7)
        t4_abab -= t1t3a.transpose(1,0,3,2,5,4,7,6)
        t4_abab += t1t3b
        t4_abab += t1t3a.transpose(2,1,0,3,4,5,6,7)
        t4_abab -= t1t3a.transpose(2,1,0,3,6,5,4,7)
        t4_abab += t1t3b.transpose(0,3,2,1,4,7,6,5)
        t4_abab -= t1t3b.transpose(0,3,2,1,4,5,6,7)

        # (t2 t2) terms + permutations
        t2t2 = einsum('ijab, klcd -> ijklabcd', t2, t2)

        t4_abab -= t2t2
        t4_abab += t2t2.transpose(0,1,2,3,4,7,6,5)
        t4_abab += t2t2.transpose(0,1,2,3,6,5,4,7)
        t4_abab -= t2t2.transpose(0,1,2,3,6,7,4,5)
        t4_abab += t2t2.transpose(0,3,2,1,4,5,6,7)
        t4_abab -= t2t2.transpose(0,3,2,1,6,5,4,7)
        t4_abab += t2t2.transpose(0,3,2,1,6,7,4,5)
        t4_abab -= einsum('ilad, jkbc -> ijklabcd', t2, t2)
        t4_abab -= einsum('ikac, jlbd -> ijklabcd', t2aa, t2aa) 

        # (t1 t1 t2) terms + permutations
        t1t1t2a = einsum('ia,jb,klcd -> ijklabcd', t1, t1, t2) 
        t1t1t2b = einsum('jb,ka,ilcd -> ijklabcd', t1, t1, t2)

        t4_abab -= t1t1t2a
        t4_abab += t1t1t2a.transpose(0,1,2,3,4,7,6,5)
        t4_abab += t1t1t2a.transpose(0,3,2,1,4,5,6,7)
        t4_abab += t1t1t2a.transpose(0,1,2,3,6,5,4,7)
        t4_abab -= t1t1t2a.transpose(0,1,2,3,6,7,4,5)
        t4_abab -= t1t1t2a.transpose(0,3,2,1,6,5,4,7)
        t4_abab += t1t1t2a.transpose(0,3,2,1,6,7,4,5)
        t4_abab -= t1t1t2a.transpose(2,3,0,1,4,5,6,7)
        t4_abab += t1t1t2a.transpose(2,3,0,1,4,7,6,5)
        t4_abab += t1t1t2a.transpose(2,3,0,1,6,5,4,7)
        t4_abab -= t1t1t2a.transpose(2,3,0,1,6,7,4,5)
        t4_abab += t1t1t2b
        t4_abab -= t1t1t2b.transpose(0,1,2,3,6,5,4,7)
        t4_abab -= t1t1t2b.transpose(0,1,2,3,4,7,6,5)
        t4_abab += t1t1t2b.transpose(0,1,2,3,6,7,4,5)
        t4_abab -= t1t1t2b.transpose(2,3,0,1,4,7,6,5)
        t4_abab -= einsum('ia,kc,jlbd -> ijklabcd', t1, t1, t2aa)
        t4_abab += einsum('ic,ka,jlbd -> ijklabcd', t1, t1, t2aa)
        t4_abab -= einsum('jb,ld,ikac -> ijklabcd', t1, t1, t2aa)
        t4_abab += einsum('jd,lb,ikac -> ijklabcd', t1, t1, t2aa)

        # (t1 t1 t1 t1) terms + permutations
        t1t1t1t1 = einsum('ia,jb,kc,ld -> ijklabcd', t1, t1, t1, t1)

        t4_abab -= t1t1t1t1
        t4_abab += t1t1t1t1.transpose(0,1,2,3,4,7,6,5)
        t4_abab += t1t1t1t1.transpose(0,1,2,3,6,5,4,7)
        t4_abab -= t1t1t1t1.transpose(0,1,2,3,6,7,4,5)

        # Pack two spin signatures into single tuple
        t4 = (t4_abaa, t4_abab)
        
        return wf_types.RCCSDTQ_WaveFunction(self.mo, t1=t1, t2=t2, t3=t3, t4=t4)

class UCISDTQ_WaveFunction(wf_types.WaveFunction):

    def __init__(self, mo, c0, c1, c2, c3, c4):
        super().__init__(mo)
        self.c0 = c0
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.c4 = c4
        # FIXME I've just disabled these checks for now - fix later
        #if not (isinstance(c3, tuple) and len(c3) == 6):
        #    raise ValueError("c4 definition in UCISDTQ wfn requires tuple of (aaa, aba, abb, bab, bba, bbb) spin signatures")
        #if not (isinstance(c4, tuple) and len(c4) == 8):
        #    raise ValueError("c4 definition in UCISDTQ wfn requires tuple of (aaaa, aaab, aaba, abaa, abab, bbab, bbba, bbbb) spin signatures")

    def as_ccsdtq(self):
        # TODO optimise these contractions
        # TODO remove redundant permutations
        c1_aa, c1_bb = (c / self.c0 for c in self.c1)
        c2_aaaa, c2_abab, c2_bbbb = (c / self.c0 for c in self.c2)
        c3_aaaaaa, c3_abaaba, c3_abbabb, c3_babbab, c3_bbabba, c3_bbbbbb = (c / self.c0 for c in self.c3)
        c4_aaaaaaaa, c4_aaabaaab, c4_aabaaaba, c4_abaaabaa, c4_abababab, c4_bbabbbab, c4_bbbabbba, c4_bbbbbbbb = (c / self.c0 for c in self.c4)

        nocc = self.nocc
        nvir = self.nvir

        t1_aa = np.zeros((nocc[0], nvir[0]), dtype=np.float64)
        t1_aa += einsum("ia->ia", c1_aa)

        t1_bb = np.zeros((nocc[1], nvir[1]), dtype=np.float64)
        t1_bb += einsum("ia->ia", c1_bb)

        t2_aaaa = np.zeros((nocc[0], nocc[0], nvir[0], nvir[0]), dtype=np.float64)
        t2_aaaa += einsum("ijab->ijab", c2_aaaa) * 2.0
        t2_aaaa += einsum("ia,jb->ijab", t1_aa, t1_aa) * -1.0
        t2_aaaa += einsum("ib,ja->ijab", t1_aa, t1_aa)

        t2_abab = np.zeros((nocc[0], nocc[1], nvir[0], nvir[1]), dtype=np.float64)
        t2_abab += einsum("ijab->ijab", c2_abab)
        t2_abab += einsum("ia,jb->ijab", t1_aa, t1_bb) * -1.0

        t2_bbbb = np.zeros((nocc[1], nocc[1], nvir[1], nvir[1]), dtype=np.float64)
        t2_bbbb += einsum("ijab->ijab", c2_bbbb) * 2.0
        t2_bbbb += einsum("ia,jb->ijab", t1_bb, t1_bb) * -1.0
        t2_bbbb += einsum("ib,ja->ijab", t1_bb, t1_bb)

        t3_aaaaaa = np.zeros((nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0]), dtype=np.float64)
        t3_aaaaaa += einsum("ijkabc->ijkabc", c3_aaaaaa) * 6.0
        t3_aaaaaa += einsum("ia,jkbc->ijkabc", t1_aa, t2_aaaa) * -2.0
        t3_aaaaaa += einsum("ib,jkac->ijkabc", t1_aa, t2_aaaa) * 2.0
        t3_aaaaaa += einsum("ic,jkab->ijkabc", t1_aa, t2_aaaa) * -2.0
        t3_aaaaaa += einsum("ja,ikbc->ijkabc", t1_aa, t2_aaaa) * 2.0
        t3_aaaaaa += einsum("jb,ikac->ijkabc", t1_aa, t2_aaaa) * -2.0
        t3_aaaaaa += einsum("jc,ikab->ijkabc", t1_aa, t2_aaaa) * 2.0
        t3_aaaaaa += einsum("ka,ijbc->ijkabc", t1_aa, t2_aaaa) * -2.0
        t3_aaaaaa += einsum("kb,ijac->ijkabc", t1_aa, t2_aaaa) * 2.0
        t3_aaaaaa += einsum("kc,ijab->ijkabc", t1_aa, t2_aaaa) * -2.0
        t3_aaaaaa += einsum("ia,jb,kc->ijkabc", t1_aa, t1_aa, t1_aa) * -1.0
        t3_aaaaaa += einsum("ia,jc,kb->ijkabc", t1_aa, t1_aa, t1_aa)
        t3_aaaaaa += einsum("ib,ja,kc->ijkabc", t1_aa, t1_aa, t1_aa)
        t3_aaaaaa += einsum("ib,jc,ka->ijkabc", t1_aa, t1_aa, t1_aa) * -1.0
        t3_aaaaaa += einsum("ic,ja,kb->ijkabc", t1_aa, t1_aa, t1_aa) * -1.0
        t3_aaaaaa += einsum("ic,jb,ka->ijkabc", t1_aa, t1_aa, t1_aa)

        t3_abaaba = np.zeros((nocc[0], nocc[1], nocc[0], nvir[0], nvir[1], nvir[0]), dtype=np.float64)
        t3_abaaba += einsum("ijkabc->ijkabc", c3_abaaba) * 2.0
        t3_abaaba += einsum("ia,kjcb->ijkabc", t1_aa, t2_abab) * -1.0
        t3_abaaba += einsum("ic,kjab->ijkabc", t1_aa, t2_abab)
        t3_abaaba += einsum("ka,ijcb->ijkabc", t1_aa, t2_abab)
        t3_abaaba += einsum("kc,ijab->ijkabc", t1_aa, t2_abab) * -1.0
        t3_abaaba += einsum("jb,ikac->ijkabc", t1_bb, t2_aaaa) * -2.0
        t3_abaaba += einsum("ia,kc,jb->ijkabc", t1_aa, t1_aa, t1_bb) * -1.0
        t3_abaaba += einsum("ic,ka,jb->ijkabc", t1_aa, t1_aa, t1_bb)

        t3_abbabb = np.zeros((nocc[0], nocc[1], nocc[1], nvir[0], nvir[1], nvir[1]), dtype=np.float64)
        t3_abbabb += einsum("ijkabc->ijkabc", c3_abbabb) * 2.0
        t3_abbabb += einsum("ia,jkbc->ijkabc", t1_aa, t2_bbbb) * -2.0
        t3_abbabb += einsum("jb,ikac->ijkabc", t1_bb, t2_abab) * -1.0
        t3_abbabb += einsum("jc,ikab->ijkabc", t1_bb, t2_abab)
        t3_abbabb += einsum("kb,ijac->ijkabc", t1_bb, t2_abab)
        t3_abbabb += einsum("kc,ijab->ijkabc", t1_bb, t2_abab) * -1.0
        t3_abbabb += einsum("ia,jb,kc->ijkabc", t1_aa, t1_bb, t1_bb) * -1.0
        t3_abbabb += einsum("ia,jc,kb->ijkabc", t1_aa, t1_bb, t1_bb)

        t3_babbab = np.zeros((nocc[1], nocc[0], nocc[1], nvir[1], nvir[0], nvir[1]), dtype=np.float64)
        t3_babbab += einsum("ijkabc->ijkabc", c3_babbab) * 2.0
        t3_babbab += einsum("jb,ikac->ijkabc", t1_aa, t2_bbbb) * -2.0
        t3_babbab += einsum("ia,jkbc->ijkabc", t1_bb, t2_abab) * -1.0
        t3_babbab += einsum("ic,jkba->ijkabc", t1_bb, t2_abab)
        t3_babbab += einsum("ka,jibc->ijkabc", t1_bb, t2_abab)
        t3_babbab += einsum("kc,jiba->ijkabc", t1_bb, t2_abab) * -1.0
        t3_babbab += einsum("jb,ia,kc->ijkabc", t1_aa, t1_bb, t1_bb) * -1.0
        t3_babbab += einsum("jb,ic,ka->ijkabc", t1_aa, t1_bb, t1_bb)

        t3_bbabba = np.zeros((nocc[1], nocc[1], nocc[0], nvir[1], nvir[1], nvir[0]), dtype=np.float64)
        t3_bbabba += einsum("ijkabc->ijkabc", c3_bbabba) * 2.0
        t3_bbabba += einsum("kc,ijab->ijkabc", t1_aa, t2_bbbb) * -2.0
        t3_bbabba += einsum("ia,kjcb->ijkabc", t1_bb, t2_abab) * -1.0
        t3_bbabba += einsum("ib,kjca->ijkabc", t1_bb, t2_abab)
        t3_bbabba += einsum("ja,kicb->ijkabc", t1_bb, t2_abab)
        t3_bbabba += einsum("jb,kica->ijkabc", t1_bb, t2_abab) * -1.0
        t3_bbabba += einsum("kc,ia,jb->ijkabc", t1_aa, t1_bb, t1_bb) * -1.0
        t3_bbabba += einsum("kc,ib,ja->ijkabc", t1_aa, t1_bb, t1_bb)

        t3_bbbbbb = np.zeros((nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1]), dtype=np.float64)
        t3_bbbbbb += einsum("ijkabc->ijkabc", c3_bbbbbb) * 6.0
        t3_bbbbbb += einsum("ia,jkbc->ijkabc", t1_bb, t2_bbbb) * -2.0
        t3_bbbbbb += einsum("ib,jkac->ijkabc", t1_bb, t2_bbbb) * 2.0
        t3_bbbbbb += einsum("ic,jkab->ijkabc", t1_bb, t2_bbbb) * -2.0
        t3_bbbbbb += einsum("ja,ikbc->ijkabc", t1_bb, t2_bbbb) * 2.0
        t3_bbbbbb += einsum("jb,ikac->ijkabc", t1_bb, t2_bbbb) * -2.0
        t3_bbbbbb += einsum("jc,ikab->ijkabc", t1_bb, t2_bbbb) * 2.0
        t3_bbbbbb += einsum("ka,ijbc->ijkabc", t1_bb, t2_bbbb) * -2.0
        t3_bbbbbb += einsum("kb,ijac->ijkabc", t1_bb, t2_bbbb) * 2.0
        t3_bbbbbb += einsum("kc,ijab->ijkabc", t1_bb, t2_bbbb) * -2.0
        t3_bbbbbb += einsum("ia,jb,kc->ijkabc", t1_bb, t1_bb, t1_bb) * -1.0
        t3_bbbbbb += einsum("ia,jc,kb->ijkabc", t1_bb, t1_bb, t1_bb)
        t3_bbbbbb += einsum("ib,ja,kc->ijkabc", t1_bb, t1_bb, t1_bb)
        t3_bbbbbb += einsum("ib,jc,ka->ijkabc", t1_bb, t1_bb, t1_bb) * -1.0
        t3_bbbbbb += einsum("ic,ja,kb->ijkabc", t1_bb, t1_bb, t1_bb) * -1.0
        t3_bbbbbb += einsum("ic,jb,ka->ijkabc", t1_bb, t1_bb, t1_bb)

        t4_aaaaaaaa = np.zeros((nocc[0], nocc[0], nocc[0], nocc[0], nvir[0], nvir[0], nvir[0], nvir[0]), dtype=np.float64)
        t4_aaaaaaaa += einsum("ijklabcd->ijklabcd", c4_aaaaaaaa) * 24.0
        t4_aaaaaaaa += einsum("ia,jklbcd->ijklabcd", t1_aa, t3_aaaaaa) * -6.0
        t4_aaaaaaaa += einsum("ib,jklacd->ijklabcd", t1_aa, t3_aaaaaa) * 6.0
        t4_aaaaaaaa += einsum("ic,jklabd->ijklabcd", t1_aa, t3_aaaaaa) * -6.0
        t4_aaaaaaaa += einsum("id,jklabc->ijklabcd", t1_aa, t3_aaaaaa) * 6.0
        t4_aaaaaaaa += einsum("ja,iklbcd->ijklabcd", t1_aa, t3_aaaaaa) * 6.0
        t4_aaaaaaaa += einsum("jb,iklacd->ijklabcd", t1_aa, t3_aaaaaa) * -6.0
        t4_aaaaaaaa += einsum("jc,iklabd->ijklabcd", t1_aa, t3_aaaaaa) * 6.0
        t4_aaaaaaaa += einsum("jd,iklabc->ijklabcd", t1_aa, t3_aaaaaa) * -6.0
        t4_aaaaaaaa += einsum("ka,ijlbcd->ijklabcd", t1_aa, t3_aaaaaa) * -6.0
        t4_aaaaaaaa += einsum("kb,ijlacd->ijklabcd", t1_aa, t3_aaaaaa) * 6.0
        t4_aaaaaaaa += einsum("kc,ijlabd->ijklabcd", t1_aa, t3_aaaaaa) * -6.0
        t4_aaaaaaaa += einsum("kd,ijlabc->ijklabcd", t1_aa, t3_aaaaaa) * 6.0
        t4_aaaaaaaa += einsum("la,ijkbcd->ijklabcd", t1_aa, t3_aaaaaa) * 6.0
        t4_aaaaaaaa += einsum("lb,ijkacd->ijklabcd", t1_aa, t3_aaaaaa) * -6.0
        t4_aaaaaaaa += einsum("lc,ijkabd->ijklabcd", t1_aa, t3_aaaaaa) * 6.0
        t4_aaaaaaaa += einsum("ld,ijkabc->ijklabcd", t1_aa, t3_aaaaaa) * -6.0
        t4_aaaaaaaa += einsum("ikac,jlbd->ijklabcd", t2_aaaa, t2_aaaa) * -4.0
        t4_aaaaaaaa += einsum("ikad,jlbc->ijklabcd", t2_aaaa, t2_aaaa) * 4.0
        t4_aaaaaaaa += einsum("ikab,jlcd->ijklabcd", t2_aaaa, t2_aaaa) * 4.0
        t4_aaaaaaaa += einsum("ikbc,jlad->ijklabcd", t2_aaaa, t2_aaaa) * 4.0
        t4_aaaaaaaa += einsum("ikbd,jlac->ijklabcd", t2_aaaa, t2_aaaa) * -4.0
        t4_aaaaaaaa += einsum("ikcd,jlab->ijklabcd", t2_aaaa, t2_aaaa) * 4.0
        t4_aaaaaaaa += einsum("ilac,jkbd->ijklabcd", t2_aaaa, t2_aaaa) * 4.0
        t4_aaaaaaaa += einsum("ilad,jkbc->ijklabcd", t2_aaaa, t2_aaaa) * -4.0
        t4_aaaaaaaa += einsum("ilab,jkcd->ijklabcd", t2_aaaa, t2_aaaa) * -4.0
        t4_aaaaaaaa += einsum("ilbc,jkad->ijklabcd", t2_aaaa, t2_aaaa) * -4.0
        t4_aaaaaaaa += einsum("ilbd,jkac->ijklabcd", t2_aaaa, t2_aaaa) * 4.0
        t4_aaaaaaaa += einsum("ilcd,jkab->ijklabcd", t2_aaaa, t2_aaaa) * -4.0
        t4_aaaaaaaa += einsum("ijac,klbd->ijklabcd", t2_aaaa, t2_aaaa) * 4.0
        t4_aaaaaaaa += einsum("ijad,klbc->ijklabcd", t2_aaaa, t2_aaaa) * -4.0
        t4_aaaaaaaa += einsum("ijab,klcd->ijklabcd", t2_aaaa, t2_aaaa) * -4.0
        t4_aaaaaaaa += einsum("ijbc,klad->ijklabcd", t2_aaaa, t2_aaaa) * -4.0
        t4_aaaaaaaa += einsum("ijbd,klac->ijklabcd", t2_aaaa, t2_aaaa) * 4.0
        t4_aaaaaaaa += einsum("ijcd,klab->ijklabcd", t2_aaaa, t2_aaaa) * -4.0
        t4_aaaaaaaa += einsum("ia,jb,klcd->ijklabcd", t1_aa, t1_aa, t2_aaaa) * -2.0
        t4_aaaaaaaa += einsum("ia,jc,klbd->ijklabcd", t1_aa, t1_aa, t2_aaaa) * 2.0
        t4_aaaaaaaa += einsum("ia,jd,klbc->ijklabcd", t1_aa, t1_aa, t2_aaaa) * -2.0
        t4_aaaaaaaa += einsum("ib,ja,klcd->ijklabcd", t1_aa, t1_aa, t2_aaaa) * 2.0
        t4_aaaaaaaa += einsum("ib,jc,klad->ijklabcd", t1_aa, t1_aa, t2_aaaa) * -2.0
        t4_aaaaaaaa += einsum("ib,jd,klac->ijklabcd", t1_aa, t1_aa, t2_aaaa) * 2.0
        t4_aaaaaaaa += einsum("ic,ja,klbd->ijklabcd", t1_aa, t1_aa, t2_aaaa) * -2.0
        t4_aaaaaaaa += einsum("ic,jb,klad->ijklabcd", t1_aa, t1_aa, t2_aaaa) * 2.0
        t4_aaaaaaaa += einsum("ic,jd,klab->ijklabcd", t1_aa, t1_aa, t2_aaaa) * -2.0
        t4_aaaaaaaa += einsum("id,ja,klbc->ijklabcd", t1_aa, t1_aa, t2_aaaa) * 2.0
        t4_aaaaaaaa += einsum("id,jb,klac->ijklabcd", t1_aa, t1_aa, t2_aaaa) * -2.0
        t4_aaaaaaaa += einsum("id,jc,klab->ijklabcd", t1_aa, t1_aa, t2_aaaa) * 2.0
        t4_aaaaaaaa += einsum("ia,kb,jlcd->ijklabcd", t1_aa, t1_aa, t2_aaaa) * 2.0
        t4_aaaaaaaa += einsum("ia,kc,jlbd->ijklabcd", t1_aa, t1_aa, t2_aaaa) * -2.0
        t4_aaaaaaaa += einsum("ia,kd,jlbc->ijklabcd", t1_aa, t1_aa, t2_aaaa) * 2.0
        t4_aaaaaaaa += einsum("ib,ka,jlcd->ijklabcd", t1_aa, t1_aa, t2_aaaa) * -2.0
        t4_aaaaaaaa += einsum("ib,kc,jlad->ijklabcd", t1_aa, t1_aa, t2_aaaa) * 2.0
        t4_aaaaaaaa += einsum("ib,kd,jlac->ijklabcd", t1_aa, t1_aa, t2_aaaa) * -2.0
        t4_aaaaaaaa += einsum("ic,ka,jlbd->ijklabcd", t1_aa, t1_aa, t2_aaaa) * 2.0
        t4_aaaaaaaa += einsum("ic,kb,jlad->ijklabcd", t1_aa, t1_aa, t2_aaaa) * -2.0
        t4_aaaaaaaa += einsum("ic,kd,jlab->ijklabcd", t1_aa, t1_aa, t2_aaaa) * 2.0
        t4_aaaaaaaa += einsum("id,ka,jlbc->ijklabcd", t1_aa, t1_aa, t2_aaaa) * -2.0
        t4_aaaaaaaa += einsum("id,kb,jlac->ijklabcd", t1_aa, t1_aa, t2_aaaa) * 2.0
        t4_aaaaaaaa += einsum("id,kc,jlab->ijklabcd", t1_aa, t1_aa, t2_aaaa) * -2.0
        t4_aaaaaaaa += einsum("ia,lb,jkcd->ijklabcd", t1_aa, t1_aa, t2_aaaa) * -2.0
        t4_aaaaaaaa += einsum("ia,lc,jkbd->ijklabcd", t1_aa, t1_aa, t2_aaaa) * 2.0
        t4_aaaaaaaa += einsum("ia,ld,jkbc->ijklabcd", t1_aa, t1_aa, t2_aaaa) * -2.0
        t4_aaaaaaaa += einsum("ib,la,jkcd->ijklabcd", t1_aa, t1_aa, t2_aaaa) * 2.0
        t4_aaaaaaaa += einsum("ib,lc,jkad->ijklabcd", t1_aa, t1_aa, t2_aaaa) * -2.0
        t4_aaaaaaaa += einsum("ib,ld,jkac->ijklabcd", t1_aa, t1_aa, t2_aaaa) * 2.0
        t4_aaaaaaaa += einsum("ic,la,jkbd->ijklabcd", t1_aa, t1_aa, t2_aaaa) * -2.0
        t4_aaaaaaaa += einsum("ic,lb,jkad->ijklabcd", t1_aa, t1_aa, t2_aaaa) * 2.0
        t4_aaaaaaaa += einsum("ic,ld,jkab->ijklabcd", t1_aa, t1_aa, t2_aaaa) * -2.0
        t4_aaaaaaaa += einsum("id,la,jkbc->ijklabcd", t1_aa, t1_aa, t2_aaaa) * 2.0
        t4_aaaaaaaa += einsum("id,lb,jkac->ijklabcd", t1_aa, t1_aa, t2_aaaa) * -2.0
        t4_aaaaaaaa += einsum("id,lc,jkab->ijklabcd", t1_aa, t1_aa, t2_aaaa) * 2.0
        t4_aaaaaaaa += einsum("ja,kb,ilcd->ijklabcd", t1_aa, t1_aa, t2_aaaa) * -2.0
        t4_aaaaaaaa += einsum("ja,kc,ilbd->ijklabcd", t1_aa, t1_aa, t2_aaaa) * 2.0
        t4_aaaaaaaa += einsum("ja,kd,ilbc->ijklabcd", t1_aa, t1_aa, t2_aaaa) * -2.0
        t4_aaaaaaaa += einsum("jb,ka,ilcd->ijklabcd", t1_aa, t1_aa, t2_aaaa) * 2.0
        t4_aaaaaaaa += einsum("jb,kc,ilad->ijklabcd", t1_aa, t1_aa, t2_aaaa) * -2.0
        t4_aaaaaaaa += einsum("jb,kd,ilac->ijklabcd", t1_aa, t1_aa, t2_aaaa) * 2.0
        t4_aaaaaaaa += einsum("jc,ka,ilbd->ijklabcd", t1_aa, t1_aa, t2_aaaa) * -2.0
        t4_aaaaaaaa += einsum("jc,kb,ilad->ijklabcd", t1_aa, t1_aa, t2_aaaa) * 2.0
        t4_aaaaaaaa += einsum("jc,kd,ilab->ijklabcd", t1_aa, t1_aa, t2_aaaa) * -2.0
        t4_aaaaaaaa += einsum("jd,ka,ilbc->ijklabcd", t1_aa, t1_aa, t2_aaaa) * 2.0
        t4_aaaaaaaa += einsum("jd,kb,ilac->ijklabcd", t1_aa, t1_aa, t2_aaaa) * -2.0
        t4_aaaaaaaa += einsum("jd,kc,ilab->ijklabcd", t1_aa, t1_aa, t2_aaaa) * 2.0
        t4_aaaaaaaa += einsum("ja,lb,ikcd->ijklabcd", t1_aa, t1_aa, t2_aaaa) * 2.0
        t4_aaaaaaaa += einsum("ja,lc,ikbd->ijklabcd", t1_aa, t1_aa, t2_aaaa) * -2.0
        t4_aaaaaaaa += einsum("ja,ld,ikbc->ijklabcd", t1_aa, t1_aa, t2_aaaa) * 2.0
        t4_aaaaaaaa += einsum("jb,la,ikcd->ijklabcd", t1_aa, t1_aa, t2_aaaa) * -2.0
        t4_aaaaaaaa += einsum("jb,lc,ikad->ijklabcd", t1_aa, t1_aa, t2_aaaa) * 2.0
        t4_aaaaaaaa += einsum("jb,ld,ikac->ijklabcd", t1_aa, t1_aa, t2_aaaa) * -2.0
        t4_aaaaaaaa += einsum("jc,la,ikbd->ijklabcd", t1_aa, t1_aa, t2_aaaa) * 2.0
        t4_aaaaaaaa += einsum("jc,lb,ikad->ijklabcd", t1_aa, t1_aa, t2_aaaa) * -2.0
        t4_aaaaaaaa += einsum("jc,ld,ikab->ijklabcd", t1_aa, t1_aa, t2_aaaa) * 2.0
        t4_aaaaaaaa += einsum("jd,la,ikbc->ijklabcd", t1_aa, t1_aa, t2_aaaa) * -2.0
        t4_aaaaaaaa += einsum("jd,lb,ikac->ijklabcd", t1_aa, t1_aa, t2_aaaa) * 2.0
        t4_aaaaaaaa += einsum("jd,lc,ikab->ijklabcd", t1_aa, t1_aa, t2_aaaa) * -2.0
        t4_aaaaaaaa += einsum("ka,lb,ijcd->ijklabcd", t1_aa, t1_aa, t2_aaaa) * -2.0
        t4_aaaaaaaa += einsum("ka,lc,ijbd->ijklabcd", t1_aa, t1_aa, t2_aaaa) * 2.0
        t4_aaaaaaaa += einsum("ka,ld,ijbc->ijklabcd", t1_aa, t1_aa, t2_aaaa) * -2.0
        t4_aaaaaaaa += einsum("kb,la,ijcd->ijklabcd", t1_aa, t1_aa, t2_aaaa) * 2.0
        t4_aaaaaaaa += einsum("kb,lc,ijad->ijklabcd", t1_aa, t1_aa, t2_aaaa) * -2.0
        t4_aaaaaaaa += einsum("kb,ld,ijac->ijklabcd", t1_aa, t1_aa, t2_aaaa) * 2.0
        t4_aaaaaaaa += einsum("kc,la,ijbd->ijklabcd", t1_aa, t1_aa, t2_aaaa) * -2.0
        t4_aaaaaaaa += einsum("kc,lb,ijad->ijklabcd", t1_aa, t1_aa, t2_aaaa) * 2.0
        t4_aaaaaaaa += einsum("kc,ld,ijab->ijklabcd", t1_aa, t1_aa, t2_aaaa) * -2.0
        t4_aaaaaaaa += einsum("kd,la,ijbc->ijklabcd", t1_aa, t1_aa, t2_aaaa) * 2.0
        t4_aaaaaaaa += einsum("kd,lb,ijac->ijklabcd", t1_aa, t1_aa, t2_aaaa) * -2.0
        t4_aaaaaaaa += einsum("kd,lc,ijab->ijklabcd", t1_aa, t1_aa, t2_aaaa) * 2.0
        t4_aaaaaaaa += einsum("ia,jb,kc,ld->ijklabcd", t1_aa, t1_aa, t1_aa, t1_aa) * -1.0
        t4_aaaaaaaa += einsum("ia,jb,kd,lc->ijklabcd", t1_aa, t1_aa, t1_aa, t1_aa)
        t4_aaaaaaaa += einsum("ia,jc,kb,ld->ijklabcd", t1_aa, t1_aa, t1_aa, t1_aa)
        t4_aaaaaaaa += einsum("ia,jc,kd,lb->ijklabcd", t1_aa, t1_aa, t1_aa, t1_aa) * -1.0
        t4_aaaaaaaa += einsum("ia,jd,kb,lc->ijklabcd", t1_aa, t1_aa, t1_aa, t1_aa) * -1.0
        t4_aaaaaaaa += einsum("ia,jd,kc,lb->ijklabcd", t1_aa, t1_aa, t1_aa, t1_aa)
        t4_aaaaaaaa += einsum("ib,ja,kc,ld->ijklabcd", t1_aa, t1_aa, t1_aa, t1_aa)
        t4_aaaaaaaa += einsum("ib,ja,kd,lc->ijklabcd", t1_aa, t1_aa, t1_aa, t1_aa) * -1.0
        t4_aaaaaaaa += einsum("ib,jc,ka,ld->ijklabcd", t1_aa, t1_aa, t1_aa, t1_aa) * -1.0
        t4_aaaaaaaa += einsum("ib,jc,kd,la->ijklabcd", t1_aa, t1_aa, t1_aa, t1_aa)
        t4_aaaaaaaa += einsum("ib,jd,ka,lc->ijklabcd", t1_aa, t1_aa, t1_aa, t1_aa)
        t4_aaaaaaaa += einsum("ib,jd,kc,la->ijklabcd", t1_aa, t1_aa, t1_aa, t1_aa) * -1.0
        t4_aaaaaaaa += einsum("ic,ja,kb,ld->ijklabcd", t1_aa, t1_aa, t1_aa, t1_aa) * -1.0
        t4_aaaaaaaa += einsum("ic,ja,kd,lb->ijklabcd", t1_aa, t1_aa, t1_aa, t1_aa)
        t4_aaaaaaaa += einsum("ic,jb,ka,ld->ijklabcd", t1_aa, t1_aa, t1_aa, t1_aa)
        t4_aaaaaaaa += einsum("ic,jb,kd,la->ijklabcd", t1_aa, t1_aa, t1_aa, t1_aa) * -1.0
        t4_aaaaaaaa += einsum("ic,jd,ka,lb->ijklabcd", t1_aa, t1_aa, t1_aa, t1_aa) * -1.0
        t4_aaaaaaaa += einsum("ic,jd,kb,la->ijklabcd", t1_aa, t1_aa, t1_aa, t1_aa)
        t4_aaaaaaaa += einsum("id,ja,kb,lc->ijklabcd", t1_aa, t1_aa, t1_aa, t1_aa)
        t4_aaaaaaaa += einsum("id,ja,kc,lb->ijklabcd", t1_aa, t1_aa, t1_aa, t1_aa) * -1.0
        t4_aaaaaaaa += einsum("id,jb,ka,lc->ijklabcd", t1_aa, t1_aa, t1_aa, t1_aa) * -1.0
        t4_aaaaaaaa += einsum("id,jb,kc,la->ijklabcd", t1_aa, t1_aa, t1_aa, t1_aa)
        t4_aaaaaaaa += einsum("id,jc,ka,lb->ijklabcd", t1_aa, t1_aa, t1_aa, t1_aa)
        t4_aaaaaaaa += einsum("id,jc,kb,la->ijklabcd", t1_aa, t1_aa, t1_aa, t1_aa) * -1.0

        t4_aaabaaab = np.zeros((nocc[0], nocc[0], nocc[0], nocc[1], nvir[0], nvir[0], nvir[0], nvir[1]), dtype=np.float64)
        t4_aaabaaab += einsum("ijklabcd->ijklabcd", c4_aaabaaab) * 6.0
        t4_aaabaaab += einsum("ia,jlkbdc->ijklabcd", t1_aa, t3_abaaba) * -2.0
        t4_aaabaaab += einsum("ib,jlkadc->ijklabcd", t1_aa, t3_abaaba) * 2.0
        t4_aaabaaab += einsum("ic,jlkadb->ijklabcd", t1_aa, t3_abaaba) * -2.0
        t4_aaabaaab += einsum("ja,ilkbdc->ijklabcd", t1_aa, t3_abaaba) * 2.0
        t4_aaabaaab += einsum("jb,ilkadc->ijklabcd", t1_aa, t3_abaaba) * -2.0
        t4_aaabaaab += einsum("jc,ilkadb->ijklabcd", t1_aa, t3_abaaba) * 2.0
        t4_aaabaaab += einsum("ka,iljbdc->ijklabcd", t1_aa, t3_abaaba) * -2.0
        t4_aaabaaab += einsum("kb,iljadc->ijklabcd", t1_aa, t3_abaaba) * 2.0
        t4_aaabaaab += einsum("kc,iljadb->ijklabcd", t1_aa, t3_abaaba) * -2.0
        t4_aaabaaab += einsum("ld,ijkabc->ijklabcd", t1_bb, t3_aaaaaa) * -6.0
        t4_aaabaaab += einsum("ilad,jkbc->ijklabcd", t2_abab, t2_aaaa) * -2.0
        t4_aaabaaab += einsum("ilbd,jkac->ijklabcd", t2_abab, t2_aaaa) * 2.0
        t4_aaabaaab += einsum("ilcd,jkab->ijklabcd", t2_abab, t2_aaaa) * -2.0
        t4_aaabaaab += einsum("jlad,ikbc->ijklabcd", t2_abab, t2_aaaa) * 2.0
        t4_aaabaaab += einsum("jlbd,ikac->ijklabcd", t2_abab, t2_aaaa) * -2.0
        t4_aaabaaab += einsum("jlcd,ikab->ijklabcd", t2_abab, t2_aaaa) * 2.0
        t4_aaabaaab += einsum("klad,ijbc->ijklabcd", t2_abab, t2_aaaa) * -2.0
        t4_aaabaaab += einsum("klbd,ijac->ijklabcd", t2_abab, t2_aaaa) * 2.0
        t4_aaabaaab += einsum("klcd,ijab->ijklabcd", t2_abab, t2_aaaa) * -2.0
        t4_aaabaaab += einsum("ia,jb,klcd->ijklabcd", t1_aa, t1_aa, t2_abab) * -1.0
        t4_aaabaaab += einsum("ia,jc,klbd->ijklabcd", t1_aa, t1_aa, t2_abab)
        t4_aaabaaab += einsum("ib,ja,klcd->ijklabcd", t1_aa, t1_aa, t2_abab)
        t4_aaabaaab += einsum("ib,jc,klad->ijklabcd", t1_aa, t1_aa, t2_abab) * -1.0
        t4_aaabaaab += einsum("ic,ja,klbd->ijklabcd", t1_aa, t1_aa, t2_abab) * -1.0
        t4_aaabaaab += einsum("ic,jb,klad->ijklabcd", t1_aa, t1_aa, t2_abab)
        t4_aaabaaab += einsum("ia,kb,jlcd->ijklabcd", t1_aa, t1_aa, t2_abab)
        t4_aaabaaab += einsum("ia,kc,jlbd->ijklabcd", t1_aa, t1_aa, t2_abab) * -1.0
        t4_aaabaaab += einsum("ib,ka,jlcd->ijklabcd", t1_aa, t1_aa, t2_abab) * -1.0
        t4_aaabaaab += einsum("ib,kc,jlad->ijklabcd", t1_aa, t1_aa, t2_abab)
        t4_aaabaaab += einsum("ic,ka,jlbd->ijklabcd", t1_aa, t1_aa, t2_abab)
        t4_aaabaaab += einsum("ic,kb,jlad->ijklabcd", t1_aa, t1_aa, t2_abab) * -1.0
        t4_aaabaaab += einsum("ja,kb,ilcd->ijklabcd", t1_aa, t1_aa, t2_abab) * -1.0
        t4_aaabaaab += einsum("ja,kc,ilbd->ijklabcd", t1_aa, t1_aa, t2_abab)
        t4_aaabaaab += einsum("jb,ka,ilcd->ijklabcd", t1_aa, t1_aa, t2_abab)
        t4_aaabaaab += einsum("jb,kc,ilad->ijklabcd", t1_aa, t1_aa, t2_abab) * -1.0
        t4_aaabaaab += einsum("jc,ka,ilbd->ijklabcd", t1_aa, t1_aa, t2_abab) * -1.0
        t4_aaabaaab += einsum("jc,kb,ilad->ijklabcd", t1_aa, t1_aa, t2_abab)
        t4_aaabaaab += einsum("ia,ld,jkbc->ijklabcd", t1_aa, t1_bb, t2_aaaa) * -2.0
        t4_aaabaaab += einsum("ib,ld,jkac->ijklabcd", t1_aa, t1_bb, t2_aaaa) * 2.0
        t4_aaabaaab += einsum("ic,ld,jkab->ijklabcd", t1_aa, t1_bb, t2_aaaa) * -2.0
        t4_aaabaaab += einsum("ja,ld,ikbc->ijklabcd", t1_aa, t1_bb, t2_aaaa) * 2.0
        t4_aaabaaab += einsum("jb,ld,ikac->ijklabcd", t1_aa, t1_bb, t2_aaaa) * -2.0
        t4_aaabaaab += einsum("jc,ld,ikab->ijklabcd", t1_aa, t1_bb, t2_aaaa) * 2.0
        t4_aaabaaab += einsum("ka,ld,ijbc->ijklabcd", t1_aa, t1_bb, t2_aaaa) * -2.0
        t4_aaabaaab += einsum("kb,ld,ijac->ijklabcd", t1_aa, t1_bb, t2_aaaa) * 2.0
        t4_aaabaaab += einsum("kc,ld,ijab->ijklabcd", t1_aa, t1_bb, t2_aaaa) * -2.0
        t4_aaabaaab += einsum("ia,jb,kc,ld->ijklabcd", t1_aa, t1_aa, t1_aa, t1_bb) * -1.0
        t4_aaabaaab += einsum("ia,jc,kb,ld->ijklabcd", t1_aa, t1_aa, t1_aa, t1_bb)
        t4_aaabaaab += einsum("ib,ja,kc,ld->ijklabcd", t1_aa, t1_aa, t1_aa, t1_bb)
        t4_aaabaaab += einsum("ib,jc,ka,ld->ijklabcd", t1_aa, t1_aa, t1_aa, t1_bb) * -1.0
        t4_aaabaaab += einsum("ic,ja,kb,ld->ijklabcd", t1_aa, t1_aa, t1_aa, t1_bb) * -1.0
        t4_aaabaaab += einsum("ic,jb,ka,ld->ijklabcd", t1_aa, t1_aa, t1_aa, t1_bb)

        t4_aabaaaba = np.zeros((nocc[0], nocc[0], nocc[1], nocc[0], nvir[0], nvir[0], nvir[1], nvir[0]), dtype=np.float64)
        t4_aabaaaba += einsum("ijklabcd->ijklabcd", c4_aabaaaba) * 6.0
        t4_aabaaaba += einsum("ia,jklbcd->ijklabcd", t1_aa, t3_abaaba) * -2.0
        t4_aabaaaba += einsum("ib,jklacd->ijklabcd", t1_aa, t3_abaaba) * 2.0
        t4_aabaaaba += einsum("id,jklacb->ijklabcd", t1_aa, t3_abaaba) * -2.0
        t4_aabaaaba += einsum("ja,iklbcd->ijklabcd", t1_aa, t3_abaaba) * 2.0
        t4_aabaaaba += einsum("jb,iklacd->ijklabcd", t1_aa, t3_abaaba) * -2.0
        t4_aabaaaba += einsum("jd,iklacb->ijklabcd", t1_aa, t3_abaaba) * 2.0
        t4_aabaaaba += einsum("la,ikjbcd->ijklabcd", t1_aa, t3_abaaba) * -2.0
        t4_aabaaaba += einsum("lb,ikjacd->ijklabcd", t1_aa, t3_abaaba) * 2.0
        t4_aabaaaba += einsum("ld,ikjacb->ijklabcd", t1_aa, t3_abaaba) * -2.0
        t4_aabaaaba += einsum("kc,ijlabd->ijklabcd", t1_bb, t3_aaaaaa) * -6.0
        t4_aabaaaba += einsum("ikac,jlbd->ijklabcd", t2_abab, t2_aaaa) * -2.0
        t4_aabaaaba += einsum("ikbc,jlad->ijklabcd", t2_abab, t2_aaaa) * 2.0
        t4_aabaaaba += einsum("ikdc,jlab->ijklabcd", t2_abab, t2_aaaa) * -2.0
        t4_aabaaaba += einsum("jkac,ilbd->ijklabcd", t2_abab, t2_aaaa) * 2.0
        t4_aabaaaba += einsum("jkbc,ilad->ijklabcd", t2_abab, t2_aaaa) * -2.0
        t4_aabaaaba += einsum("jkdc,ilab->ijklabcd", t2_abab, t2_aaaa) * 2.0
        t4_aabaaaba += einsum("lkac,ijbd->ijklabcd", t2_abab, t2_aaaa) * -2.0
        t4_aabaaaba += einsum("lkbc,ijad->ijklabcd", t2_abab, t2_aaaa) * 2.0
        t4_aabaaaba += einsum("lkdc,ijab->ijklabcd", t2_abab, t2_aaaa) * -2.0
        t4_aabaaaba += einsum("ia,jb,lkdc->ijklabcd", t1_aa, t1_aa, t2_abab) * -1.0
        t4_aabaaaba += einsum("ia,jd,lkbc->ijklabcd", t1_aa, t1_aa, t2_abab)
        t4_aabaaaba += einsum("ib,ja,lkdc->ijklabcd", t1_aa, t1_aa, t2_abab)
        t4_aabaaaba += einsum("ib,jd,lkac->ijklabcd", t1_aa, t1_aa, t2_abab) * -1.0
        t4_aabaaaba += einsum("id,ja,lkbc->ijklabcd", t1_aa, t1_aa, t2_abab) * -1.0
        t4_aabaaaba += einsum("id,jb,lkac->ijklabcd", t1_aa, t1_aa, t2_abab)
        t4_aabaaaba += einsum("ia,lb,jkdc->ijklabcd", t1_aa, t1_aa, t2_abab)
        t4_aabaaaba += einsum("ia,ld,jkbc->ijklabcd", t1_aa, t1_aa, t2_abab) * -1.0
        t4_aabaaaba += einsum("ib,la,jkdc->ijklabcd", t1_aa, t1_aa, t2_abab) * -1.0
        t4_aabaaaba += einsum("ib,ld,jkac->ijklabcd", t1_aa, t1_aa, t2_abab)
        t4_aabaaaba += einsum("id,la,jkbc->ijklabcd", t1_aa, t1_aa, t2_abab)
        t4_aabaaaba += einsum("id,lb,jkac->ijklabcd", t1_aa, t1_aa, t2_abab) * -1.0
        t4_aabaaaba += einsum("ja,lb,ikdc->ijklabcd", t1_aa, t1_aa, t2_abab) * -1.0
        t4_aabaaaba += einsum("ja,ld,ikbc->ijklabcd", t1_aa, t1_aa, t2_abab)
        t4_aabaaaba += einsum("jb,la,ikdc->ijklabcd", t1_aa, t1_aa, t2_abab)
        t4_aabaaaba += einsum("jb,ld,ikac->ijklabcd", t1_aa, t1_aa, t2_abab) * -1.0
        t4_aabaaaba += einsum("jd,la,ikbc->ijklabcd", t1_aa, t1_aa, t2_abab) * -1.0
        t4_aabaaaba += einsum("jd,lb,ikac->ijklabcd", t1_aa, t1_aa, t2_abab)
        t4_aabaaaba += einsum("ia,kc,jlbd->ijklabcd", t1_aa, t1_bb, t2_aaaa) * -2.0
        t4_aabaaaba += einsum("ib,kc,jlad->ijklabcd", t1_aa, t1_bb, t2_aaaa) * 2.0
        t4_aabaaaba += einsum("id,kc,jlab->ijklabcd", t1_aa, t1_bb, t2_aaaa) * -2.0
        t4_aabaaaba += einsum("ja,kc,ilbd->ijklabcd", t1_aa, t1_bb, t2_aaaa) * 2.0
        t4_aabaaaba += einsum("jb,kc,ilad->ijklabcd", t1_aa, t1_bb, t2_aaaa) * -2.0
        t4_aabaaaba += einsum("jd,kc,ilab->ijklabcd", t1_aa, t1_bb, t2_aaaa) * 2.0
        t4_aabaaaba += einsum("la,kc,ijbd->ijklabcd", t1_aa, t1_bb, t2_aaaa) * -2.0
        t4_aabaaaba += einsum("lb,kc,ijad->ijklabcd", t1_aa, t1_bb, t2_aaaa) * 2.0
        t4_aabaaaba += einsum("ld,kc,ijab->ijklabcd", t1_aa, t1_bb, t2_aaaa) * -2.0
        t4_aabaaaba += einsum("ia,jb,ld,kc->ijklabcd", t1_aa, t1_aa, t1_aa, t1_bb) * -1.0
        t4_aabaaaba += einsum("ia,jd,lb,kc->ijklabcd", t1_aa, t1_aa, t1_aa, t1_bb)
        t4_aabaaaba += einsum("ib,ja,ld,kc->ijklabcd", t1_aa, t1_aa, t1_aa, t1_bb)
        t4_aabaaaba += einsum("ib,jd,la,kc->ijklabcd", t1_aa, t1_aa, t1_aa, t1_bb) * -1.0
        t4_aabaaaba += einsum("id,ja,lb,kc->ijklabcd", t1_aa, t1_aa, t1_aa, t1_bb) * -1.0
        t4_aabaaaba += einsum("id,jb,la,kc->ijklabcd", t1_aa, t1_aa, t1_aa, t1_bb)

        t4_abaaabaa = np.zeros((nocc[0], nocc[1], nocc[0], nocc[0], nvir[0], nvir[1], nvir[0], nvir[0]), dtype=np.float64)
        t4_abaaabaa += einsum("ijklabcd->ijklabcd", c4_abaaabaa) * 6.0
        t4_abaaabaa += einsum("ia,kjlcbd->ijklabcd", t1_aa, t3_abaaba) * -2.0
        t4_abaaabaa += einsum("ic,kjlabd->ijklabcd", t1_aa, t3_abaaba) * 2.0
        t4_abaaabaa += einsum("id,kjlabc->ijklabcd", t1_aa, t3_abaaba) * -2.0
        t4_abaaabaa += einsum("ka,ijlcbd->ijklabcd", t1_aa, t3_abaaba) * 2.0
        t4_abaaabaa += einsum("kc,ijlabd->ijklabcd", t1_aa, t3_abaaba) * -2.0
        t4_abaaabaa += einsum("kd,ijlabc->ijklabcd", t1_aa, t3_abaaba) * 2.0
        t4_abaaabaa += einsum("la,ijkcbd->ijklabcd", t1_aa, t3_abaaba) * -2.0
        t4_abaaabaa += einsum("lc,ijkabd->ijklabcd", t1_aa, t3_abaaba) * 2.0
        t4_abaaabaa += einsum("ld,ijkabc->ijklabcd", t1_aa, t3_abaaba) * -2.0
        t4_abaaabaa += einsum("jb,iklacd->ijklabcd", t1_bb, t3_aaaaaa) * -6.0
        t4_abaaabaa += einsum("ijab,klcd->ijklabcd", t2_abab, t2_aaaa) * -2.0
        t4_abaaabaa += einsum("ijcb,klad->ijklabcd", t2_abab, t2_aaaa) * 2.0
        t4_abaaabaa += einsum("ijdb,klac->ijklabcd", t2_abab, t2_aaaa) * -2.0
        t4_abaaabaa += einsum("kjab,ilcd->ijklabcd", t2_abab, t2_aaaa) * 2.0
        t4_abaaabaa += einsum("kjcb,ilad->ijklabcd", t2_abab, t2_aaaa) * -2.0
        t4_abaaabaa += einsum("kjdb,ilac->ijklabcd", t2_abab, t2_aaaa) * 2.0
        t4_abaaabaa += einsum("ljab,ikcd->ijklabcd", t2_abab, t2_aaaa) * -2.0
        t4_abaaabaa += einsum("ljcb,ikad->ijklabcd", t2_abab, t2_aaaa) * 2.0
        t4_abaaabaa += einsum("ljdb,ikac->ijklabcd", t2_abab, t2_aaaa) * -2.0
        t4_abaaabaa += einsum("ia,kc,ljdb->ijklabcd", t1_aa, t1_aa, t2_abab) * -1.0
        t4_abaaabaa += einsum("ia,kd,ljcb->ijklabcd", t1_aa, t1_aa, t2_abab)
        t4_abaaabaa += einsum("ic,ka,ljdb->ijklabcd", t1_aa, t1_aa, t2_abab)
        t4_abaaabaa += einsum("ic,kd,ljab->ijklabcd", t1_aa, t1_aa, t2_abab) * -1.0
        t4_abaaabaa += einsum("id,ka,ljcb->ijklabcd", t1_aa, t1_aa, t2_abab) * -1.0
        t4_abaaabaa += einsum("id,kc,ljab->ijklabcd", t1_aa, t1_aa, t2_abab)
        t4_abaaabaa += einsum("ia,lc,kjdb->ijklabcd", t1_aa, t1_aa, t2_abab)
        t4_abaaabaa += einsum("ia,ld,kjcb->ijklabcd", t1_aa, t1_aa, t2_abab) * -1.0
        t4_abaaabaa += einsum("ic,la,kjdb->ijklabcd", t1_aa, t1_aa, t2_abab) * -1.0
        t4_abaaabaa += einsum("ic,ld,kjab->ijklabcd", t1_aa, t1_aa, t2_abab)
        t4_abaaabaa += einsum("id,la,kjcb->ijklabcd", t1_aa, t1_aa, t2_abab)
        t4_abaaabaa += einsum("id,lc,kjab->ijklabcd", t1_aa, t1_aa, t2_abab) * -1.0
        t4_abaaabaa += einsum("ka,lc,ijdb->ijklabcd", t1_aa, t1_aa, t2_abab) * -1.0
        t4_abaaabaa += einsum("ka,ld,ijcb->ijklabcd", t1_aa, t1_aa, t2_abab)
        t4_abaaabaa += einsum("kc,la,ijdb->ijklabcd", t1_aa, t1_aa, t2_abab)
        t4_abaaabaa += einsum("kc,ld,ijab->ijklabcd", t1_aa, t1_aa, t2_abab) * -1.0
        t4_abaaabaa += einsum("kd,la,ijcb->ijklabcd", t1_aa, t1_aa, t2_abab) * -1.0
        t4_abaaabaa += einsum("kd,lc,ijab->ijklabcd", t1_aa, t1_aa, t2_abab)
        t4_abaaabaa += einsum("ia,jb,klcd->ijklabcd", t1_aa, t1_bb, t2_aaaa) * -2.0
        t4_abaaabaa += einsum("ic,jb,klad->ijklabcd", t1_aa, t1_bb, t2_aaaa) * 2.0
        t4_abaaabaa += einsum("id,jb,klac->ijklabcd", t1_aa, t1_bb, t2_aaaa) * -2.0
        t4_abaaabaa += einsum("ka,jb,ilcd->ijklabcd", t1_aa, t1_bb, t2_aaaa) * 2.0
        t4_abaaabaa += einsum("kc,jb,ilad->ijklabcd", t1_aa, t1_bb, t2_aaaa) * -2.0
        t4_abaaabaa += einsum("kd,jb,ilac->ijklabcd", t1_aa, t1_bb, t2_aaaa) * 2.0
        t4_abaaabaa += einsum("la,jb,ikcd->ijklabcd", t1_aa, t1_bb, t2_aaaa) * -2.0
        t4_abaaabaa += einsum("lc,jb,ikad->ijklabcd", t1_aa, t1_bb, t2_aaaa) * 2.0
        t4_abaaabaa += einsum("ld,jb,ikac->ijklabcd", t1_aa, t1_bb, t2_aaaa) * -2.0
        t4_abaaabaa += einsum("ia,kc,ld,jb->ijklabcd", t1_aa, t1_aa, t1_aa, t1_bb) * -1.0
        t4_abaaabaa += einsum("ia,kd,lc,jb->ijklabcd", t1_aa, t1_aa, t1_aa, t1_bb)
        t4_abaaabaa += einsum("ic,ka,ld,jb->ijklabcd", t1_aa, t1_aa, t1_aa, t1_bb)
        t4_abaaabaa += einsum("ic,kd,la,jb->ijklabcd", t1_aa, t1_aa, t1_aa, t1_bb) * -1.0
        t4_abaaabaa += einsum("id,ka,lc,jb->ijklabcd", t1_aa, t1_aa, t1_aa, t1_bb) * -1.0
        t4_abaaabaa += einsum("id,kc,la,jb->ijklabcd", t1_aa, t1_aa, t1_aa, t1_bb)

        t4_abababab = np.zeros((nocc[0], nocc[1], nocc[0], nocc[1], nvir[0], nvir[1], nvir[0], nvir[1]), dtype=np.float64)
        t4_abababab += einsum("ijklabcd->ijklabcd", c4_abababab) * 4.0
        t4_abababab += einsum("ia,jklbcd->ijklabcd", t1_aa, t3_babbab) * -2.0
        t4_abababab += einsum("ic,jklbad->ijklabcd", t1_aa, t3_babbab) * 2.0
        t4_abababab += einsum("ka,ijlcbd->ijklabcd", t1_aa, t3_abbabb) * 2.0
        t4_abababab += einsum("kc,ijlabd->ijklabcd", t1_aa, t3_abbabb) * -2.0
        t4_abababab += einsum("jb,ilkadc->ijklabcd", t1_bb, t3_abaaba) * -2.0
        t4_abababab += einsum("jd,ilkabc->ijklabcd", t1_bb, t3_abaaba) * 2.0
        t4_abababab += einsum("lb,ijkadc->ijklabcd", t1_bb, t3_abaaba) * 2.0
        t4_abababab += einsum("ld,ijkabc->ijklabcd", t1_bb, t3_abaaba) * -2.0
        t4_abababab += einsum("ijab,klcd->ijklabcd", t2_abab, t2_abab) * -1.0
        t4_abababab += einsum("ijad,klcb->ijklabcd", t2_abab, t2_abab)
        t4_abababab += einsum("ijcb,klad->ijklabcd", t2_abab, t2_abab)
        t4_abababab += einsum("ijcd,klab->ijklabcd", t2_abab, t2_abab) * -1.0
        t4_abababab += einsum("ilab,kjcd->ijklabcd", t2_abab, t2_abab)
        t4_abababab += einsum("ilad,kjcb->ijklabcd", t2_abab, t2_abab) * -1.0
        t4_abababab += einsum("ilcb,kjad->ijklabcd", t2_abab, t2_abab) * -1.0
        t4_abababab += einsum("ilcd,kjab->ijklabcd", t2_abab, t2_abab)
        t4_abababab += einsum("ikac,jlbd->ijklabcd", t2_aaaa, t2_bbbb) * -4.0
        t4_abababab += einsum("ia,kc,jlbd->ijklabcd", t1_aa, t1_aa, t2_bbbb) * -2.0
        t4_abababab += einsum("ic,ka,jlbd->ijklabcd", t1_aa, t1_aa, t2_bbbb) * 2.0
        t4_abababab += einsum("ia,jb,klcd->ijklabcd", t1_aa, t1_bb, t2_abab) * -1.0
        t4_abababab += einsum("ia,jd,klcb->ijklabcd", t1_aa, t1_bb, t2_abab)
        t4_abababab += einsum("ic,jb,klad->ijklabcd", t1_aa, t1_bb, t2_abab)
        t4_abababab += einsum("ic,jd,klab->ijklabcd", t1_aa, t1_bb, t2_abab) * -1.0
        t4_abababab += einsum("ia,lb,kjcd->ijklabcd", t1_aa, t1_bb, t2_abab)
        t4_abababab += einsum("ia,ld,kjcb->ijklabcd", t1_aa, t1_bb, t2_abab) * -1.0
        t4_abababab += einsum("ic,lb,kjad->ijklabcd", t1_aa, t1_bb, t2_abab) * -1.0
        t4_abababab += einsum("ic,ld,kjab->ijklabcd", t1_aa, t1_bb, t2_abab)
        t4_abababab += einsum("ka,jb,ilcd->ijklabcd", t1_aa, t1_bb, t2_abab)
        t4_abababab += einsum("ka,jd,ilcb->ijklabcd", t1_aa, t1_bb, t2_abab) * -1.0
        t4_abababab += einsum("kc,jb,ilad->ijklabcd", t1_aa, t1_bb, t2_abab) * -1.0
        t4_abababab += einsum("kc,jd,ilab->ijklabcd", t1_aa, t1_bb, t2_abab)
        t4_abababab += einsum("ka,lb,ijcd->ijklabcd", t1_aa, t1_bb, t2_abab) * -1.0
        t4_abababab += einsum("ka,ld,ijcb->ijklabcd", t1_aa, t1_bb, t2_abab)
        t4_abababab += einsum("kc,lb,ijad->ijklabcd", t1_aa, t1_bb, t2_abab)
        t4_abababab += einsum("kc,ld,ijab->ijklabcd", t1_aa, t1_bb, t2_abab) * -1.0
        t4_abababab += einsum("jb,ld,ikac->ijklabcd", t1_bb, t1_bb, t2_aaaa) * -2.0
        t4_abababab += einsum("jd,lb,ikac->ijklabcd", t1_bb, t1_bb, t2_aaaa) * 2.0
        t4_abababab += einsum("ia,kc,jb,ld->ijklabcd", t1_aa, t1_aa, t1_bb, t1_bb) * -1.0
        t4_abababab += einsum("ia,kc,jd,lb->ijklabcd", t1_aa, t1_aa, t1_bb, t1_bb)
        t4_abababab += einsum("ic,ka,jb,ld->ijklabcd", t1_aa, t1_aa, t1_bb, t1_bb)
        t4_abababab += einsum("ic,ka,jd,lb->ijklabcd", t1_aa, t1_aa, t1_bb, t1_bb) * -1.0

        t4_bbabbbab = np.zeros((nocc[1], nocc[1], nocc[0], nocc[1], nvir[1], nvir[1], nvir[0], nvir[1]), dtype=np.float64)
        t4_bbabbbab += einsum("ijklabcd->ijklabcd", c4_bbabbbab) * 6.0
        t4_bbabbbab += einsum("kc,ijlabd->ijklabcd", t1_aa, t3_bbbbbb) * -6.0
        t4_bbabbbab += einsum("ia,jklbcd->ijklabcd", t1_bb, t3_babbab) * -2.0
        t4_bbabbbab += einsum("ib,jklacd->ijklabcd", t1_bb, t3_babbab) * 2.0
        t4_bbabbbab += einsum("id,jklacb->ijklabcd", t1_bb, t3_babbab) * -2.0
        t4_bbabbbab += einsum("ja,iklbcd->ijklabcd", t1_bb, t3_babbab) * 2.0
        t4_bbabbbab += einsum("jb,iklacd->ijklabcd", t1_bb, t3_babbab) * -2.0
        t4_bbabbbab += einsum("jd,iklacb->ijklabcd", t1_bb, t3_babbab) * 2.0
        t4_bbabbbab += einsum("la,ijkbdc->ijklabcd", t1_bb, t3_bbabba) * -2.0
        t4_bbabbbab += einsum("lb,ijkadc->ijklabcd", t1_bb, t3_bbabba) * 2.0
        t4_bbabbbab += einsum("ld,ijkabc->ijklabcd", t1_bb, t3_bbabba) * -2.0
        t4_bbabbbab += einsum("kjcb,ilad->ijklabcd", t2_abab, t2_bbbb) * -2.0
        t4_bbabbbab += einsum("kjcd,ilab->ijklabcd", t2_abab, t2_bbbb) * 2.0
        t4_bbabbbab += einsum("kjca,ilbd->ijklabcd", t2_abab, t2_bbbb) * 2.0
        t4_bbabbbab += einsum("klcb,ijad->ijklabcd", t2_abab, t2_bbbb) * 2.0
        t4_bbabbbab += einsum("klcd,ijab->ijklabcd", t2_abab, t2_bbbb) * -2.0
        t4_bbabbbab += einsum("klca,ijbd->ijklabcd", t2_abab, t2_bbbb) * -2.0
        t4_bbabbbab += einsum("kicb,jlad->ijklabcd", t2_abab, t2_bbbb) * 2.0
        t4_bbabbbab += einsum("kicd,jlab->ijklabcd", t2_abab, t2_bbbb) * -2.0
        t4_bbabbbab += einsum("kica,jlbd->ijklabcd", t2_abab, t2_bbbb) * -2.0
        t4_bbabbbab += einsum("kc,ia,jlbd->ijklabcd", t1_aa, t1_bb, t2_bbbb) * -2.0
        t4_bbabbbab += einsum("kc,ib,jlad->ijklabcd", t1_aa, t1_bb, t2_bbbb) * 2.0
        t4_bbabbbab += einsum("kc,id,jlab->ijklabcd", t1_aa, t1_bb, t2_bbbb) * -2.0
        t4_bbabbbab += einsum("kc,ja,ilbd->ijklabcd", t1_aa, t1_bb, t2_bbbb) * 2.0
        t4_bbabbbab += einsum("kc,jb,ilad->ijklabcd", t1_aa, t1_bb, t2_bbbb) * -2.0
        t4_bbabbbab += einsum("kc,jd,ilab->ijklabcd", t1_aa, t1_bb, t2_bbbb) * 2.0
        t4_bbabbbab += einsum("kc,la,ijbd->ijklabcd", t1_aa, t1_bb, t2_bbbb) * -2.0
        t4_bbabbbab += einsum("kc,lb,ijad->ijklabcd", t1_aa, t1_bb, t2_bbbb) * 2.0
        t4_bbabbbab += einsum("kc,ld,ijab->ijklabcd", t1_aa, t1_bb, t2_bbbb) * -2.0
        t4_bbabbbab += einsum("ia,jb,klcd->ijklabcd", t1_bb, t1_bb, t2_abab) * -1.0
        t4_bbabbbab += einsum("ia,jd,klcb->ijklabcd", t1_bb, t1_bb, t2_abab)
        t4_bbabbbab += einsum("ib,ja,klcd->ijklabcd", t1_bb, t1_bb, t2_abab)
        t4_bbabbbab += einsum("ib,jd,klca->ijklabcd", t1_bb, t1_bb, t2_abab) * -1.0
        t4_bbabbbab += einsum("id,ja,klcb->ijklabcd", t1_bb, t1_bb, t2_abab) * -1.0
        t4_bbabbbab += einsum("id,jb,klca->ijklabcd", t1_bb, t1_bb, t2_abab)
        t4_bbabbbab += einsum("ia,lb,kjcd->ijklabcd", t1_bb, t1_bb, t2_abab)
        t4_bbabbbab += einsum("ia,ld,kjcb->ijklabcd", t1_bb, t1_bb, t2_abab) * -1.0
        t4_bbabbbab += einsum("ib,la,kjcd->ijklabcd", t1_bb, t1_bb, t2_abab) * -1.0
        t4_bbabbbab += einsum("ib,ld,kjca->ijklabcd", t1_bb, t1_bb, t2_abab)
        t4_bbabbbab += einsum("id,la,kjcb->ijklabcd", t1_bb, t1_bb, t2_abab)
        t4_bbabbbab += einsum("id,lb,kjca->ijklabcd", t1_bb, t1_bb, t2_abab) * -1.0
        t4_bbabbbab += einsum("ja,lb,kicd->ijklabcd", t1_bb, t1_bb, t2_abab) * -1.0
        t4_bbabbbab += einsum("ja,ld,kicb->ijklabcd", t1_bb, t1_bb, t2_abab)
        t4_bbabbbab += einsum("jb,la,kicd->ijklabcd", t1_bb, t1_bb, t2_abab)
        t4_bbabbbab += einsum("jb,ld,kica->ijklabcd", t1_bb, t1_bb, t2_abab) * -1.0
        t4_bbabbbab += einsum("jd,la,kicb->ijklabcd", t1_bb, t1_bb, t2_abab) * -1.0
        t4_bbabbbab += einsum("jd,lb,kica->ijklabcd", t1_bb, t1_bb, t2_abab)
        t4_bbabbbab += einsum("kc,ia,jb,ld->ijklabcd", t1_aa, t1_bb, t1_bb, t1_bb) * -1.0
        t4_bbabbbab += einsum("kc,ia,jd,lb->ijklabcd", t1_aa, t1_bb, t1_bb, t1_bb)
        t4_bbabbbab += einsum("kc,ib,ja,ld->ijklabcd", t1_aa, t1_bb, t1_bb, t1_bb)
        t4_bbabbbab += einsum("kc,ib,jd,la->ijklabcd", t1_aa, t1_bb, t1_bb, t1_bb) * -1.0
        t4_bbabbbab += einsum("kc,id,ja,lb->ijklabcd", t1_aa, t1_bb, t1_bb, t1_bb) * -1.0
        t4_bbabbbab += einsum("kc,id,jb,la->ijklabcd", t1_aa, t1_bb, t1_bb, t1_bb)

        t4_bbbabbba = np.zeros((nocc[1], nocc[1], nocc[1], nocc[0], nvir[1], nvir[1], nvir[1], nvir[0]), dtype=np.float64)
        t4_bbbabbba += einsum("ijklabcd->ijklabcd", c4_bbbabbba) * 6.0
        t4_bbbabbba += einsum("ld,ijkabc->ijklabcd", t1_aa, t3_bbbbbb) * -6.0
        t4_bbbabbba += einsum("ia,jklbcd->ijklabcd", t1_bb, t3_bbabba) * -2.0
        t4_bbbabbba += einsum("ib,jklacd->ijklabcd", t1_bb, t3_bbabba) * 2.0
        t4_bbbabbba += einsum("ic,jklabd->ijklabcd", t1_bb, t3_bbabba) * -2.0
        t4_bbbabbba += einsum("ja,iklbcd->ijklabcd", t1_bb, t3_bbabba) * 2.0
        t4_bbbabbba += einsum("jb,iklacd->ijklabcd", t1_bb, t3_bbabba) * -2.0
        t4_bbbabbba += einsum("jc,iklabd->ijklabcd", t1_bb, t3_bbabba) * 2.0
        t4_bbbabbba += einsum("ka,ijlbcd->ijklabcd", t1_bb, t3_bbabba) * -2.0
        t4_bbbabbba += einsum("kb,ijlacd->ijklabcd", t1_bb, t3_bbabba) * 2.0
        t4_bbbabbba += einsum("kc,ijlabd->ijklabcd", t1_bb, t3_bbabba) * -2.0
        t4_bbbabbba += einsum("ljdb,ikac->ijklabcd", t2_abab, t2_bbbb) * -2.0
        t4_bbbabbba += einsum("ljdc,ikab->ijklabcd", t2_abab, t2_bbbb) * 2.0
        t4_bbbabbba += einsum("ljda,ikbc->ijklabcd", t2_abab, t2_bbbb) * 2.0
        t4_bbbabbba += einsum("lkdb,ijac->ijklabcd", t2_abab, t2_bbbb) * 2.0
        t4_bbbabbba += einsum("lkdc,ijab->ijklabcd", t2_abab, t2_bbbb) * -2.0
        t4_bbbabbba += einsum("lkda,ijbc->ijklabcd", t2_abab, t2_bbbb) * -2.0
        t4_bbbabbba += einsum("lidb,jkac->ijklabcd", t2_abab, t2_bbbb) * 2.0
        t4_bbbabbba += einsum("lidc,jkab->ijklabcd", t2_abab, t2_bbbb) * -2.0
        t4_bbbabbba += einsum("lida,jkbc->ijklabcd", t2_abab, t2_bbbb) * -2.0
        t4_bbbabbba += einsum("ld,ia,jkbc->ijklabcd", t1_aa, t1_bb, t2_bbbb) * -2.0
        t4_bbbabbba += einsum("ld,ib,jkac->ijklabcd", t1_aa, t1_bb, t2_bbbb) * 2.0
        t4_bbbabbba += einsum("ld,ic,jkab->ijklabcd", t1_aa, t1_bb, t2_bbbb) * -2.0
        t4_bbbabbba += einsum("ld,ja,ikbc->ijklabcd", t1_aa, t1_bb, t2_bbbb) * 2.0
        t4_bbbabbba += einsum("ld,jb,ikac->ijklabcd", t1_aa, t1_bb, t2_bbbb) * -2.0
        t4_bbbabbba += einsum("ld,jc,ikab->ijklabcd", t1_aa, t1_bb, t2_bbbb) * 2.0
        t4_bbbabbba += einsum("ld,ka,ijbc->ijklabcd", t1_aa, t1_bb, t2_bbbb) * -2.0
        t4_bbbabbba += einsum("ld,kb,ijac->ijklabcd", t1_aa, t1_bb, t2_bbbb) * 2.0
        t4_bbbabbba += einsum("ld,kc,ijab->ijklabcd", t1_aa, t1_bb, t2_bbbb) * -2.0
        t4_bbbabbba += einsum("ia,jb,lkdc->ijklabcd", t1_bb, t1_bb, t2_abab) * -1.0
        t4_bbbabbba += einsum("ia,jc,lkdb->ijklabcd", t1_bb, t1_bb, t2_abab)
        t4_bbbabbba += einsum("ib,ja,lkdc->ijklabcd", t1_bb, t1_bb, t2_abab)
        t4_bbbabbba += einsum("ib,jc,lkda->ijklabcd", t1_bb, t1_bb, t2_abab) * -1.0
        t4_bbbabbba += einsum("ic,ja,lkdb->ijklabcd", t1_bb, t1_bb, t2_abab) * -1.0
        t4_bbbabbba += einsum("ic,jb,lkda->ijklabcd", t1_bb, t1_bb, t2_abab)
        t4_bbbabbba += einsum("ia,kb,ljdc->ijklabcd", t1_bb, t1_bb, t2_abab)
        t4_bbbabbba += einsum("ia,kc,ljdb->ijklabcd", t1_bb, t1_bb, t2_abab) * -1.0
        t4_bbbabbba += einsum("ib,ka,ljdc->ijklabcd", t1_bb, t1_bb, t2_abab) * -1.0
        t4_bbbabbba += einsum("ib,kc,ljda->ijklabcd", t1_bb, t1_bb, t2_abab)
        t4_bbbabbba += einsum("ic,ka,ljdb->ijklabcd", t1_bb, t1_bb, t2_abab)
        t4_bbbabbba += einsum("ic,kb,ljda->ijklabcd", t1_bb, t1_bb, t2_abab) * -1.0
        t4_bbbabbba += einsum("ja,kb,lidc->ijklabcd", t1_bb, t1_bb, t2_abab) * -1.0
        t4_bbbabbba += einsum("ja,kc,lidb->ijklabcd", t1_bb, t1_bb, t2_abab)
        t4_bbbabbba += einsum("jb,ka,lidc->ijklabcd", t1_bb, t1_bb, t2_abab)
        t4_bbbabbba += einsum("jb,kc,lida->ijklabcd", t1_bb, t1_bb, t2_abab) * -1.0
        t4_bbbabbba += einsum("jc,ka,lidb->ijklabcd", t1_bb, t1_bb, t2_abab) * -1.0
        t4_bbbabbba += einsum("jc,kb,lida->ijklabcd", t1_bb, t1_bb, t2_abab)
        t4_bbbabbba += einsum("ld,ia,jb,kc->ijklabcd", t1_aa, t1_bb, t1_bb, t1_bb) * -1.0
        t4_bbbabbba += einsum("ld,ia,jc,kb->ijklabcd", t1_aa, t1_bb, t1_bb, t1_bb)
        t4_bbbabbba += einsum("ld,ib,ja,kc->ijklabcd", t1_aa, t1_bb, t1_bb, t1_bb)
        t4_bbbabbba += einsum("ld,ib,jc,ka->ijklabcd", t1_aa, t1_bb, t1_bb, t1_bb) * -1.0
        t4_bbbabbba += einsum("ld,ic,ja,kb->ijklabcd", t1_aa, t1_bb, t1_bb, t1_bb) * -1.0
        t4_bbbabbba += einsum("ld,ic,jb,ka->ijklabcd", t1_aa, t1_bb, t1_bb, t1_bb)

        t4_bbbbbbbb = np.zeros((nocc[1], nocc[1], nocc[1], nocc[1], nvir[1], nvir[1], nvir[1], nvir[1]), dtype=np.float64)
        t4_bbbbbbbb += einsum("ijklabcd->ijklabcd", c4_bbbbbbbb) * 24.0
        t4_bbbbbbbb += einsum("ia,jklbcd->ijklabcd", t1_bb, t3_bbbbbb) * -6.0
        t4_bbbbbbbb += einsum("ib,jklacd->ijklabcd", t1_bb, t3_bbbbbb) * 6.0
        t4_bbbbbbbb += einsum("ic,jklabd->ijklabcd", t1_bb, t3_bbbbbb) * -6.0
        t4_bbbbbbbb += einsum("id,jklabc->ijklabcd", t1_bb, t3_bbbbbb) * 6.0
        t4_bbbbbbbb += einsum("ja,iklbcd->ijklabcd", t1_bb, t3_bbbbbb) * 6.0
        t4_bbbbbbbb += einsum("jb,iklacd->ijklabcd", t1_bb, t3_bbbbbb) * -6.0
        t4_bbbbbbbb += einsum("jc,iklabd->ijklabcd", t1_bb, t3_bbbbbb) * 6.0
        t4_bbbbbbbb += einsum("jd,iklabc->ijklabcd", t1_bb, t3_bbbbbb) * -6.0
        t4_bbbbbbbb += einsum("ka,ijlbcd->ijklabcd", t1_bb, t3_bbbbbb) * -6.0
        t4_bbbbbbbb += einsum("kb,ijlacd->ijklabcd", t1_bb, t3_bbbbbb) * 6.0
        t4_bbbbbbbb += einsum("kc,ijlabd->ijklabcd", t1_bb, t3_bbbbbb) * -6.0
        t4_bbbbbbbb += einsum("kd,ijlabc->ijklabcd", t1_bb, t3_bbbbbb) * 6.0
        t4_bbbbbbbb += einsum("la,ijkbcd->ijklabcd", t1_bb, t3_bbbbbb) * 6.0
        t4_bbbbbbbb += einsum("lb,ijkacd->ijklabcd", t1_bb, t3_bbbbbb) * -6.0
        t4_bbbbbbbb += einsum("lc,ijkabd->ijklabcd", t1_bb, t3_bbbbbb) * 6.0
        t4_bbbbbbbb += einsum("ld,ijkabc->ijklabcd", t1_bb, t3_bbbbbb) * -6.0
        t4_bbbbbbbb += einsum("ikac,jlbd->ijklabcd", t2_bbbb, t2_bbbb) * -4.0
        t4_bbbbbbbb += einsum("ikad,jlbc->ijklabcd", t2_bbbb, t2_bbbb) * 4.0
        t4_bbbbbbbb += einsum("ikab,jlcd->ijklabcd", t2_bbbb, t2_bbbb) * 4.0
        t4_bbbbbbbb += einsum("ikbc,jlad->ijklabcd", t2_bbbb, t2_bbbb) * 4.0
        t4_bbbbbbbb += einsum("ikbd,jlac->ijklabcd", t2_bbbb, t2_bbbb) * -4.0
        t4_bbbbbbbb += einsum("ikcd,jlab->ijklabcd", t2_bbbb, t2_bbbb) * 4.0
        t4_bbbbbbbb += einsum("ilac,jkbd->ijklabcd", t2_bbbb, t2_bbbb) * 4.0
        t4_bbbbbbbb += einsum("ilad,jkbc->ijklabcd", t2_bbbb, t2_bbbb) * -4.0
        t4_bbbbbbbb += einsum("ilab,jkcd->ijklabcd", t2_bbbb, t2_bbbb) * -4.0
        t4_bbbbbbbb += einsum("ilbc,jkad->ijklabcd", t2_bbbb, t2_bbbb) * -4.0
        t4_bbbbbbbb += einsum("ilbd,jkac->ijklabcd", t2_bbbb, t2_bbbb) * 4.0
        t4_bbbbbbbb += einsum("ilcd,jkab->ijklabcd", t2_bbbb, t2_bbbb) * -4.0
        t4_bbbbbbbb += einsum("ijac,klbd->ijklabcd", t2_bbbb, t2_bbbb) * 4.0
        t4_bbbbbbbb += einsum("ijad,klbc->ijklabcd", t2_bbbb, t2_bbbb) * -4.0
        t4_bbbbbbbb += einsum("ijab,klcd->ijklabcd", t2_bbbb, t2_bbbb) * -4.0
        t4_bbbbbbbb += einsum("ijbc,klad->ijklabcd", t2_bbbb, t2_bbbb) * -4.0
        t4_bbbbbbbb += einsum("ijbd,klac->ijklabcd", t2_bbbb, t2_bbbb) * 4.0
        t4_bbbbbbbb += einsum("ijcd,klab->ijklabcd", t2_bbbb, t2_bbbb) * -4.0
        t4_bbbbbbbb += einsum("ia,jb,klcd->ijklabcd", t1_bb, t1_bb, t2_bbbb) * -2.0
        t4_bbbbbbbb += einsum("ia,jc,klbd->ijklabcd", t1_bb, t1_bb, t2_bbbb) * 2.0
        t4_bbbbbbbb += einsum("ia,jd,klbc->ijklabcd", t1_bb, t1_bb, t2_bbbb) * -2.0
        t4_bbbbbbbb += einsum("ib,ja,klcd->ijklabcd", t1_bb, t1_bb, t2_bbbb) * 2.0
        t4_bbbbbbbb += einsum("ib,jc,klad->ijklabcd", t1_bb, t1_bb, t2_bbbb) * -2.0
        t4_bbbbbbbb += einsum("ib,jd,klac->ijklabcd", t1_bb, t1_bb, t2_bbbb) * 2.0
        t4_bbbbbbbb += einsum("ic,ja,klbd->ijklabcd", t1_bb, t1_bb, t2_bbbb) * -2.0
        t4_bbbbbbbb += einsum("ic,jb,klad->ijklabcd", t1_bb, t1_bb, t2_bbbb) * 2.0
        t4_bbbbbbbb += einsum("ic,jd,klab->ijklabcd", t1_bb, t1_bb, t2_bbbb) * -2.0
        t4_bbbbbbbb += einsum("id,ja,klbc->ijklabcd", t1_bb, t1_bb, t2_bbbb) * 2.0
        t4_bbbbbbbb += einsum("id,jb,klac->ijklabcd", t1_bb, t1_bb, t2_bbbb) * -2.0
        t4_bbbbbbbb += einsum("id,jc,klab->ijklabcd", t1_bb, t1_bb, t2_bbbb) * 2.0
        t4_bbbbbbbb += einsum("ia,kb,jlcd->ijklabcd", t1_bb, t1_bb, t2_bbbb) * 2.0
        t4_bbbbbbbb += einsum("ia,kc,jlbd->ijklabcd", t1_bb, t1_bb, t2_bbbb) * -2.0
        t4_bbbbbbbb += einsum("ia,kd,jlbc->ijklabcd", t1_bb, t1_bb, t2_bbbb) * 2.0
        t4_bbbbbbbb += einsum("ib,ka,jlcd->ijklabcd", t1_bb, t1_bb, t2_bbbb) * -2.0
        t4_bbbbbbbb += einsum("ib,kc,jlad->ijklabcd", t1_bb, t1_bb, t2_bbbb) * 2.0
        t4_bbbbbbbb += einsum("ib,kd,jlac->ijklabcd", t1_bb, t1_bb, t2_bbbb) * -2.0
        t4_bbbbbbbb += einsum("ic,ka,jlbd->ijklabcd", t1_bb, t1_bb, t2_bbbb) * 2.0
        t4_bbbbbbbb += einsum("ic,kb,jlad->ijklabcd", t1_bb, t1_bb, t2_bbbb) * -2.0
        t4_bbbbbbbb += einsum("ic,kd,jlab->ijklabcd", t1_bb, t1_bb, t2_bbbb) * 2.0
        t4_bbbbbbbb += einsum("id,ka,jlbc->ijklabcd", t1_bb, t1_bb, t2_bbbb) * -2.0
        t4_bbbbbbbb += einsum("id,kb,jlac->ijklabcd", t1_bb, t1_bb, t2_bbbb) * 2.0
        t4_bbbbbbbb += einsum("id,kc,jlab->ijklabcd", t1_bb, t1_bb, t2_bbbb) * -2.0
        t4_bbbbbbbb += einsum("ia,lb,jkcd->ijklabcd", t1_bb, t1_bb, t2_bbbb) * -2.0
        t4_bbbbbbbb += einsum("ia,lc,jkbd->ijklabcd", t1_bb, t1_bb, t2_bbbb) * 2.0
        t4_bbbbbbbb += einsum("ia,ld,jkbc->ijklabcd", t1_bb, t1_bb, t2_bbbb) * -2.0
        t4_bbbbbbbb += einsum("ib,la,jkcd->ijklabcd", t1_bb, t1_bb, t2_bbbb) * 2.0
        t4_bbbbbbbb += einsum("ib,lc,jkad->ijklabcd", t1_bb, t1_bb, t2_bbbb) * -2.0
        t4_bbbbbbbb += einsum("ib,ld,jkac->ijklabcd", t1_bb, t1_bb, t2_bbbb) * 2.0
        t4_bbbbbbbb += einsum("ic,la,jkbd->ijklabcd", t1_bb, t1_bb, t2_bbbb) * -2.0
        t4_bbbbbbbb += einsum("ic,lb,jkad->ijklabcd", t1_bb, t1_bb, t2_bbbb) * 2.0
        t4_bbbbbbbb += einsum("ic,ld,jkab->ijklabcd", t1_bb, t1_bb, t2_bbbb) * -2.0
        t4_bbbbbbbb += einsum("id,la,jkbc->ijklabcd", t1_bb, t1_bb, t2_bbbb) * 2.0
        t4_bbbbbbbb += einsum("id,lb,jkac->ijklabcd", t1_bb, t1_bb, t2_bbbb) * -2.0
        t4_bbbbbbbb += einsum("id,lc,jkab->ijklabcd", t1_bb, t1_bb, t2_bbbb) * 2.0
        t4_bbbbbbbb += einsum("ja,kb,ilcd->ijklabcd", t1_bb, t1_bb, t2_bbbb) * -2.0
        t4_bbbbbbbb += einsum("ja,kc,ilbd->ijklabcd", t1_bb, t1_bb, t2_bbbb) * 2.0
        t4_bbbbbbbb += einsum("ja,kd,ilbc->ijklabcd", t1_bb, t1_bb, t2_bbbb) * -2.0
        t4_bbbbbbbb += einsum("jb,ka,ilcd->ijklabcd", t1_bb, t1_bb, t2_bbbb) * 2.0
        t4_bbbbbbbb += einsum("jb,kc,ilad->ijklabcd", t1_bb, t1_bb, t2_bbbb) * -2.0
        t4_bbbbbbbb += einsum("jb,kd,ilac->ijklabcd", t1_bb, t1_bb, t2_bbbb) * 2.0
        t4_bbbbbbbb += einsum("jc,ka,ilbd->ijklabcd", t1_bb, t1_bb, t2_bbbb) * -2.0
        t4_bbbbbbbb += einsum("jc,kb,ilad->ijklabcd", t1_bb, t1_bb, t2_bbbb) * 2.0
        t4_bbbbbbbb += einsum("jc,kd,ilab->ijklabcd", t1_bb, t1_bb, t2_bbbb) * -2.0
        t4_bbbbbbbb += einsum("jd,ka,ilbc->ijklabcd", t1_bb, t1_bb, t2_bbbb) * 2.0
        t4_bbbbbbbb += einsum("jd,kb,ilac->ijklabcd", t1_bb, t1_bb, t2_bbbb) * -2.0
        t4_bbbbbbbb += einsum("jd,kc,ilab->ijklabcd", t1_bb, t1_bb, t2_bbbb) * 2.0
        t4_bbbbbbbb += einsum("ja,lb,ikcd->ijklabcd", t1_bb, t1_bb, t2_bbbb) * 2.0
        t4_bbbbbbbb += einsum("ja,lc,ikbd->ijklabcd", t1_bb, t1_bb, t2_bbbb) * -2.0
        t4_bbbbbbbb += einsum("ja,ld,ikbc->ijklabcd", t1_bb, t1_bb, t2_bbbb) * 2.0
        t4_bbbbbbbb += einsum("jb,la,ikcd->ijklabcd", t1_bb, t1_bb, t2_bbbb) * -2.0
        t4_bbbbbbbb += einsum("jb,lc,ikad->ijklabcd", t1_bb, t1_bb, t2_bbbb) * 2.0
        t4_bbbbbbbb += einsum("jb,ld,ikac->ijklabcd", t1_bb, t1_bb, t2_bbbb) * -2.0
        t4_bbbbbbbb += einsum("jc,la,ikbd->ijklabcd", t1_bb, t1_bb, t2_bbbb) * 2.0
        t4_bbbbbbbb += einsum("jc,lb,ikad->ijklabcd", t1_bb, t1_bb, t2_bbbb) * -2.0
        t4_bbbbbbbb += einsum("jc,ld,ikab->ijklabcd", t1_bb, t1_bb, t2_bbbb) * 2.0
        t4_bbbbbbbb += einsum("jd,la,ikbc->ijklabcd", t1_bb, t1_bb, t2_bbbb) * -2.0
        t4_bbbbbbbb += einsum("jd,lb,ikac->ijklabcd", t1_bb, t1_bb, t2_bbbb) * 2.0
        t4_bbbbbbbb += einsum("jd,lc,ikab->ijklabcd", t1_bb, t1_bb, t2_bbbb) * -2.0
        t4_bbbbbbbb += einsum("ka,lb,ijcd->ijklabcd", t1_bb, t1_bb, t2_bbbb) * -2.0
        t4_bbbbbbbb += einsum("ka,lc,ijbd->ijklabcd", t1_bb, t1_bb, t2_bbbb) * 2.0
        t4_bbbbbbbb += einsum("ka,ld,ijbc->ijklabcd", t1_bb, t1_bb, t2_bbbb) * -2.0
        t4_bbbbbbbb += einsum("kb,la,ijcd->ijklabcd", t1_bb, t1_bb, t2_bbbb) * 2.0
        t4_bbbbbbbb += einsum("kb,lc,ijad->ijklabcd", t1_bb, t1_bb, t2_bbbb) * -2.0
        t4_bbbbbbbb += einsum("kb,ld,ijac->ijklabcd", t1_bb, t1_bb, t2_bbbb) * 2.0
        t4_bbbbbbbb += einsum("kc,la,ijbd->ijklabcd", t1_bb, t1_bb, t2_bbbb) * -2.0
        t4_bbbbbbbb += einsum("kc,lb,ijad->ijklabcd", t1_bb, t1_bb, t2_bbbb) * 2.0
        t4_bbbbbbbb += einsum("kc,ld,ijab->ijklabcd", t1_bb, t1_bb, t2_bbbb) * -2.0
        t4_bbbbbbbb += einsum("kd,la,ijbc->ijklabcd", t1_bb, t1_bb, t2_bbbb) * 2.0
        t4_bbbbbbbb += einsum("kd,lb,ijac->ijklabcd", t1_bb, t1_bb, t2_bbbb) * -2.0
        t4_bbbbbbbb += einsum("kd,lc,ijab->ijklabcd", t1_bb, t1_bb, t2_bbbb) * 2.0
        t4_bbbbbbbb += einsum("ia,jb,kc,ld->ijklabcd", t1_bb, t1_bb, t1_bb, t1_bb) * -1.0
        t4_bbbbbbbb += einsum("ia,jb,kd,lc->ijklabcd", t1_bb, t1_bb, t1_bb, t1_bb)
        t4_bbbbbbbb += einsum("ia,jc,kb,ld->ijklabcd", t1_bb, t1_bb, t1_bb, t1_bb)
        t4_bbbbbbbb += einsum("ia,jc,kd,lb->ijklabcd", t1_bb, t1_bb, t1_bb, t1_bb) * -1.0
        t4_bbbbbbbb += einsum("ia,jd,kb,lc->ijklabcd", t1_bb, t1_bb, t1_bb, t1_bb) * -1.0
        t4_bbbbbbbb += einsum("ia,jd,kc,lb->ijklabcd", t1_bb, t1_bb, t1_bb, t1_bb)
        t4_bbbbbbbb += einsum("ib,ja,kc,ld->ijklabcd", t1_bb, t1_bb, t1_bb, t1_bb)
        t4_bbbbbbbb += einsum("ib,ja,kd,lc->ijklabcd", t1_bb, t1_bb, t1_bb, t1_bb) * -1.0
        t4_bbbbbbbb += einsum("ib,jc,ka,ld->ijklabcd", t1_bb, t1_bb, t1_bb, t1_bb) * -1.0
        t4_bbbbbbbb += einsum("ib,jc,kd,la->ijklabcd", t1_bb, t1_bb, t1_bb, t1_bb)
        t4_bbbbbbbb += einsum("ib,jd,ka,lc->ijklabcd", t1_bb, t1_bb, t1_bb, t1_bb)
        t4_bbbbbbbb += einsum("ib,jd,kc,la->ijklabcd", t1_bb, t1_bb, t1_bb, t1_bb) * -1.0
        t4_bbbbbbbb += einsum("ic,ja,kb,ld->ijklabcd", t1_bb, t1_bb, t1_bb, t1_bb) * -1.0
        t4_bbbbbbbb += einsum("ic,ja,kd,lb->ijklabcd", t1_bb, t1_bb, t1_bb, t1_bb)
        t4_bbbbbbbb += einsum("ic,jb,ka,ld->ijklabcd", t1_bb, t1_bb, t1_bb, t1_bb)
        t4_bbbbbbbb += einsum("ic,jb,kd,la->ijklabcd", t1_bb, t1_bb, t1_bb, t1_bb) * -1.0
        t4_bbbbbbbb += einsum("ic,jd,ka,lb->ijklabcd", t1_bb, t1_bb, t1_bb, t1_bb) * -1.0
        t4_bbbbbbbb += einsum("ic,jd,kb,la->ijklabcd", t1_bb, t1_bb, t1_bb, t1_bb)
        t4_bbbbbbbb += einsum("id,ja,kb,lc->ijklabcd", t1_bb, t1_bb, t1_bb, t1_bb)
        t4_bbbbbbbb += einsum("id,ja,kc,lb->ijklabcd", t1_bb, t1_bb, t1_bb, t1_bb) * -1.0
        t4_bbbbbbbb += einsum("id,jb,ka,lc->ijklabcd", t1_bb, t1_bb, t1_bb, t1_bb) * -1.0
        t4_bbbbbbbb += einsum("id,jb,kc,la->ijklabcd", t1_bb, t1_bb, t1_bb, t1_bb)
        t4_bbbbbbbb += einsum("id,jc,ka,lb->ijklabcd", t1_bb, t1_bb, t1_bb, t1_bb)
        t4_bbbbbbbb += einsum("id,jc,kb,la->ijklabcd", t1_bb, t1_bb, t1_bb, t1_bb) * -1.0

        t1 = (t1_aa, t1_bb)
        t2 = (t2_aaaa, t2_abab, t2_bbbb)
        t3 = (t3_aaaaaa, t3_abaaba, t3_abbabb, t3_babbab, t3_bbabba, t3_bbbbbb)
        t4 = (t4_aaaaaaaa, t4_aaabaaab, t4_aabaaaba, t4_abaaabaa, t4_abababab, t4_bbabbbab, t4_bbbabbba, t4_bbbbbbbb)

        return wf_types.UCCSDTQ_WaveFunction(self.mo, t1=t1, t2=t2, t3=t3, t4=t4)
