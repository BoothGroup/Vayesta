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
        c4_aaaaaaaa, c4_aaabaaab, c4_aabaaaba, c4_abaaabaa, c4_abababab, c4_abbbabbb, c4_bbabbbab, c4_bbbabbba, c4_bbbbbbbb = (c / self.c0 for c in self.c4)

        from ._conversion_routines import \
                t1_uhf_aa, t1_uhf_bb, \
                t2_uhf_aaaa, t2_uhf_abab, t2_uhf_bbbb, \
                t3_uhf_aaaaaa, t3_uhf_abaaba, t3_uhf_abbabb, t3_uhf_babbab, t3_uhf_bbabba, t3_uhf_bbbbbb, \
                t4_uhf_aaaaaaaa, t4_uhf_aaabaaab, t4_uhf_aabaaaba, t4_uhf_abaaabaa, t4_uhf_abababab, t4_uhf_abbbabbb, t4_uhf_bbabbbab, t4_uhf_bbbabbba, t4_uhf_bbbbbbbb

        from types import SimpleNamespace
        c1 = SimpleNamespace(aa=c1_aa, bb=c1_bb)
        c2 = SimpleNamespace(aaaa=c2_aaaa, abab=c2_abab, bbbb=c2_bbbb)
        c3 = SimpleNamespace(aaaaaa=c3_aaaaaa, abaaba=c3_abaaba, abbabb=c3_abbabb, babbab=c3_babbab, bbabba=c3_bbabba, bbbbbb=c3_bbbbbb)
        c4 = SimpleNamespace(aaaaaaaa=c4_aaaaaaaa, aaabaaab=c4_aaabaaab, aabaaaba=c4_aabaaaba, abaaabaa=c4_abaaabaa, abababab=c4_abababab, abbbabbb=c4_abbbabbb, bbabbbab=c4_bbabbbab, bbbabbba=c4_bbbabbba, bbbbbbbb=c4_bbbbbbbb)

        nocc = (c1.aa.shape[0], c1.bb.shape[0])
        nvir = (c1.aa.shape[1], c1.bb.shape[1])

        t1_aa = t1_uhf_aa(c1=c1, nocc=nocc, nvir=nvir)
        t1_bb = t1_uhf_bb(c1=c1, nocc=nocc, nvir=nvir)
        t1 = SimpleNamespace(aa=t1_aa, bb=t1_bb)

        t2_aaaa = t2_uhf_aaaa(c2=c2, t1=t1, nocc=nocc, nvir=nvir)
        t2_abab = t2_uhf_abab(c2=c2, t1=t1, nocc=nocc, nvir=nvir)
        t2_bbbb = t2_uhf_bbbb(c2=c2, t1=t1, nocc=nocc, nvir=nvir)
        t2 = SimpleNamespace(aaaa=t2_aaaa, abab=t2_abab, bbbb=t2_bbbb)

        t3_aaaaaa = t3_uhf_aaaaaa(c3=c3, t1=t1, t2=t2, nocc=nocc, nvir=nvir)
        t3_abaaba = t3_uhf_abaaba(c3=c3, t1=t1, t2=t2, nocc=nocc, nvir=nvir)
        t3_abbabb = t3_uhf_abbabb(c3=c3, t1=t1, t2=t2, nocc=nocc, nvir=nvir)
        t3_babbab = t3_uhf_babbab(c3=c3, t1=t1, t2=t2, nocc=nocc, nvir=nvir)
        t3_bbabba = t3_uhf_bbabba(c3=c3, t1=t1, t2=t2, nocc=nocc, nvir=nvir)
        t3_bbbbbb = t3_uhf_bbbbbb(c3=c3, t1=t1, t2=t2, nocc=nocc, nvir=nvir)
        t3 = SimpleNamespace(aaaaaa=t3_aaaaaa, abaaba=t3_abaaba, abbabb=t3_abbabb, babbab=t3_babbab, bbabba=t3_bbabba, bbbbbb=t3_bbbbbb)

        t4_aaaaaaaa = t4_uhf_aaaaaaaa(c4=c4, t1=t1, t2=t2, t3=t3, nocc=nocc, nvir=nvir)
        t4_aaabaaab = t4_uhf_aaabaaab(c4=c4, t1=t1, t2=t2, t3=t3, nocc=nocc, nvir=nvir)
        t4_aabaaaba = t4_uhf_aabaaaba(c4=c4, t1=t1, t2=t2, t3=t3, nocc=nocc, nvir=nvir)
        t4_abaaabaa = t4_uhf_abaaabaa(c4=c4, t1=t1, t2=t2, t3=t3, nocc=nocc, nvir=nvir)
        t4_abababab = t4_uhf_abababab(c4=c4, t1=t1, t2=t2, t3=t3, nocc=nocc, nvir=nvir)
        t4_abbbabbb = t4_uhf_abbbabbb(c4=c4, t1=t1, t2=t2, t3=t3, nocc=nocc, nvir=nvir)
        t4_bbabbbab = t4_uhf_bbabbbab(c4=c4, t1=t1, t2=t2, t3=t3, nocc=nocc, nvir=nvir)
        t4_bbbabbba = t4_uhf_bbbabbba(c4=c4, t1=t1, t2=t2, t3=t3, nocc=nocc, nvir=nvir)
        t4_bbbbbbbb = t4_uhf_bbbbbbbb(c4=c4, t1=t1, t2=t2, t3=t3, nocc=nocc, nvir=nvir)

        t1 = (t1_aa, t1_bb)
        t2 = (t2_aaaa, t2_abab, t2_bbbb)
        t3 = (t3_aaaaaa, t3_abaaba, t3_abbabb, t3_babbab, t3_bbabba, t3_bbbbbb)
        t4 = (t4_aaaaaaaa, t4_aaabaaab, t4_aabaaaba, t4_abaaabaa, t4_abababab, t4_abbbabbb, t4_bbabbbab, t4_bbbabbba, t4_bbbbbbbb)

        return wf_types.UCCSDTQ_WaveFunction(self.mo, t1=t1, t2=t2, t3=t3, t4=t4)
