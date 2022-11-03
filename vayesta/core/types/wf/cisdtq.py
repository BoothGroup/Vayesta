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

    def __init__(self, mo, c0, c1, c2, c3, c4_abab, c4_abaa):
        super().__init__(mo)
        self.c0 = c0
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.c4_abab = c4_abab
        self.c4_abaa = c4_abaa

    def as_ccsdtq(self):
        t1 = self.c1/self.c0
        t2 = self.c2/self.c0 - einsum('ia,jb->ijab', t1, t1)
        raise NotImplementedError
        # TODO:
        # see also THE JOURNAL OF CHEMICAL PHYSICS 147, 154105 (2017)
        t3 = self.c3/self.c0 # - C1*C2 - (C1^3)/3
        t4 = self.c4/self.c0 # - C1*C3 - (C2^2)/2 - C1^2*C2 - (C1^4)/4
        return wf_types.RCCSDTQ_WaveFunction(self.mo, t1=t1, t2=t2, t3=t3, t4=t4)


class UCISDTQ_WaveFunction(RCISDTQ_WaveFunction):

    def as_ccsdtq(self):
        c1a, c1b = self.c1
        c2aa, c2ab, c2bb = self.c2
        # TODO
        #c3aaa, c3aab, ... = self.c3
        #c4aaaa, c4aaab, ... = self.c4

        # T1
        t1a = c1a/self.c0
        t1b = c1b/self.c0
        # T2
        t2aa = c2aa/self.c0 - einsum('ia,jb->ijab', t1a, t1a) + einsum('ib,ja->ijab', t1a, t1a)
        t2bb = c2bb/self.c0 - einsum('ia,jb->ijab', t1b, t1b) + einsum('ib,ja->ijab', t1b, t1b)
        t2ab = c2ab/self.c0 - einsum('ia,jb->ijab', t1a, t1b)
        # T3
        raise NotImplementedError
        #t3aaa = c3aaa/self.c0 - einsum('ijab,kc->ijkabc', t2a, t1a) - ...
        # T4
        #t4aaaa = c4aaaa/self.c0 - einsum('ijkabc,ld->ijklabcd', t3a, t1a) - ...

        t1 = (t1a, t1b)
        t2 = (t2aa, t2ab, t2bb)
        # TODO
        #t3 = (t3aaa, t3aab, ...)
        #t4 = (t4aaaa, t4aaab, ...)
        return wf_types.UCCSDTQ_WaveFunction(self.mo, t1=t1, t2=t2, t3=t3, t4=t4)
