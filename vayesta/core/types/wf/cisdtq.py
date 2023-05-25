import numpy as np
import vayesta
from vayesta.core.types import wf as wf_types
from vayesta.core.types.wf import t_to_c


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
        c1 = self.c1 / self.c0
        c2 = self.c2 / self.c0
        c3 = self.c3 / self.c0
        c4 = tuple(c / self.c0 for c in self.c4)

        t1 = t_to_c.t1_rhf(c1)
        t2 = t_to_c.t2_rhf(t1, c2)
        t3 = t_to_c.t3_rhf(t1, t2, c3)
        t4 = t_to_c.t4_rhf(t1, t2, t3, c4)

        return wf_types.RCCSDTQ_WaveFunction(self.mo, t1=t1, t2=t2, t3=t3, t4=t4)


class UCISDTQ_WaveFunction(wf_types.WaveFunction):

    def __init__(self, mo, c0, c1, c2, c3, c4):
        super().__init__(mo)
        self.c0 = c0
        self.c1 = c1
        self.c2 = c2
        self.c3 = c3
        self.c4 = c4
        if not (isinstance(c3, tuple) and len(c3) == 4):
            raise ValueError("c4 definition in UCISDTQ wfn requires tuple of (aaa, aba, bab, bbb) spin signatures")
        if not (isinstance(c4, tuple) and len(c4) == 5):
            raise ValueError(
                    "c4 definition in UCISDTQ wfn requires tuple of (aaaa, aaab, abab, abbb, bbbb) spin signatures")

    def as_ccsdtq(self):
        c1 = tuple(c / self.c0 for c in self.c1)
        c2 = tuple(c / self.c0 for c in self.c2)
        c3 = tuple(c / self.c0 for c in self.c3)
        c4 = tuple(c / self.c0 for c in self.c4)

        t1 = t_to_c.t1_uhf(c1)
        t2 = t_to_c.t2_uhf(t1, c2)
        t3 = t_to_c.t3_uhf(t1, t2, c3)
        t4 = t_to_c.t4_uhf(t1, t2, t3, c4)

        return wf_types.UCCSDTQ_WaveFunction(self.mo, t1=t1, t2=t2, t3=t3, t4=t4)
