from vayesta.core.types import wf as wf_types


def CCSDTQ_WaveFunction(mo, *args, **kwargs):
    if mo.nspin == 1:
        cls = RCCSDTQ_WaveFunction
    elif mo.nspin == 2:
        cls = UCCSDTQ_WaveFunction
    return cls(mo, *args, **kwargs)


class RCCSDTQ_WaveFunction(wf_types.WaveFunction):
    # TODO: Contract T4's down to intermediates to reduce EC-CC memory overheads.

    def __init__(self, mo, t1, t2, t3, t4):
        super().__init__(mo)
        self.t1 = t1
        self.t2 = t2
        self.t3 = t3
        self.t4 = t4
        self._check_amps()

    def _check_amps(self):
        if not (isinstance(self.t4, tuple) and len(self.t4) == 2):
            raise ValueError("t4 definition in RCCSDTQ wfn requires tuple of (abaa, abab) spin signatures")

    def as_ccsdtq(self):
        return self

    def as_ccsd(self):
        if self.projector is not None:
            raise NotImplementedError
        return wf_types.CCSD_WaveFunction(self.mo, self.t1, self.t2)

    def as_cisd(self, c0=1.0):
        return self.as_ccsd().as_cisd()


class UCCSDTQ_WaveFunction(RCCSDTQ_WaveFunction):
    def _check_amps(self):
        if not (isinstance(self.t3, tuple) and len(self.t3) == 4):
            raise ValueError("t4 definition in UCCSDTQ wfn requires tuple of (aaa, aba, bab, bbb) spin signatures")
        if not (isinstance(self.t4, tuple) and len(self.t4) == 5):
            raise ValueError(
                "t4 definition in UCCSDTQ wfn requires tuple of (aaaa, aaab, abab, abbb, bbbb) spin signatures"
            )
