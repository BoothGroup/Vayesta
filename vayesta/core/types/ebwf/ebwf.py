from vayesta.core.types import wf
from vayesta.core.util import AbstractMethodError

class EBWavefunction(wf.WaveFunction):
    def __init__(self, mo, mbos=None, projector=None):
        super().__init__(mo, projector)
        self.mbos = mbos

    @property
    def nbos(self):
        return 0 if self.mbos is None else self.mbos.nbos

    def __repr__(self):
        return "%s(norb= %r, nocc= %r, nvir=%r, nbos= %r)" % (self.__class__.__name__, self.norb, self.nocc, self.nvir, self.nbos)

    def make_rdm_eb(self, *args, **kwargs):
        raise AbstractMethodError

    def make_rdm_bb(self, *args, **kwargs):
        raise AbstractMethodError
