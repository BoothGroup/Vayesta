import numpy as np

from vayesta.core.types import wf
from vayesta.core.util import *

class EBWavefunction(wf.Wavefunction):
    def __init__(self, mo, nbos, projector=None):
        super().__init(mo, projector)
        self.nbos = nbos

    def __repr__(self):
        return "%s(norb= %r, nocc= %r, nvir=%r, nbos= %r)" % (self.__class__.__name__, self.norb, self.nocc, self.nvir, self.nbos)

    def make_rdm_eb(self, *args, **kwargs):
        raise AbstractMethodError

    def make_rdm_bb(self, *args, **kwargs):
        raise AbstractMethodError

