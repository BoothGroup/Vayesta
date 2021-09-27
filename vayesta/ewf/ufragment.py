import numpy as np

from vayesta.core.util import *
from vayesta.core.qemb import UFragment
from vayesta.core.bath import UDMET_Bath
from .fragment import EWFFragment


class UEWFFragment(UFragment, EWFFragment):

    def set_cas(self, *args, **kwargs):
        raise NotImplementedError()

    def truncate_bno(self, c_no, n_no, *args, **kwargs):
        results = []
        for s, spin in enumerate(('alpha', 'beta')):
            self.log.info("%s:", spin.capitalize())
            results.append(super().truncate_bno(c_no[s], n_no[s], *args, **kwargs))
        return tuple(zip(*results))

    def make_bath(self, bath_type=NotSet):
        if bath_type is NotSet:
            bath_type = self.opts.bath_type
        # DMET bath only
        if bath_type is None or bath_type.lower() == 'dmet':
            bath = UDMET_Bath(self, dmet_threshold=self.opts.dmet_threshold)
        # All environment orbitals as bath
        elif bath_type.lower() in ('all', 'complete'):
            raise NotImplementedError()
            #bath = CompleteBath(self, dmet_threshold=self.opts.dmet_threshold)
        # MP2 bath natural orbitals
        elif bath_type.lower() == 'mp2-bno':
            raise NotImplementedError()
            #bath = BNO_Bath(self, dmet_threshold=self.opts.dmet_threshold)
        bath.kernel()
        self.bath = bath
        return bath
