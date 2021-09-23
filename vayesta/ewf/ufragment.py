
from vayesta.core.qemb import UFragment
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
