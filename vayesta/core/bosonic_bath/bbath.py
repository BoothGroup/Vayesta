from vayesta.core.util import einsum, AbstractMethodError
from vayesta.core.bath import BNO_Threshold, helper
from vayesta.core.bath.bath import Bath
import numpy as np
import numbers


class Boson_Threshold(BNO_Threshold):
    def __init__(self, type, threshold):
        if type in ('electron-percent', 'excited-percent'):
            raise ValueError("Electron-percent and excited-percent are not supported truncations for bosonic baths.")
        super().__init__(type, threshold)

class Bosonic_Bath(Bath):
    def __init__(self, fragment):
        super().__init__(fragment)
        self.coeff, self.occup = self.kernel()

    @property
    def cluster_excitations(self):
        co = self.fragment.get_overlap('cluster[occ]|mo[occ]')
        cv = self.fragment.get_overlap('cluster[vir]|mo[vir]')
        ov_ss = einsum("Ii,Aa->IAia", co, cv).reshape(-1, co.shape[1] * cv.shape[1])
        return np.hstack((ov_ss, ov_ss))

    def kernel(self):
        mystr = "Making Bosonic Bath"
        self.log.info("%s", mystr)
        self.log.info("-" * len(mystr))
        self.log.changeIndentLevel(1)
        coeff, occup = self.make_boson_coeff()
        self.log_histogram(occup)
        self.log.changeIndentLevel(-1)
        self.coeff = coeff
        self.occup = occup
        return coeff, occup

    def make_boson_coeff(self):
        raise AbstractMethodError

    def get_bath(self, boson_threshold=None, **kwargs):
        return self.truncate_bosons(boson_threshold=boson_threshold, **kwargs)

    def truncate_bosons(self, coeff=None, occup=None, boson_threshold=None, verbose=True):
        if coeff is None:
            coeff = self.coeff
        if occup is None:
            occup = self.occup

        if isinstance(boson_threshold, numbers.Number):
            boson_threshold = Boson_Threshold('occupation', boson_threshold)

        boson_number = boson_threshold.get_number(occup)

        if verbose:
            fmt = "  %4s: N= %4d  max= % 9.3g  min= % 9.3g  sum= % 9.3g ( %7.3f %%)"
            def log_space(name, n_part):
                if len(n_part) == 0:
                    self.log.info(fmt[:fmt.index('max')].rstrip(), name, 0)
                    return
                with np.errstate(invalid='ignore'): # supress 0/0 warning
                    self.log.info(fmt, name, len(n_part), max(n_part), min(n_part), np.sum(n_part),
                            100*np.sum(n_part)/np.sum(occup))
            log_space("Bath", occup[:boson_number])
            log_space("Rest", occup[boson_number:])

        c_bath, c_rest = np.hsplit(coeff, [boson_number])
        return c_bath, c_rest

    def log_histogram(self, n_bos):
        if len(n_bos) == 0:
            return
        self.log.info("Bosonic Coupling histogram:")
        bins = np.hstack([-np.inf, np.logspace(-3, -10, 8)[::-1], np.inf])
        labels = '    ' + ''.join('{:{w}}'.format('E-%d' % d, w=5) for d in range(3, 11))
        self.log.info(helper.make_histogram(n_bos, bins=bins, labels=labels))
