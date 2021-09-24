
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

    def check_occupation(self, c_occ, c_vir, *args, **kwargs):
        if tol is None: tol = 2*self.opts.dmet_threshold
        n_occ = self.get_mo_occupation(c_occ)
        n_vir = self.get_mo_occupation(c_vir)
        for s, spin in enumerate(('alpha', 'beta')):
            if not np.allclose(n_occ[s], 1, atol=tol):
                raise RuntimeError("Incorrect occupation of occupied %s-orbitals:\n%r" % (spin, n_occ))
            if not np.allclose(n_vir[s], 0, atol=tol):
                raise RuntimeError("Incorrect occupation of virtual %s-orbitals:\n%r" % (spin, n_vir))
