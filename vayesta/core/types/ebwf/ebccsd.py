from vayesta.core.types import ebwf

import ebcc
import numpy as np

# Note that we don't subclass the existing CCSD_WaveFunction class since we need to use ebcc as a backend, rather than
# pyscf.

class REBCC_WaveFunction(ebwf.EBWavefunction):

    _spin_type = "R"
    _driver = ebcc.REBCC

    def __init__(self, mo, ansatz, amplitudes, lambdas=None, mbos=None, projector=None):
        super().__init__(mo, mbos, projector)
        self.amplitudes = amplitudes
        self.lambdas = lambdas
        if isinstance(ansatz, ebcc.Ansatz):
            self.ansatz = ansatz
        else:
            self.ansatz = ebcc.Ansatz.from_string(ansatz)
        self._eqns = self.ansatz._get_eqns(self._spin_type)

    @property
    def name(self):
        """Get a string representation of the method name."""
        return self._spin_type + self.ansatz.name

    @property
    def t1(self):
        return self.amplitudes.t1

    @property
    def t2(self):
        return self.amplitudes.t2

    @property
    def l1(self):
        return None if self.lambdas is None else self.lambdas.l1

    @property
    def l2(self):
        return None if self.lambdas is None else self.lambdas.l2

    def _load_function(self, *args, **kwargs):
        return self._driver._load_function(self, *args, **kwargs)

    def _pack_codegen_kwargs(self, *extra_kwargs, eris=False):
        """
        Pack all the possible keyword arguments for generated code
        into a dictionary.
        """
        eris = False
        # This is always accessed but never used for any density matrix calculation.
        g = ebcc.util.Namespace()
        g["boo"] = g["bov"] = g["bvo"] = g["bvv"] = np.zeros((self.nbos, 0, 0))
        kwargs = dict(
            v=eris,
            g=g,
            nocc=self.mo.nocc,
            nvir=self.mo.nvir,
            nbos=self.nbos,
        )
        for kw in extra_kwargs:
            if kw is not None:
                kwargs.update(kw)
        return kwargs

    def make_rdm1(self, *args, **kwargs):
        return self._driver.make_rdm1_f(self, eris=False, amplitudes=self.amplitudes, lambdas=self.lambdas, hermitise=True)

    def make_rdm2(self, *args, **kwargs):
        return self._driver.make_rdm2_f(self, eris=False, amplitudes=self.amplitudes, lambdas=self.lambdas, hermitise=True)

    def make_rdm1_b(self, *args, **kwargs):
        return self._driver.make_rdm1_b(self, eris=False, amplitudes=self.amplitudes, lambdas=self.lambdas, hermitise=True)

    def make_sing_b_dm(self, *args, **kwargs):
        return self._driver.make_sing_b_dm(self, eris=False, amplitudes=self.amplitudes, lambdas=self.lambdas, hermitise=True)

    def make_eb_coup_rdm(self, *args, **kwargs):
        return self._driver.make_eb_coup_rdm(self, eris=False, amplitudes=self.amplitudes, lambdas=self.lambdas, hermitise=True)


class UEBCC_WaveFunction(REBCC_WaveFunction):
    _spin_type = "U"
    _driver = ebcc.UEBCC

    @property
    def t1a(self):
        return self.amplitudes.t1.aa

    @property
    def t1b(self):
        return self.amplitudes.t1.bb

    @property
    def t1(self):
        return (self.t1a, self.t1b)

    @property
    def t2aa(self):
        return self.amplitudes.t2.aaaa

    @property
    def t2ab(self):
        return self.amplitudes.t2.aabb

    @property
    def t2ba(self):
        return self.amplitudes.t2.bbaa

    @property
    def t2bb(self):
        return self.amplitudes.t2.bbbb

    @property
    def t2(self):
        return (self.t2aa, self.t2ab, self.t2bb)

    def _pack_codegen_kwargs(self, *extra_kwargs, eris=False):
        """
        Pack all the possible keyword arguments for generated code
        into a dictionary.
        """
        eris = False
        # This is always accessed but never used for any density matrix calculation.
        g = ebcc.util.Namespace()
        g["aa"] = ebcc.util.Namespace()
        g["aa"]["boo"] = g["aa"]["bov"] = g["aa"]["bvo"] = g["aa"]["bvv"] = np.zeros((self.nbos, 0, 0))
        g["bb"] = g["aa"]
        kwargs = dict(
            v=eris,
            g=g,
            nocc=self.mo.nocc,
            nvir=self.mo.nvir,
            nbos=self.nbos,
        )
        for kw in extra_kwargs:
            if kw is not None:
                kwargs.update(kw)
        return kwargs
