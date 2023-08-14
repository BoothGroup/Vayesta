from vayesta.core.types import ebwf

import ebcc
import numpy as np

# Note that we don't subclass the existing CCSD_WaveFunction class since we need to use ebcc as a backend, rather than
# pyscf.

class EB_RCCSD_WaveFunction(ebwf.EBWavefunction):

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

    # This allows us to access all relevant amplitudes and lambdas as attributes of the wavefunction object.
    def __getattribute__(self, key):
        amps = super(EB_RCCSD_WaveFunction, self).__getattribute__("amplitudes")
        lambdas = super(EB_RCCSD_WaveFunction, self).__getattribute__("lambdas")
        if key in amps._keys:
            return amps[key]
        if lambdas is not None:
            if key in lambdas._keys:
                return lambdas[key]
        return super(EB_RCCSD_WaveFunction, self).__getattribute__(key)

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


class EB_UCCSD_WaveFunction(EB_RCCSD_WaveFunction):
    _spin_type = "U"
    _driver = ebcc.UEBCC
