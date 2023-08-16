from vayesta.core.types import ebwf
from vayesta.core.types import RCCSD_WaveFunction, UCCSD_WaveFunction
from vayesta.core import spinalg
from vayesta.core.util import callif

import ebcc
import numpy as np

from copy import deepcopy


# Subclass existing CC methods only for the various utility functions (projection, symmetrisation etc).
# Just need to set properties to correctly interact with the ebcc storage objects.
class REBCC_WaveFunction(ebwf.EBWavefunction, RCCSD_WaveFunction):

    _spin_type = "R"
    _driver = ebcc.REBCC

    def __init__(self, mo, ansatz, amplitudes, lambdas=None, mbos=None, projector=None):
        super(ebwf.EBWavefunction, self).__init__(mo, mbos, projector)
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

    @t1.setter
    def t1(self, value):
        self.amplitudes.t1 = value

    @property
    def t2(self):
        return self.amplitudes.t2

    @t2.setter
    def t2(self, value):
        self.amplitudes.t2 = value

    @property
    def l1(self):
        return None if self.lambdas is None else self.lambdas.l1

    @l1.setter
    def l1(self, value):
        if self.lambdas is None:
            self.lambdas = ebcc.util.Namespace()
        self.lambdas.l1 = value

    @property
    def l2(self):
        return None if self.lambdas is None else self.lambdas.l2

    @l2.setter
    def l2(self, value):
        if self.lambdas is None:
            self.lambdas = ebcc.util.Namespace()
        self.lambdas.l2 = value

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

    def copy(self):
        proj = callif(spinalg.copy, self.projector)
        type(self)(self.mo.copy(), deepcopy(self.ansatz), deepcopy(self.amplitudes), deepcopy(self.lambdas),
                   self.mbos.copy(), proj)

    def as_ccsd(self):
        proj = callif(spinalg.copy, self.projector)
        return type(self)(self.mo.copy(), "CCSD", deepcopy(self.amplitudes), deepcopy(self.lambdas),
                          self.mbos.copy(), proj)


class UEBCC_WaveFunction(REBCC_WaveFunction, UCCSD_WaveFunction):
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

    @t1.setter
    def t1(self, value):
        self.amplitudes.t1.aa = value[0]
        self.amplitudes.t1.bb = value[1]

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

    @t2.setter
    def t2(self, value):
        self.amplitudes.t2.aaaa = value[0]
        self.amplitudes.t2.aabb = value[1]
        self.amplitudes.t2.bbbb = value[-1]
        if len(value) == 4:
            self.amplitudes.t2.bbaa = value[2]
        else:
            self.amplitudes.t2.bbaa = value[1].transpose(1, 0, 3, 2)

    @property
    def l1a(self):
        return None if self.lambdas is None else self.lambdas.l1.aa

    @property
    def l1b(self):
        return None if self.lambdas is None else self.lambdas.l1.bb

    @property
    def l1(self):
        return None if self.lambdas is None else (self.l1a, self.l1b)

    @l1.setter
    def l1(self, value):
        if self.lambdas is None:
            self.lambdas = ebcc.util.Namespace()
        self.lambdas.l1.aa = value[0]
        self.lambdas.l1.bb = value[1]

    @property
    def l2aaaa(self):
        return None if self.lambdas is None else self.lambdas.l2.aaaa

    @property
    def l2aabb(self):
        return None if self.lambdas is None else self.lambdas.l2.aabb

    @property
    def l2bbaa(self):
        return None if self.lambdas is None else self.lambdas.l2.bbaa

    @property
    def l2bbbb(self):
        return None if self.lambdas is None else self.lambdas.l2.bbbb

    @property
    def l2(self):
        return None if self.lambdas is None else (self.l2aaaa, self.l2aabb, self.l2bbbb)

    @l2.setter
    def l2(self, value):
        if self.lambdas is None:
            self.lambdas = ebcc.util.Namespace()
        self.lambdas.l2.aaaa = value[0]
        self.lambdas.l2.aabb = value[1]
        self.lambdas.l2.bbbb = value[-1]
        if len(value) == 4:
            self.lambdas.l2.bbaa = value[2]
        else:
            self.lambdas.l2.bbaa = value[1].transpose(1, 0, 3, 2)

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
