"""WaveFunction for interaction with ebcc solvers for arbitrary ansatzes.
We subclass existing CCSD wavefunction functionality for the various utility functions (projection, symmetrisation etc),
but don't implement these for higher-order terms for now.
Where ebcc and pyscf have different conventions, we store quantities following the ebcc convention and return values
by pyscf convention for consistency and compatibility with existing interfaces.
"""

from vayesta.core.types.ebwf import EBWavefunction
from vayesta.core.types.wf.project import (
    project_u11_ferm,
    project_u11_bos,
    project_s1,
    project_s2,
    symmetrize_c2,
    symmetrize_s2,
)
from vayesta.core.types import RCCSD_WaveFunction, UCCSD_WaveFunction
from vayesta.core import spinalg
from vayesta.core.util import callif, dot, einsum

import ebcc
import numpy as np

from copy import deepcopy


def EBCC_WaveFunction(mo, *args, **kwargs):
    if mo.nspin == 1:
        cls = REBCC_WaveFunction
    elif mo.nspin == 2:
        cls = UEBCC_WaveFunction
    else:
        raise ValueError("EBCC WaveFunction is only implemented for mo.spin of 1 or 2.")
    return cls(mo, *args, **kwargs)


# Subclass existing CC methods only for the various utility functions (projection, symmetrisation etc).
# Just need to set properties to correctly interact with the ebcc storage objects.
# Notable convention differences between ebcc and pyscf:
# - ebcc includes an additional factor of 1/2 in the definition of T2aaaa, T2bbbb, L2aaaa, L2bbbb etc.
# - ebcc stores lambda amplitudes ai, abij while pyscf stores them ia, ijab.
class REBCC_WaveFunction(EBWavefunction, RCCSD_WaveFunction):
    _spin_type = "R"
    _driver = ebcc.REBCC

    def __init__(self, mo, ansatz, amplitudes, lambdas=None, mbos=None, projector=None, xi=None, ovlp_occ=None):
        super().__init__(mo, mbos, projector)
        self.amplitudes = amplitudes
        if lambdas is not None and len(lambdas) == 0:
            lambdas = None
        self.lambdas = lambdas
        if isinstance(ansatz, ebcc.Ansatz):
            self.ansatz = ansatz
        else:
            self.ansatz = ebcc.Ansatz.from_string(ansatz)
        self._eqns = self.ansatz._get_eqns(self._spin_type)
        self.xi = xi
        # Need this to relate quasibosonic spaces to their original fermionic indices.
        self._ovlp_occ = ovlp_occ

    @property
    def options(self):
        return ebcc.util.Namespace(shift=self.xi is not None)

    @property
    def inc_bosons(self):
        return self.nbos > 0

    @property
    def nbos(self):
        if "s1" in self.amplitudes:
            return self.amplitudes.s1.shape[0]
        return 0

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
        return None if self.lambdas is None else self.lambdas.l1.T

    @l1.setter
    def l1(self, value):
        if value is None:
            return
        if self.lambdas is None:
            self.lambdas = ebcc.util.Namespace()
        self.lambdas.l1 = value.T

    @property
    def l2(self):
        return None if self.lambdas is None else self.lambdas.l2.transpose((2, 3, 0, 1))

    @l2.setter
    def l2(self, value):
        if value is None:
            return
        if self.lambdas is None:
            self.lambdas = ebcc.util.Namespace()
        self.lambdas.l2 = value.transpose((2, 3, 0, 1))

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

    def make_rdm1(self, t_as_lambda=False, with_mf=True, ao_basis=False, hermitise=True, **kwargs):
        assert not t_as_lambda and with_mf and not ao_basis
        return self._driver.make_rdm1_f(
            self, eris=False, amplitudes=self.amplitudes, lambdas=self.lambdas, hermitise=True, **kwargs
        )

    def make_rdm2(
        self, t_as_lambda=False, with_dm1=True, ao_basis=False, approx_cumulant=False, hermitise=True, **kwargs
    ):
        assert not t_as_lambda and with_dm1 and not ao_basis and not approx_cumulant
        return self._driver.make_rdm2_f(
            self, eris=False, amplitudes=self.amplitudes, lambdas=self.lambdas, hermitise=hermitise, **kwargs
        )

    def make_rdm1_b(self, hermitise=True, **kwargs):
        return self._driver.make_rdm1_b(
            self, eris=False, amplitudes=self.amplitudes, lambdas=self.lambdas, hermitise=hermitise, **kwargs
        )

    def make_sing_b_dm(self, hermitise=True, **kwargs):
        return self._driver.make_sing_b_dm(
            self, eris=False, amplitudes=self.amplitudes, lambdas=self.lambdas, hermitise=hermitise, **kwargs
        )

    def make_rdm_eb(self, hermitise=True, **kwargs):
        dmeb = self._driver.make_eb_coup_rdm(
            self, eris=False, amplitudes=self.amplitudes, lambdas=self.lambdas, hermitise=hermitise, **kwargs
        )
        return (dmeb[0].transpose((1, 2, 0)) / 2, dmeb[0].transpose((1, 2, 0)) / 2)

    make_rdm1_f = make_rdm1
    make_rdm2_f = make_rdm2

    def project(self, projector, inplace=False, project_bosons=True):
        wf = super().project(projector, inplace)

        if (not self.inc_bosons) or (not project_bosons):
            return wf

        # Construct projector for bosonic space.
        # Note that we just directly project the bosonic amplitudes rather than storing in an intermediate state, as
        # any efficiency gains are comparatively minimal.)
        ex_coeff = np.tensordot(self.mbos.coeff_ex_3d[0], dot(projector, self._ovlp_occ), axes=((1,), (1,)))

        pbos = 2 * np.tensordot(
            ex_coeff, ex_coeff, axes=((1, 2), (1, 2))
        )  # Sum over both spin components of the bosons.

        wf.amplitudes.u11_ferm = project_u11_ferm(self.amplitudes.u11, dot(projector.T, projector))
        wf.amplitudes.u11 = (
            project_u11_ferm(self.amplitudes.u11, dot(projector.T, projector))
            + project_u11_bos(self.amplitudes.u11, pbos)
        ) / 2
        wf.amplitudes.s1 = project_s1(self.amplitudes.s1, pbos)
        if "s2" in self.amplitudes:
            wf.amplitudes.s2 = project_s2(self.amplitudes.s2, pbos)

        if wf.lambdas is None:
            return wf

        wf.lambdas.lu11_ferm = project_u11_ferm(
            self.lambdas.lu11.transpose(0, 2, 1), dot(projector.T, projector)
        ).transpose(0, 2, 1)
        wf.lambdas.lu11 = (
            project_u11_ferm(self.lambdas.lu11.transpose(0, 2, 1), dot(projector.T, projector)).transpose(0, 2, 1)
            + project_u11_bos(self.lambdas.lu11, pbos)
        ) / 2
        wf.lambdas.ls1 = project_s1(self.lambdas.ls1, pbos)
        if "ls2" in self.lambdas:
            wf.lambdas.ls2 = project_s2(self.lambdas.ls2, pbos)
        return wf

    def restore(self, projector=None, inplace=False, sym=True):
        if projector is None:
            projector = self.projector
        wf = self.project(projector.T, inplace=inplace, project_bosons=False)
        wf.projector = None
        if not sym:
            return wf
        wf.t2 = symmetrize_c2(wf.t2)
        if wf.l2 is None:
            return wf
        wf.l2 = symmetrize_c2(wf.l2)

        if "s2" in self.amplitudes:
            wf.amplitudes.s2 = symmetrize_s2(wf.amplitudes.s2)
        if wf.lambdas is not None:
            if "ls2" in self.lambdas:
                wf.lambdas.ls2 = symmetrize_s2(wf.lambdas.ls2)
        return wf

    def copy(self):
        proj = callif(spinalg.copy, self.projector)
        return type(self)(
            self.mo.copy(),
            deepcopy(self.ansatz),
            deepcopy(self.amplitudes),
            deepcopy(self.lambdas),
            None if self.mbos is None else self.mbos.copy(),
            proj,
        )

    def as_ccsd(self):
        proj = callif(spinalg.copy, self.projector)
        return type(self)(
            self.mo.copy(), "CCSD", deepcopy(self.amplitudes), deepcopy(self.lambdas), self.mbos.copy(), proj
        )

    def rotate_ov(self, *args, **kwargs):
        # Note that this is slightly dodgy until we implement rotation of the coupled amplitudes.
        if "t3" in self.amplitudes:
            # can't access log within wavefunction classes currently; this should be a warning.
            pass
            # raise NotImplementedError("Only rotation of CCSD components is implemented.")
        return super().rotate_ov(*args, **kwargs)

    def expand_quasibosons(self, ovlp, fglobal=None):
        """For a coupled electron-quasiboson wavefunction, expand the quasibosonic degrees of freedom into an overall,
        fermionic wavefunction in the full space.
        """
        if self.nbos < 1:
            return self.copy()
        if not hasattr(self.mbos, "forbitals"):
            raise ValueError(
                "Bosonic component of wavefunction lacks information required for expansion of quasibosons."
            )
        # Default to using basis in which quasibosons are defined.
        fglobal = fglobal or self.mbos.forbitals

        # First, transform fermionic contributions.
        ro_ferm = dot(fglobal.coeff_occ.T, ovlp, self.mo.coeff_occ)
        rv_ferm = dot(fglobal.coeff_vir.T, ovlp, self.mo.coeff_vir)
        t1 = dot(ro_ferm, self.t1, rv_ferm.T)
        t2 = einsum("ijab,Ii,Jj,Aa,Bb->IJAB", self.t2, ro_ferm, ro_ferm, rv_ferm, rv_ferm)

        # Now, transform bosonic contributions.
        ca, cb = self.mbos.coeff_ex_3d

        ro_bos = dot(fglobal.coeff_occ.T, ovlp, self.mbos.forbitals.coeff_occ)
        rv_bos = dot(fglobal.coeff_vir.T, ovlp, self.mbos.forbitals.coeff_vir)

        # Check that spin components are identical.
        assert abs(ca - cb).max() < 1e-8
        u11 = self.amplitudes.u11
        s1 = self.amplitudes.s1
        s2 = self.amplitudes.s2

        rbos_global = einsum("nia,Ii,Aa->nIA", ca, ro_bos, rv_bos)

        t1 += np.tensordot(s1, rbos_global, 1)
        t2 += np.tensordot(rbos_global, einsum("nia,Ii,Aa->nIA", u11, ro_ferm, rv_ferm), ((0,), (0,))).transpose(
            0, 2, 1, 3
        )
        t2 += einsum("nm,nia,mjb->ijab", s2, rbos_global, rbos_global)

        print(ro_ferm.shape, ro_bos.shape, rbos_global.shape, t1.shape, t2.shape)

        return type(self)(
            fglobal.copy(),
            "CCSD",
            amplitudes=ebcc.util.Namespace(t1=t1, t2=t2),
            lambdas=None,
            mbos=None,
            projector=None,
        )


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
        return 2 * self.amplitudes.t2.aaaa

    @property
    def t2ab(self):
        return self.amplitudes.t2.abab

    @property
    def t2ba(self):
        return self.amplitudes.t2.abab.transpose(1, 0, 3, 2)

    @property
    def l2ba(self):
        if "baba" in self.amplitudes.t2:
            return self.amplitudes.t2.baba
        else:
            return self.t2ab.transpose(1, 0, 3, 2)

    @property
    def t2bb(self):
        return 2 * self.amplitudes.t2.bbbb

    @property
    def t2(self):
        return (self.t2aa, self.t2ab, self.t2bb)

    @t2.setter
    def t2(self, value):
        self.amplitudes.t2.aaaa = 0.5 * value[0]
        self.amplitudes.t2.abab = value[1]
        self.amplitudes.t2.bbbb = 0.5 * value[-1]
        if len(value) == 4:
            self.amplitudes.t2.baba = value[2]

    @property
    def l1a(self):
        return None if self.lambdas is None else self.lambdas.l1.aa.T

    @property
    def l1b(self):
        return None if self.lambdas is None else self.lambdas.l1.bb.T

    @property
    def l1(self):
        return None if self.lambdas is None else (self.l1a, self.l1b)

    @l1.setter
    def l1(self, value):
        if value is None:
            return
        if self.lambdas is None:
            self.lambdas = ebcc.util.Namespace()
        self.lambdas.l1.aa = value[0].T
        self.lambdas.l1.bb = value[1].T

    @property
    def l2aa(self):
        return None if self.lambdas is None else 2 * self.lambdas.l2.aaaa.transpose(2, 3, 0, 1)

    @property
    def l2ab(self):
        return None if self.lambdas is None else self.lambdas.l2.abab.transpose(2, 3, 0, 1)

    @property
    def l2ba(self):
        if self.lambdas is None:
            return None
        if "baba" in self.lambdas.l2:
            return self.lambdas.l2.baba
        else:
            return self.l2ab.transpose(1, 0, 3, 2)

    @property
    def l2bb(self):
        return None if self.lambdas is None else 2 * self.lambdas.l2.bbbb.transpose(2, 3, 0, 1)

    @property
    def l2(self):
        return None if self.lambdas is None else (self.l2aa, self.l2ab, self.l2bb)

    @l2.setter
    def l2(self, value):
        if value is None:
            return
        if self.lambdas is None:
            self.lambdas = ebcc.util.Namespace()
        self.lambdas.l2.aaaa = value[0].transpose(2, 3, 0, 1) / 2.0
        self.lambdas.l2.abab = value[1].transpose(2, 3, 0, 1)
        self.lambdas.l2.bbbb = value[-1].transpose(2, 3, 0, 1) / 2.0
        if len(value) == 4:
            self.lambdas.l2.baba = value[2].transpose(2, 3, 0, 1)

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

    def make_rdm1(self, *args, **kwargs):
        dm1 = super().make_rdm1(*args, **kwargs)
        return dm1.aa, dm1.bb

    def make_rdm2(self, *args, **kwargs):
        dm2 = super().make_rdm2(*args, **kwargs)
        return dm2.aaaa, dm2.aabb, dm2.bbbb

    def make_rdm_eb(self, hermitise=True, **kwargs):
        dmeb = self._driver.make_eb_coup_rdm(
            self, eris=False, amplitudes=self.amplitudes, lambdas=self.lambdas, hermitise=hermitise, **kwargs
        )

        return (dmeb.aa[0].transpose(1, 2, 0), dmeb.bb[0].transpose(1, 2, 0))
