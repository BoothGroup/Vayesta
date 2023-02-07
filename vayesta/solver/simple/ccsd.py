"""Module for CCSD solvers in terms of MO-integrals for small problem sizes."""

import numpy as np
import pyscf
import pyscf.gto
import pyscf.scf
import pyscf.cc
import pyscf.lib
import pyscf.ao2mo
import vayesta
import vayesta.core.ao2mo.pyscf_eris
from vayesta.core.types.wf import WaveFunction


def _pack_ovvv(ovvv):
    no, nv = ovvv.shape[:2]
    nvp = nv*(nv+1)//2
    ovvv = pyscf.lib.pack_tril(ovvv.reshape(no*nv,nv,nv)).reshape(no,nv,nvp)
    return ovvv


def _pack_vvvv(vvvv):
    nv = vvvv.shape[0]
    return pyscf.ao2mo.restore(4, vvvv, nv)


def CCSD(fock, eris, nocc, **kwargs):
    ndim = (np.ndim(fock[0]) + 1)
    if ndim == 2:
        return RCCSD(fock, eris, nocc, **kwargs)
    if ndim == 3:
        return UCCSD(fock, eris, nocc, **kwargs)
    raise ValueError("Fock ndim= %d" % ndim)


class RCCSD:
    """Light-weight CCSD solver."""

    def __init__(self, fock, eris, nocc, mo_energy=None, conv_tol=1e-8, conv_tol_normt=1e-6):
        mf = self.init_mf(fock, nocc)
        self.eris = self.get_eris(fock, eris, nocc=nocc, mo_energy=mo_energy)
        self.solver = pyscf.cc.CCSD(mf)
        self.solver.conv_tol = conv_tol
        self.solver.conv_tol_normt = conv_tol_normt
        self.solver.get_e_hf = (lambda self: 0.0).__get__(self.solver)
        # Result
        self.wf = None

    def init_mf(self, fock, nocc):
        mol = pyscf.gto.M()
        mol.verbose = 0
        mf = pyscf.scf.RHF(mol)
        nmo = fock.shape[-1]
        nvir = nmo - nocc
        mo_coeff = np.eye(nmo)
        mo_occ = np.zeros(nmo)
        mo_occ[:nocc] = 2
        mf.mo_coeff = mo_coeff
        mf.mo_occ = mo_occ
        return mf

    @property
    def e_tot(self):
        raise NotImplementedError

    def __getattr__(self, key):
        if key in ['converged', 'e_corr']:
            return getattr(self.solver, key)
        raise AttributeError

    def get_eris(self, fock, eris, nocc, mo_energy=None):
        return vayesta.core.ao2mo.pyscf_eris.make_ccsd_eris(fock, eris, nocc, mo_energy=mo_energy)

    def kernel(self, t1=None, t2=None):
        self.solver.kernel(t1=t1, t2=t2, eris=self.eris)
        self.wf = WaveFunction.from_pyscf(self.solver, eris=self.eris)
        return self.wf


class UCCSD(RCCSD):

    def init_mf(self, fock, nocc):
        mol = pyscf.gto.M()
        mol.verbose = 0
        mf = pyscf.scf.UHF(mol)
        nmo = (fock[0].shape[-1], fock[1].shape[-1])
        nvir = (nmo[0]-nocc[0], nmo[1]-nocc[1])
        mo_coeff = (np.eye(nmo[0]), np.eye(nmo[1]))
        mo_occ = (np.zeros(nmo[0]), np.zeros(nmo[1]))
        mo_occ[0][:nocc[0]] = 1
        mo_occ[1][:nocc[1]] = 1
        mf.mo_coeff = mo_coeff
        mf.mo_occ = mo_occ
        return mf

    def get_eris(self, fock, eris, nocc, mo_energy=None):
        return vayesta.core.ao2mo.pyscf_eris.make_uccsd_eris(fock, eris, nocc, mo_energy=mo_energy)
