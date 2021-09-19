"""Self-consistent mean-field decorators"""

import copy

import numpy as np
import scipy
import scipy.linalg

import pyscf
import pyscf.lib

from vayesta.misc import PCDIIS
from vayesta.core.util import *


class SCMF:

    def __init__(self, qemb, etol=1e-8, dtol=1e-6, maxiter=100, damping=0.0, diis=True):
        self.qemb = qemb
        self.etol = (etol if etol is not None else np.inf)
        self.dtol = (dtol if dtol is not None else np.inf)
        self.maxiter = maxiter
        self.damping = damping
        self.diis = diis
        self.iteration = 0
        # Save original kernel
        self._kernel_orig = self.qemb.kernel
        # Save original orbitals
        self._mo_orig = self.mf.mo_coeff
        # Output
        self.converged = False
        self.energies = []          # Total energy per iteration

    @property
    def log(self):
        return self.qemb.log

    @property
    def mf(self):
        return self.qemb.mf

    @property
    def e_tot(self):
        return self.energies[-1]

    @property
    def e_tot_oneshot(self):
        return self.energies[0]

    def update_mo_coeff(self, mf, diis=None):
        """Must be implemented for any SCMF method."""
        raise NotImplementedError()

    #def update_mf(self, mo_coeff):
    #    if not np.allclose(dot(mo_coeff.T, self.mf.get_ovlp(), mo_coeff) - np.eye(mo_coeff.shape[-1]), 0):
    #        raise ValueError("Input MO coefficients not orthonormal!")
    #    mf_new = copy.copy(self.mf)
    #    mf_new.mo_coeff = mo_coeff
    #    mf_new.mo_energy = None
    #    mf_new.e_tot = mf_new.energy_tot()

    #    return mf_new

    def update_mf(self, mo_coeff):
        if not np.allclose(dot(mo_coeff.T, self.mf.get_ovlp(), mo_coeff) - np.eye(mo_coeff.shape[-1]), 0):
            raise ValueError("Input MO coefficients not orthonormal!")
        self.qemb.update_mf(mo_coeff)

    def kernel(self, *args, **kwargs):
        diis = (self.get_diis() if self.diis else None)

        for self.iteration in range(1, self.maxiter+1):

            self.log.info("SCMF iteration %3d", self.iteration)
            self.log.info("==================")

            # Run clusters, save results
            res = self._kernel_orig(*args, **kwargs)
            #res = self.qemb.kernel(*args, **kwargs)
            #res = self.qemb.kernel_one_iteration(*args, **kwargs)
            e_mf = self.mf.e_tot
            e_corr = self.qemb.get_e_corr()
            e_tot = (e_mf + e_corr)
            self.energies.append(e_tot)

            # Update MF
            mo_coeff = self.update_mo_coeff(self.mf, diis=diis)
            self.update_mf(mo_coeff)
            #mf = self.update_mf(mo_coeff)
            #self.qemb.mf = mf

            # Check convergence
            dm1 = self.mf.make_rdm1()
            if self.iteration > 1:
                de = (e_tot - e0)
                ddm = abs(dm1-dm0).max()
            else:
                de = ddm = np.inf
            e0, dm0 = e_tot, dm1
            self.log.output("SCMF iteration %3d (dE= %+16.8f Ha  dDM= %9.3e): E(mf)= %+16.8f Ha  E(corr)= %+16.8f Ha  E(tot)= %+16.8f Ha",
                    self.iteration, de, ddm, e_mf, e_corr, e_tot)
            if (abs(de) < (1-self.damping)*self.etol) and (ddm < (1-self.damping)*self.dtol):
                self.log.info("SCMF converged in %d iterations", self.iteration)
                self.converged = True
                break

            #self.qemb._veff = None
            self.qemb.reset_fragments()

        else:
            self.log.warning("SCMF did not converge in %d iterations!", self.iteration)
        return res


class PDMET_SCMF(SCMF):

    def __init__(self, *args, dm_type='demo', **kwargs):
        super().__init__(*args, **kwargs)
        self.dm_type = dm_type.lower()

    def get_diis(self):
        return pyscf.lib.diis.DIIS()

    def get_rdm1(self):
        """DM1 in MO basis."""
        dm_type = self.dm_type
        if dm_type.startswith('demo'):
            dm1 = self.qemb.make_rdm1_demo()
        elif dm_type.startswith('pwf-ccsd'):
            dm1 = self.qemb.make_rdm1_ccsd()
        else:
            raise NotImplementedError("dm_type= %r" % dm_type)
        # Check electron number
        nelec_err = abs(np.trace(dm1) - self.qemb.mol.nelectron)
        if nelec_err > 1e-5:
            self.log.warning("Large electron error in 1DM= %.3e", nelec_err)
        return dm1

    def update_mo_coeff(self, mf, diis=None):
        dm1 = self.get_rdm1()
        # Transform to original MO basis
        r = dot(self.qemb.mo_coeff.T, self.qemb.get_ovlp(), self._mo_orig)
        dm1 = dot(r.T, dm1, r)
        if diis is not None:
            dm1 = diis.update(dm1)
        mo_occ, rot = np.linalg.eigh(dm1)
        mo_occ, rot = mo_occ[::-1], rot[:,::-1]
        nocc = np.count_nonzero(mf.mo_occ > 0)
        self.log.debug("p-DMET MO occupation numbers (occupied):\n%s", mo_occ[:nocc])
        self.log.debug("p-DMET MO occupation numbers (virtual):\n%s", mo_occ[nocc:])
        if abs(mo_occ[nocc-1] - mo_occ[nocc]) < 1e-8:
            raise RuntimeError("Degeneracy in MO occupation:\n%r" % mo_occ)
        mo_coeff = np.dot(self._mo_orig, rot)
        return mo_coeff


class Brueckner_SCMF(SCMF):

    def __init__(self, *args, diis_obj='dm1', **kwargs):
        super().__init__(*args, **kwargs)
        self.diis_obj = diis_obj.lower()

    def get_diis(self):
        return pyscf.lib.diis.DIIS()
        #PC-DIIS
        #nocc = np.count_nonzero(self.mf.mo_occ > 0)
        #diis = PCDIIS(self._mo_orig[:,:nocc].copy())
        #return diis

    def get_t1(self):
        return self.qemb.get_t1()

    def update_mo_coeff(self, mf, diis=None):
        t1 = self.get_t1()
        self.log.debug("Norm of T1: L(2)= %.3e  L(inf)= %.3e", np.linalg.norm(t1), abs(t1).max())
        nocc, nvir = t1.shape
        nmo = (nocc + nvir)
        occ, vir = np.s_[:nocc], np.s_[nocc:]
        if diis and self.diis_obj == 't1':
            # Transform to original MO basis
            ro = dot(mf.mo_coeff[:,occ].T, self.qemb.get_ovlp(), self._mo_orig)
            rv = dot(mf.mo_coeff[:,vir].T, self.qemb.get_ovlp(), self._mo_orig)
            t1 = dot(ro.T, t1, rv)
            t1 = diis.update(t1, xerr=t1)
            ## Transform back
            t1 = dot(ro, t1, rv.T)

        mo_update = (1-self.damping)*np.dot(mf.mo_coeff[:,vir], t1.T)
        self.log.debug("Change of occupied Brueckner orbitals= %.3e", np.linalg.norm(mo_update))
        bmo_occ = (mf.mo_coeff[:,occ] +  mo_update)

        # Orthogonalize occupied orbitals
        # If there was no AO-overlap...:
        #bmo_occ = np.linalg.qr(bmo_occ)[0]
        ovlp = mf.get_ovlp()
        dm_occ = np.dot(bmo_occ, bmo_occ.T)
        e, v = scipy.linalg.eigh(dm_occ, b=ovlp, type=2)
        bmo_occ = v[:,-nocc:]
        # DIIS of occupied density
        if diis and self.diis_obj == 'dm1':
            dm_occ = np.dot(bmo_occ, bmo_occ.T)
            r = np.dot(self.qemb.get_ovlp(), self._mo_orig)
            dm_occ = dot(r.T, dm_occ, r)
            dm_occ = diis.update(dm_occ)
            e, v = np.linalg.eigh(dm_occ)
            bmo_occ = np.dot(self._mo_orig, v)[:,-nocc:]

        # Virtual space
        dm_vir = (np.linalg.inv(ovlp) - np.dot(bmo_occ, bmo_occ.T))
        e, v = scipy.linalg.eigh(dm_vir, b=ovlp, type=2)
        bmo_vir = v[:,-nvir:]

        assert (bmo_occ.shape[-1] == nocc) and (bmo_vir.shape[-1] == nvir)
        mo_coeff = np.hstack((bmo_occ, bmo_vir))
        return mo_coeff
