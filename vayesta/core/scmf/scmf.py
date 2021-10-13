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

    name = "SCMF"

    def __init__(self, emb, etol=1e-8, dtol=1e-6, maxiter=100, damping=0.0, diis=True):
        self.emb = emb
        self.etol = (etol if etol is not None else np.inf)
        self.dtol = (dtol if dtol is not None else np.inf)
        self.maxiter = maxiter
        self.damping = damping
        self.diis = diis
        self.iteration = 0
        # Save original kernel
        self._kernel_orig = self.emb.kernel
        # Save original orbitals
        self._mo_orig = self.mf.mo_coeff
        # Output
        self.converged = False
        self.energies = []          # Total energy per iteration

    @property
    def log(self):
        return self.emb.log

    @property
    def mf(self):
        return self.emb.mf

    @property
    def e_tot(self):
        return self.energies[-1]

    @property
    def e_tot_oneshot(self):
        return self.energies[0]

    @property
    def kernel_orig(self):
        """Original kernel of embedding method."""
        return self._kernel_orig

    def update_mo_coeff(self, mf, diis=None):
        """Get new set of MO coefficients.

        Must be implemented for any SCMF method."""
        raise AbstractMethodError()

    def check_convergence(self, e_tot, dm1, e_last=None, dm1_last=None, etol=None, dtol=None):
        if etol is None: etol = self.etol
        if dtol is None: dtol = self.dtol
        if e_last is not None:
            de = (e_tot - e_last)
            # RHF:
            if self.emb.is_rhf:
                ddm = abs(dm1-dm1_last).max() / 2
            else:
            # UHF:
                ddm = max(abs(dm1[0]-dm1_last[0]).max(),
                          abs(dm1[1]-dm1_last[1]).max())
        else:
            de = ddm = np.inf
        tighten = (1-self.damping)
        if (abs(de) < tighten*etol) and (ddm < tighten*dtol):
            return True, de, ddm
        return False, de, ddm

    def kernel(self, *args, **kwargs):
        diis = (self.get_diis() if self.diis else None)

        e_last = dm1_last = None
        for self.iteration in range(1, self.maxiter+1):

            self.log.info("%s iteration %3d", self.name, self.iteration)
            self.log.info("%s==============", len(self.name)*"=")

            # Run clusters, save results
            res = self.kernel_orig(*args, **kwargs)
            e_mf = self.mf.e_tot
            e_corr = self.emb.get_e_corr()
            e_tot = (e_mf + e_corr)
            self.energies.append(e_tot)

            # Update MF
            mo_coeff = self.update_mo_coeff(self.mf, diis=diis)
            self.emb.update_mf(mo_coeff)

            dm1 = self.mf.make_rdm1()
            # Check symmetry
            self.emb.check_fragment_symmetry(dm1)

            # Check convergence
            conv, de, ddm = self.check_convergence(e_tot, dm1, e_last, dm1_last)
            fmt = "%s iteration %3d (dE= %s  dDM= %9.3e): E(MF)= %s  E(corr)= %s  E(tot)= %s"
            estr = energy_string
            self.log.output(fmt, self.name, self.iteration, estr(de), ddm, estr(e_mf), estr(e_corr), estr(e_tot))
            if conv:
                self.log.info("%s converged in %d iterations", self.name, self.iteration)
                self.converged = True
                break
            e_last, dm1_last = e_tot, dm1

            # Reset
            self.emb.reset_fragments()
        else:
            self.log.warning("%s did not converge in %d iterations!", self.name, self.iteration)
        return res


class PDMET_RHF(SCMF):

    name = "p-DMET"

    def __init__(self, *args, dm_type='demo', **kwargs):
        super().__init__(*args, **kwargs)
        self.dm_type = dm_type.lower()

    def get_diis(self):
        return pyscf.lib.diis.DIIS()

    def get_rdm1(self):
        """DM1 in MO basis."""
        dm_type = self.dm_type
        if dm_type.startswith('demo'):
            dm1 = self.emb.make_rdm1_demo()
        elif dm_type.startswith('pwf-ccsd'):
            dm1 = self.emb.make_rdm1_ccsd()
        else:
            raise NotImplementedError("dm_type= %r" % dm_type)
        # Check electron number
        nelec_err = abs(np.trace(dm1) - self.emb.mol.nelectron)
        if nelec_err > 1e-5:
            self.log.warning("Large electron error in 1DM= %.3e", nelec_err)
        return dm1

    def update_mo_coeff(self, mf, diis=None):
        dm1 = self.get_rdm1()
        # Transform to original MO basis
        r = dot(self.emb.mo_coeff.T, self.emb.get_ovlp(), self._mo_orig)
        dm1 = dot(r.T, dm1, r)
        if diis is not None:
            dm1 = diis.update(dm1)
        mo_occ, rot = np.linalg.eigh(dm1)
        mo_occ, rot = mo_occ[::-1], rot[:,::-1]
        nocc = np.count_nonzero(mf.mo_occ > 0)
        if abs(mo_occ[nocc-1] - mo_occ[nocc]) < 1e-8:
            self.log.critical("p-DMET MO occupation numbers (occupied):\n%s", mo_occ[:nocc])
            self.log.critical("p-DMET MO occupation numbers (virtual):\n%s", mo_occ[nocc:])
            raise RuntimeError("Degeneracy in MO occupation!")
        else:
            self.log.debug("p-DMET MO occupation numbers (occupied):\n%s", mo_occ[:nocc])
            self.log.debug("p-DMET MO occupation numbers (virtual):\n%s", mo_occ[nocc:])
        mo_coeff = np.dot(self._mo_orig, rot)
        return mo_coeff

class PDMET_UHF(PDMET_RHF):

    def get_rdm1(self):
        """DM1 in MO basis."""
        dm_type = self.dm_type
        if dm_type.startswith('demo'):
            dm1 = self.emb.make_rdm1_demo()
        elif dm_type.startswith('pwf-ccsd'):
            raise NotImplementedError("dm_type= %r" % dm_type)
            #dm1 = self.emb.make_rdm1_ccsd()
        else:
            raise NotImplementedError("dm_type= %r" % dm_type)
        # Check electron number
        nelec_err = abs(np.trace(dm1[0]+dm1[1]) - self.emb.mol.nelectron)
        if nelec_err > 1e-5:
            self.log.warning("Large electron error in 1DM= %.3e", nelec_err)
        # Check spin
        spin_err = abs(np.trace(dm1[0]-dm1[1]) - self.emb.mol.spin)
        if spin_err > 1e-5:
            self.log.warning("Large spin error in 1DM= %.3e", spin_err)
        return dm1

    def update_mo_coeff(self, mf, diis=None):
        dma, dmb = self.get_rdm1()
        mo_coeff = self.emb.mo_coeff
        ovlp = self.emb.get_ovlp()
        # Transform DM to original MO basis
        ra = dot(mo_coeff[0].T, ovlp, self._mo_orig[0])
        rb = dot(mo_coeff[1].T, ovlp, self._mo_orig[1])
        dma = dot(ra.T, dma, ra)
        dmb = dot(rb.T, dmb, rb)
        if diis is not None:
            assert dma.shape == dmb.shape
            dma, dmb = diis.update(np.asarray((dma, dmb)))
        mo_occ_a, rot_a = np.linalg.eigh(dma)
        mo_occ_b, rot_b = np.linalg.eigh(dmb)
        mo_occ_a, rot_a = mo_occ_a[::-1], rot_a[:,::-1]
        mo_occ_b, rot_b = mo_occ_b[::-1], rot_b[:,::-1]
        nocc_a = np.count_nonzero(mf.mo_occ[0] > 0)
        nocc_b = np.count_nonzero(mf.mo_occ[1] > 0)
        if min(abs(mo_occ_a[nocc_a-1] - mo_occ_a[nocc_a]),
               abs(mo_occ_b[nocc_b-1] - mo_occ_b[nocc_b])) < 1e-8:
            self.log.critical("p-DMET MO occupation numbers (alpha-occupied):\n%s", mo_occ_a[:nocc_a])
            self.log.critical("p-DMET MO occupation numbers (beta-occupied):\n%s", mo_occ_b[:nocc_b])
            self.log.critical("p-DMET MO occupation numbers (alpha-virtual):\n%s", mo_occ_a[nocc_a:])
            self.log.critical("p-DMET MO occupation numbers (beta-virtual):\n%s", mo_occ_b[nocc_b:])
            raise RuntimeError("Degeneracy in MO occupation!")
        else:
            self.log.debug("p-DMET MO occupation numbers (alpha-occupied):\n%s", mo_occ_a[:nocc_a])
            self.log.debug("p-DMET MO occupation numbers (beta-occupied):\n%s", mo_occ_b[:nocc_b])
            self.log.debug("p-DMET MO occupation numbers (alpha-virtual):\n%s", mo_occ_a[nocc_a:])
            self.log.debug("p-DMET MO occupation numbers (beta-virtual):\n%s", mo_occ_b[nocc_b:])
        mo_coeff_a = np.dot(self._mo_orig[0], rot_a)
        mo_coeff_b = np.dot(self._mo_orig[1], rot_b)
        return (mo_coeff_a, mo_coeff_b)


class Brueckner_RHF(SCMF):

    name = "Brueckner"

    def __init__(self, *args, diis_obj='dm1', **kwargs):
        super().__init__(*args, **kwargs)
        self.diis_obj = diis_obj.lower()

    def get_diis(self):
        """Get DIIS object."""
        return pyscf.lib.diis.DIIS()
        #PC-DIIS
        #nocc = np.count_nonzero(self.mf.mo_occ > 0)
        #diis = PCDIIS(self._mo_orig[:,:nocc].copy())
        #return diis

    def get_t1(self):
        """Get global T1 amplitudes from quantum embedding calculation."""
        return self.emb.get_t1()

    def update_mo_coeff(self, mf, diis=None):
        """Get new MO coefficients."""
        t1 = self.get_t1()
        self.log.debug("Norm of T1: L(2)= %.3e  L(inf)= %.3e", np.linalg.norm(t1), abs(t1).max())
        nocc, nvir = t1.shape
        nmo = (nocc + nvir)
        occ, vir = np.s_[:nocc], np.s_[nocc:]
        ovlp = self.emb.get_ovlp()
        # Perform DIIS in original MO basis, then transform back:
        if diis and self.diis_obj == 't1':
            ro = dot(mf.mo_coeff[:,occ].T, ovlp, self._mo_orig)
            rv = dot(mf.mo_coeff[:,vir].T, ovlp, self._mo_orig)
            t1 = dot(ro.T, t1, rv)
            t1 = diis.update(t1, xerr=t1)
            ## Transform back
            t1 = dot(ro, t1, rv.T)

        mo_change = (1-self.damping)*np.dot(mf.mo_coeff[:,vir], t1.T)
        self.log.debug("Change of occupied Brueckner orbitals= %.3e", np.linalg.norm(mo_change))
        bmo_occ = (mf.mo_coeff[:,occ] + mo_change)

        # Orthogonalize occupied orbitals
        # If there was no AO-overlap matrix:  bmo_occ = np.linalg.qr(bmo_occ)[0]
        dm_occ = np.dot(bmo_occ, bmo_occ.T)
        e, v = scipy.linalg.eigh(dm_occ, b=ovlp, type=2)
        bmo_occ = v[:,-nocc:]
        # DIIS of occupied density
        if diis and self.diis_obj == 'dm1':
            dm_occ = np.dot(bmo_occ, bmo_occ.T)
            r = np.dot(ovlp, self._mo_orig)
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

class Brueckner_UHF(Brueckner_RHF):

    def update_mo_coeff(self, mf, diis=None):
        t1a, t1b = self.get_t1()
        # ...
        raise NotImplementedError()
