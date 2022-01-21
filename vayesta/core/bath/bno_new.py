import numbers

import numpy as np

import pyscf
import pyscf.mp
import pyscf.pbc
import pyscf.pbc.mp

from vayesta.core.util import *
from vayesta.core.actspace import ActiveSpace
from vayesta.core.linalg import recursive_block_svd
from . import helper

from .bath import FragmentBath

class BNO_Threshold:

    def __init__(self, type, threshold):
        if type not in ('number', 'occupation', 'truncation', 'excited-percent', 'electron-percent'):
            raise ValueError()
        self.type = type
        self.threshold = threshold

    def __repr__(self):
        return "%s(type=%s, threshold=%g)" % (self.__class__.__name__, self.type, self.threshold)

    def get_number(self, bno_occup, electron_total=None):
        """Get number of BNOs."""
        nbno = len(bno_occup)
        if nbno == 0:
            return 0
        if self.type == 'number':
            return self.threshold
        if self.type in ('truncation', 'excited-percent', 'electron-percent'):
            npos = np.clip(bno_occup, 0.0, None)
            nexcited = np.sum(npos)
            if self.type == 'truncation':
                ntarget = (nexcited - self.threshold)
                nelec0 = 0
            elif self.type == 'electron-percent':
                assert (electron_total is not None)
                ntarget = (1.0-self.threshold) * electron_total
                nelec0 = (electron_total - nexcited)
            elif self.type == 'excited-percent':
                ntarget = (1.0-self.threshold) * nexcited
                nelec0 = 0
            #print("electron_total= %f nexcited= %f nelec0= %f, ntarget= %f" % (electron_total, nexcited, nelec0, ntarget))
            for bno_number in range(nbno+1):
                nelec = nelec0 + np.sum(npos[:bno_number])
                #print(bno_number, nelec)
                if nelec >= ntarget:
                    return bno_number
            raise RuntimeError()
        if self.type == 'occupation':
            return np.count_nonzero(bno_occup >= self.threshold)
        raise RuntimeError()

class BNO_Bath(FragmentBath):
    """Bath natural orbital (BNO) bath, requires DMET bath."""

    def __init__(self, fragment, ref_bath, *args, canonicalize=True, **kwargs):
        super().__init__(fragment, *args, **kwargs)
        self.ref_bath = ref_bath
        # Canonicalization can be set separately for occupied and virtual:
        if np.ndim(canonicalize) == 0:
            canonicalize = 2*[canonicalize]
        self.canonicalize = canonicalize
        # Results
        # Bath orbital coefficients:
        self.c_bno_occ = None
        self.c_bno_vir = None
        # Bath orbital natural occupation numbers:
        self.n_bno_occ = None
        self.n_bno_vir = None

    @property
    def c_cluster_occ(self):
        """Occupied DMET cluster orbitals."""
        return self.ref_bath.c_cluster_occ

    @property
    def c_cluster_vir(self):
        """Virtual DMET cluster orbitals."""
        return self.ref_bath.c_cluster_vir

    def kernel(self):
        """Make bath natural orbitals."""
        self.c_bno_occ, self.n_bno_occ = self.make_bno_bath('occ')
        self.c_bno_vir, self.n_bno_vir = self.make_bno_bath('vir')

    def make_bno_coeff(self, *args, **kwargs):
        raise AbstractMethodError()

    @property
    def dmet_bath(self):
        """The BNO bath can be build on top of a EwDMET bath. This returns the pure DMET bath in this case.

        Use two attributes here:
        Of the BNO bath is build on top of a EwDMET bath, self.dmet will be the EwDMET bath,
        but self.dmet_bath.dmet_bath will be the DMET bath only!
        """
        return self.ref_bath.dmet_bath

    def make_bno_bath(self, kind):
        if kind == 'occ':
            c_env = self.dmet_bath.c_env_occ
            name = "occupied"
        elif kind == 'vir':
            c_env = self.dmet_bath.c_env_vir
            name = "virtual"
        else:
            raise ValueError("kind not in ['occ', 'vir']: %r" % kind)

        if self.spin_restricted and (c_env.shape[-1] == 0):
            return c_env, np.zeros(0)
        if self.spin_unrestricted and (c_env[0].shape[-1] + c_env[1].shape[-1] == 0):
            return c_env, tuple(2*[np.zeros(0)])

        self.log.info("Making %s Bath NOs", name.capitalize())
        self.log.info("-------%s---------", len(name)*'-')
        self.log.changeIndentLevel(1)
        with log_time(self.log.timing, "Time for %s BNOs: %s", name):
            c_bno, n_bno = self.make_bno_coeff(kind)
        self.log_histogram(n_bno, name=name)
        self.log.changeIndentLevel(-1)

        return c_bno, n_bno

    def log_histogram(self, n_bno, name):
        if len(n_bno) == 0:
            return
        if np.ndim(n_bno[0]) == 1:
            self.log_histogram(n_bno[0], name='Alpha-%s' % name)
            self.log_histogram(n_bno[1], name='Beta-%s' % name)
            return
        self.log.info("%s BNO histogram", name.capitalize())
        self.log.info("%s--------------", len(name)*'-')
        for line in helper.plot_histogram(n_bno):
            self.log.info(line)

    def get_occupied_bath(self, bno_threshold=None, **kwargs):
        return self.truncate_bno(self.c_bno_occ, self.n_bno_occ, bno_threshold=bno_threshold,
                header="occupied BNOs:", **kwargs)

    def get_virtual_bath(self, bno_threshold=None, bno_number=None, **kwargs):
        return self.truncate_bno(self.c_bno_vir, self.n_bno_vir, bno_threshold=bno_threshold,
                header="virtual BNOs:", **kwargs)

    def truncate_bno(self, c_bno, n_bno, bno_threshold=None, header=None, verbose=True):
        """Split natural orbitals (NO) into bath and rest."""

        # For UHF, call recursively:
        if np.ndim(c_bno[0]) == 2:
            c_bno_a, c_rest_a = self.truncate_bno(c_bno[0], n_bno[0], bno_threshold=bno_threshold,
                    header='Alpha-%s' % header, verbose=verbose)
            c_bno_b, c_rest_b = self.truncate_bno(c_bno[1], n_bno[1], bno_threshold=bno_threshold,
                    header='Beta-%s' % header, verbose=verbose)
            return (c_bno_a, c_bno_b), (c_rest_a, c_rest_b)

        if isinstance(bno_threshold, numbers.Number):
            bno_threshold = BNO_Threshold('occupation', bno_threshold)
        nelec_cluster = self.dmet_bath.get_cluster_electrons()
        bno_number = bno_threshold.get_number(n_bno, electron_total=nelec_cluster)

        # Logging
        if verbose:
            if header:
                self.log.info(header[0].upper() + header[1:])
            fmt = "  %4s: N= %4d  max= % 9.3g  min= % 9.3g  sum= % 9.3g ( %7.3f %%)"
            def log_space(name, n_part):
                if len(n_part) == 0:
                    self.log.info(fmt[:fmt.index('max')].rstrip(), name, 0)
                    return
                with np.errstate(invalid='ignore'): # supress 0/0 warning
                    self.log.info(fmt, name, len(n_part), max(n_part), min(n_part), np.sum(n_part),
                            100*np.sum(n_part)/np.sum(n_bno))
            log_space("Bath", n_bno[:bno_number])
            log_space("Rest", n_bno[bno_number:])

        c_bno, c_rest = np.hsplit(c_bno, [bno_number])
        return c_bno, c_rest

    def get_active_space(self, kind):
        ref_bath = self.ref_bath
        dmet_bath = self.dmet_bath
        nao = self.mol.nao
        zero_space = np.zeros((nao, 0)) if self.spin_restricted else np.zeros((2, nao, 0))
        if kind == 'occ':
            c_active_occ = stack_mo(dmet_bath.c_cluster_occ, dmet_bath.c_env_occ)
            c_frozen_occ = zero_space
            c_active_vir = ref_bath.c_cluster_vir
            c_frozen_vir = ref_bath.c_env_vir
        elif kind == 'vir':
            c_active_occ = ref_bath.c_cluster_occ
            c_frozen_occ = ref_bath.c_env_occ
            c_active_vir = stack_mo(dmet_bath.c_cluster_vir, dmet_bath.c_env_vir)
            c_frozen_vir = zero_space
        else:
            raise ValueError("Unknown kind: %r" % kind)
        actspace = ActiveSpace(self.mf, c_active_occ, c_active_vir, c_frozen_occ, c_frozen_vir)
        return actspace

    def _undo_canonicalization(self, dm, rot):
        self.log.debugv("Undoing canonicalization")
        return dot(rot, dm, rot.T)

    def _dm_take_env(self, dm, kind):
        if kind == 'occ':
            ncluster = self.dmet_bath.c_cluster_occ.shape[-1]
        elif kind == 'vir':
            ncluster = self.dmet_bath.c_cluster_vir.shape[-1]
        self.log.debugv("n(cluster)= %d", ncluster)
        self.log.debugv("tr(D)= %g", np.trace(dm))
        dm = dm[ncluster:,ncluster:]
        self.log.debugv("tr(D[env,env])= %g", np.trace(dm))
        return dm

    def _diagonalize_dm(self, dm):
        sort = np.s_[::-1]
        n_bno, r_bno = np.linalg.eigh(dm)
        n_bno = n_bno[sort]
        r_bno = r_bno[:,sort]
        return r_bno, n_bno


class BNO_Bath_UHF(BNO_Bath):

    def _undo_canonicalization(self, dm, rot):
        self.log.debugv("Undoing canonicalization")
        return (dot(rot[0], dm[0], rot[0].T),
                dot(rot[1], dm[1], rot[1].T))

    def _dm_take_env(self, dm, kind):
        if kind == 'occ':
            ncluster = (self.dmet_bath.c_cluster_occ[0].shape[-1], self.dmet_bath.c_cluster_occ[1].shape[-1])
        elif kind == 'vir':
            ncluster = (self.dmet_bath.c_cluster_vir[0].shape[-1], self.dmet_bath.c_cluster_vir[1].shape[-1])
        self.log.debugv("n(cluster)= (%d, %d)", ncluster[0], ncluster[1])
        self.log.debugv("tr(alpha-D)= %g", np.trace(dm[0]))
        self.log.debugv("tr( beta-D)= %g", np.trace(dm[1]))
        dm = (dm[0][ncluster[0]:,ncluster[0]:], dm[1][ncluster[1]:,ncluster[1]:])
        self.log.debugv("tr(alpha-D[env,env])= %g", np.trace(dm[0]))
        self.log.debugv("tr( beta-D[env,env])= %g", np.trace(dm[1]))
        return dm

    def _diagonalize_dm(self, dm):
        r_bno_a, n_bno_a = super()._diagonalize_dm(dm[0])
        r_bno_b, n_bno_b = super()._diagonalize_dm(dm[1])
        return (r_bno_a, r_bno_b), (n_bno_a, n_bno_b)


class MP2_BNO_Bath(BNO_Bath):

    def __init__(self, *args, project_t2=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.project_t2 = project_t2

    def _make_t2(self, mo_energy, eris=None, cderi=None, cderi_neg=None, blksize=None):
        """Make T2 amplitudes"""
        # (ov|ov)
        if eris is not None:
            self.log.debugv("Making T2 amplitudes from ERIs")
            assert (eris.ndim == 4)
            nocc, nvir = eris.shape[:2]
        # (L|ov)
        elif cderi is not None:
            self.log.debugv("Making T2 amplitudes from CD-ERIs")
            assert (cderi.ndim == 3)
            assert (cderi_neg is None or cderi_neg.ndim == 3)
            nocc, nvir = cderi.shape[1:]
        else:
            raise ValueError()

        t2 = np.empty((nocc, nocc, nvir, nvir))
        eia = (mo_energy[:nocc,None] - mo_energy[None,nocc:])
        if blksize is None:
            blksize = int(1e9 / max(nocc*nvir*nvir * 8, 1))
        for blk in brange(0, nocc, blksize):
            if eris is not None:
                gijab = eris[blk].transpose(0,2,1,3)
            else:
                gijab = einsum('Lia,Ljb->ijab', cderi[:,blk], cderi)
                if cderi_neg is not None:
                    gijab -= einsum('Lia,Ljb->ijab', cderi_neg[:,blk], cderi_neg)
            eijab = (eia[blk][:,None,:,None] + eia[None,:,None,:])
            t2[blk] = (gijab / eijab)
        return t2

    def _get_mo_energy(self, fock, actspace):
        c_act = actspace.c_active
        mo_energy = einsum('ai,ab,bi->i', c_act, fock, c_act)
        return mo_energy

    def _get_eris(self, actspace):
        # We only need the (ov|ov) block for MP2:
        mo_coeff = 2*[actspace.c_active_occ, actspace.c_active_vir]
        eris = self.base.get_eris_array(mo_coeff)
        return eris

    def _get_cderi(self, actspace):
        # We only need the (L|ov) block for MP2:
        mo_coeff = (actspace.c_active_occ, actspace.c_active_vir)
        cderi, cderi_neg = self.base.get_cderi(mo_coeff)
        return cderi, cderi_neg

    def make_delta_dm1_old(self, kind, t2, t2loc):
        """Delta MP2 density matrix"""
        if self.local_dm is False:
            self.log.debug("Constructing DM from full T2-amplitudes.")
            t2l = t2r = t2
            # This is equivalent to:
            # do, dv = pyscf.mp.mp2._gamma1_intermediates(mp2, eris=eris)
            # do, dv = -2*do, 2*dv
        elif self.local_dm is True:
            # LOCAL DM IS NOT RECOMMENDED - USE SEMI OR FALSE
            self.log.warning("Using local_dm=True is not recommended - use 'semi' or False")
            self.log.debug("Constructing DM from local T2-amplitudes.")
            t2l = t2r = t2loc
        elif self.local_dm == "semi":
            self.log.debug("Constructing DM from semi-local T2-amplitudes.")
            t2l, t2r = t2loc, t2
        else:
            raise ValueError("Unknown value for local_dm: %r" % self.local_dm)

        #norm = 2
        norm = 1
        if kind == 'occ':
            dm = norm*(2*einsum('ikab,jkab->ij', t2l, t2r)
                       - einsum('ikab,kjab->ij', t2l, t2r))
            # Note that this is equivalent to:
            #dm = 2*(2*einsum("kiba,kjba->ij", t2l, t2r)
            #        - einsum("kiba,kjab->ij", t2l, t2r))
        else:
            dm = norm*(2*einsum('ijac,ijbc->ab', t2l, t2r)
                       - einsum('ijac,ijcb->ab', t2l, t2r))
        if self.local_dm == 'semi':
            dm = (dm + dm.T)/2
        assert np.allclose(dm, dm.T)
        return dm

    def make_delta_dm1(self, kind, t2, actspace):
        """Delta MP2 density matrix"""
        norm = 1
        if not self.project_t2:
            self.log.debug("Constructing DM from full T2-amplitudes.")
            # This is equivalent to:
            # do, dv = pyscf.mp.mp2._gamma1_intermediates(mp2, eris=eris)
            # do, dv = -2*do, 2*dv
            if kind == 'occ':
                dm = norm*(2*einsum('ikab,jkab->ij', t2, t2)
                           - einsum('ikab,kjab->ij', t2, t2))
            else:
                dm = norm*(2*einsum('ijac,ijbc->ab', t2, t2)
                           - einsum('ijac,ijcb->ab', t2, t2))
            assert np.allclose(dm, dm.T)
            return dm

        # Project one T-amplitude onto fragment
        self.log.debug("Constructing DM from projected T2-amplitudes.")
        ovlp = self.fragment.base.get_ovlp()
        if kind == 'occ':
            px = dot(actspace.c_active_vir.T, ovlp, self.dmet_bath.c_cluster_vir)
            t2x = einsum('ax,ijab->ijxb', px, t2)
            dm = norm*(2*einsum('ikab,jkab->ij', t2x, t2x)
                       - einsum('ikab,kjab->ij', t2x, t2x)
                     + 2*einsum('kiba,kjba->ij', t2x, t2x)
                       - einsum('kiba,jkba->ij', t2x, t2x))/2
        else:
            px = dot(actspace.c_active_occ.T, ovlp, self.dmet_bath.c_cluster_occ)
            t2x = einsum('ix,ijab->xjab', px, t2)
            dm = norm*(2*einsum('ijac,ijbc->ab', t2x, t2x)
                       - einsum('ijac,ijcb->ab', t2x, t2x)
                     + 2*einsum('jica,jicb->ab', t2x, t2x)
                       - einsum('jica,jibc->ab', t2x, t2x))/2

        assert np.allclose(dm, dm.T)
        return dm

    def make_bno_coeff(self, kind, eris=None):
        """Construct MP2 bath natural orbital coefficients and occupation numbers.

        This routine works for both for spin-restricted and unrestricted.

        Parameters
        ----------
        kind: ['occ', 'vir']
        eris: mp2._ChemistERIs

        Returns
        -------
        c_bno: (n(AO), n(BNO)) array
            Bath natural orbital coefficients.
        n_bno: (n(BNO)) array
            Bath natural orbital occupation numbers.
        """

        actspace_orig = self.get_active_space(kind)
        fock = self.fragment.base.get_fock_for_bath()

        # --- Canonicalization [optional]
        if self.canonicalize[0]:
            self.log.debugv("Canonicalizing occupied orbitals")
            c_active_occ, r_occ = self.fragment.canonicalize_mo(actspace_orig.c_active_occ, fock=fock)
        else:
            c_active_occ = actspace_orig.c_active_occ
            r_occ = None
        if self.canonicalize[1]:
            self.log.debugv("Canonicalizing virtual orbitals")
            c_active_vir, r_vir = self.fragment.canonicalize_mo(actspace_orig.c_active_vir, fock=fock)
        else:
            c_active_vir = actspace_orig.c_active_vir
            r_vir = None
        actspace = ActiveSpace(self.mf, c_active_occ, c_active_vir,
                actspace_orig.c_frozen_occ, actspace_orig.c_frozen_vir)

        # -- Integral transformation
        if eris is None:
            eris = cderi = cderi_neg = None
            with log_time(self.log.timing, "Time for AO->MO transformation: %s"):
                if self.fragment.base.has_df:
                    cderi, cderi_neg = self._get_cderi(actspace)
                else:
                    eris = self._get_eris(actspace)
        # Reuse previously obtained integral transformation into N^2 sized quantity (rather than N^4)
        #else:
        #    self.log.debug("Transforming previous eris.")
        #    eris = transform_mp2_eris(eris, actspace.c_active_occ, actspace.c_active_vir, ovlp=self.base.get_ovlp())
        # TODO: DF-MP2
        #assert (eris.ovov is not None)
        #nocc = actspace.nocc_active
        #nvir = actspace.nvir_active

        mo_energy = self._get_mo_energy(fock, actspace)
        with log_time(self.log.timing, "Time for MP2 T-amplitudes: %s"):
            t2 = self._make_t2(mo_energy, eris=eris, cderi=cderi, cderi_neg=cderi_neg)

        dm = self.make_delta_dm1(kind, t2, actspace)

        # --- Undo canonicalization
        if kind == 'occ' and r_occ is not None:
            c_env = self.dmet_bath.c_env_occ
            dm = self._undo_canonicalization(dm, r_occ)
        elif kind == 'vir' and r_vir is not None:
            c_env = self.dmet_bath.c_env_vir
            dm = self._undo_canonicalization(dm, r_vir)
        # --- Diagonalize environment-environment block
        dm = self._dm_take_env(dm, kind)
        r_bno, n_bno = self._diagonalize_dm(dm)
        c_bno = dot_s(c_env, r_bno)
        c_bno = fix_orbital_sign(c_bno)[0]

        return c_bno, n_bno


class UMP2_BNO_Bath(MP2_BNO_Bath, BNO_Bath_UHF):

    def _get_mo_energy(self, fock, actspace):
        c_act_a, c_act_b = actspace.c_active
        mo_energy_a = einsum('ai,ab,bi->i', c_act_a, fock[0], c_act_a)
        mo_energy_b = einsum('ai,ab,bi->i', c_act_b, fock[1], c_act_b)
        return (mo_energy_a, mo_energy_b)

    def _get_eris(self, actspace):
        # We only need the (ov|ov) block for MP2:
        mo_ov_a = [actspace.c_active_occ[0], actspace.c_active_vir[0]]
        mo_ov_b = [actspace.c_active_occ[1], actspace.c_active_vir[1]]
        mo_aa = mo_ov_a + mo_ov_a
        mo_ab = mo_ov_a + mo_ov_b
        mo_bb = mo_ov_b + mo_ov_b
        eris_aa = self.base.get_eris_array(mo_aa)
        eris_ab = self.base.get_eris_array(mo_ab)
        eris_bb = self.base.get_eris_array(mo_bb)
        return (eris_aa, eris_ab, eris_bb)

    def _get_cderi(self, actspace):
        # We only need the (ov|ov) block for MP2:
        mo_a = [actspace.c_active_occ[0], actspace.c_active_vir[0]]
        mo_b = [actspace.c_active_occ[1], actspace.c_active_vir[1]]
        cderi_a, cderi_neg_a = self.base.get_cderi(mo_a)
        cderi_b, cderi_neg_b = self.base.get_cderi(mo_b)
        return (cderi_a, cderi_b), (cderi_neg_a, cderi_neg_b)

    def _make_t2(self, mo_energy, eris=None, cderi=None, cderi_neg=None, blksize=None, workmem=int(1e9)):
        """Make T2 amplitudes"""
        # (ov|ov)
        if eris is not None:
            assert len(eris) == 3
            assert (eris[0].ndim == 4)
            assert (eris[1].ndim == 4)
            assert (eris[2].ndim == 4)
            nocca, nvira = eris[0].shape[:2]
            noccb, nvirb = eris[2].shape[:2]
        # (L|ov)
        elif cderi is not None:
            assert len(cderi) == 2
            assert (cderi[0].ndim == 3)
            assert (cderi[1].ndim == 3)
            nocca, nvira = cderi[0].shape[1:]
            noccb, nvirb = cderi[1].shape[1:]
        else:
            raise ValueError()

        t2aa = np.empty((nocca, nocca, nvira, nvira))
        t2ab = np.empty((nocca, noccb, nvira, nvirb))
        t2bb = np.empty((noccb, noccb, nvirb, nvirb))
        eia_a = (mo_energy[0][:nocca,None] - mo_energy[0][None,nocca:])
        eia_b = (mo_energy[1][:noccb,None] - mo_energy[1][None,noccb:])

        # Alpha-alpha and Alpha-beta:
        if blksize is None:
            blksize_a = int(workmem / max(nocca*nvira*nvira * 8, 1))
        else:
            blksize_a = blksize
        for blk in brange(0, nocca, blksize_a):
            # Alpha-alpha
            if eris is not None:
                gijab = eris[0][blk].transpose(0,2,1,3)
            else:
                gijab = einsum('Lia,Ljb->ijab', cderi[0][:,blk], cderi[0])
                if cderi_neg[0] is not None:
                    gijab -= einsum('Lia,Ljb->ijab', cderi_neg[0][:,blk], cderi_neg[0])
            eijab = (eia_a[blk][:,None,:,None] + eia_a[None,:,None,:])
            t2aa[blk] = (gijab / eijab)
            t2aa[blk] -= t2aa[blk].transpose(0,1,3,2)
            # Alpha-beta
            if eris is not None:
                gijab = eris[1][blk].transpose(0,2,1,3)
            else:
                gijab = einsum('Lia,Ljb->ijab', cderi[0][:,blk], cderi[1])
                if cderi_neg[0] is not None:
                    gijab -= einsum('Lia,Ljb->ijab', cderi_neg[0][:,blk], cderi_neg[1])
            eijab = (eia_a[blk][:,None,:,None] + eia_b[None,:,None,:])
            t2ab[blk] = (gijab / eijab)
        # Beta-beta:
        if blksize is None:
            blksize_b = int(workmem / max(noccb*nvirb*nvirb * 8, 1))
        else:
            blksize_b = blksize
        for blk in brange(0, noccb, blksize_b):
            if eris is not None:
                gijab = eris[2][blk].transpose(0,2,1,3)
            else:
                gijab = einsum('Lia,Ljb->ijab', cderi[1][:,blk], cderi[1])
                if cderi_neg[0] is not None:
                    gijab -= einsum('Lia,Ljb->ijab', cderi_neg[1][:,blk], cderi_neg[1])
            eijab = (eia_b[blk][:,None,:,None] + eia_b[None,:,None,:])
            t2bb[blk] = (gijab / eijab)
            t2bb[blk] -= t2bb[blk].transpose(0,1,3,2)

        return (t2aa, t2ab, t2bb)

    def make_delta_dm1(self, kind, t2, actspace):
        taa, tab, tbb = t2
        if not self.project_t2:
            # Construct occupied-occupied DM
            if kind == 'occ':
                dma  = (einsum('imef,jmef->ij', taa.conj(), taa)/2
                      + einsum('imef,jmef->ij', tab.conj(), tab))
                dmb  = (einsum('imef,jmef->ij', tbb.conj(), tbb)/2
                      + einsum('mief,mjef->ij', tab.conj(), tab))
            # Construct virtual-virtual DM
            elif kind == 'vir':
                dma  = (einsum('mnae,mnbe->ba', taa.conj(), taa)/2
                      + einsum('mnae,mnbe->ba', tab.conj(), tab))
                dmb  = (einsum('mnae,mnbe->ba', tbb.conj(), tbb)/2
                      + einsum('mnea,mneb->ba', tab.conj(), tab))
        else:
            raise NotImplementedError()
        assert np.allclose(dma, dma.T)
        assert np.allclose(dmb, dmb.T)
        return (dma, dmb)


# ================================================================================================ #

#    if self.opts.plot_orbitals:
#        #bins = np.hstack((-np.inf, np.self.logspace(-9, -3, 9-3+1), np.inf))
#        bins = np.hstack((1, np.self.logspace(-3, -9, 9-3+1), -1))
#        for idx, upper in enumerate(bins[:-1]):
#            lower = bins[idx+1]
#            mask = np.self.logical_and((dm_occ > lower), (dm_occ <= upper))
#            if np.any(mask):
#                coeff = c_rot[:,mask]
#                self.log.info("Plotting MP2 bath density between %.0e and %.0e containing %d orbitals." % (upper, lower, coeff.shape[-1]))
#                dm = np.dot(coeff, coeff.T)
#                dset_idx = (4001 if kind == "occ" else 5001) + idx
#                self.cubefile.add_density(dm, dset_idx=dset_idx)
