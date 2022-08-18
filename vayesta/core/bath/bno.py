import numbers
import numpy as np
from vayesta.core.util import *
from vayesta.core import spinalg
from vayesta.core.types import Cluster
from vayesta.core.linalg import recursive_block_svd
from . import helper
from .bath import Bath


class BNO_Threshold:

    def __init__(self, type, threshold):
        """
        number:             Fixed number of BNOs
        occupation:         Occupation thresehold for BNOs ("eta")
        truncation:         Maximum number of electrons to be ignored
        electron-percent:   Add BNOs until 100-x% of the total number of all electrons is captured
        excited-percent:    Add BNOs until 100-x% of the total number of excited electrons is captured
        """
        if type not in ('number', 'occupation', 'truncation', 'electron-percent', 'excited-percent'):
            raise ValueError()
        self.type = type
        self.threshold = threshold

    def __repr__(self):
        return "%s(type=%s, threshold=%g)" % (self.__class__.__name__, self.type, self.threshold)

    def get_number(self, bno_occup, electron_total=None):
        """Get number of BNOs."""
        nbno = len(bno_occup)
        if (nbno == 0):
            return 0
        if (self.type == 'number'):
            return self.threshold
        if (self.type in ('truncation', 'electron-percent', 'excited-percent')):
            npos = np.clip(bno_occup, 0.0, None)
            nexcited = np.sum(npos)
            nelec0 = 0
            if self.type == 'truncation':
                ntarget = (nexcited - self.threshold)
            elif self.type == 'electron-percent':
                assert (electron_total is not None)
                ntarget = (1.0-self.threshold) * electron_total
                nelec0 = (electron_total - nexcited)
            elif self.type == 'excited-percent':
                ntarget = (1.0-self.threshold) * nexcited
            for bno_number in range(nbno+1):
                nelec = (nelec0 + np.sum(npos[:bno_number]))
                if nelec >= ntarget:
                    return bno_number
            raise RuntimeError()
        if (self.type == 'occupation'):
            return np.count_nonzero(bno_occup >= self.threshold)
        raise RuntimeError()


class BNO_Bath(Bath):
    """Bath natural orbital (BNO) bath, requires DMET bath."""

    def __init__(self, fragment, dmet_bath, occtype, *args, c_buffer=None, canonicalize=True, **kwargs):
        super().__init__(fragment, *args, **kwargs)
        self.dmet_bath = dmet_bath
        if occtype not in ('occupied', 'virtual'):
            raise ValueError("Invalid occtype: %s" % occtype)
        self.occtype = occtype
        self.c_buffer = c_buffer
        # Canonicalization can be set separately for occupied and virtual:
        if np.ndim(canonicalize) == 0:
            canonicalize = (canonicalize, canonicalize)
        self.canonicalize = canonicalize
        # Coefficients and occupations:
        self.coeff, self.occup = self.kernel()

    @property
    def c_cluster_occ(self):
        """Occupied DMET cluster orbitals."""
        return self.dmet_bath.c_cluster_occ

    @property
    def c_cluster_vir(self):
        """Virtual DMET cluster orbitals."""
        return self.dmet_bath.c_cluster_vir

    def make_bno_coeff(self, *args, **kwargs):
        raise AbstractMethodError()

    @property
    def c_env(self):
        if self.occtype == 'occupied':
            return self.dmet_bath.c_env_occ
        if self.occtype == 'virtual':
            return self.dmet_bath.c_env_vir

    @property
    def ncluster(self):
        if self.occtype == 'occupied':
            return self.dmet_bath.c_cluster_occ.shape[-1]
        if self.occtype == 'virtual':
            return  self.dmet_bath.c_cluster_vir.shape[-1]

    def kernel(self):
        c_env = self.c_env
        if self.spin_restricted and (c_env.shape[-1] == 0):
            return c_env, np.zeros(0)
        if self.spin_unrestricted and (c_env[0].shape[-1] + c_env[1].shape[-1] == 0):
            return c_env, tuple(2*[np.zeros(0)])
        self.log.info("Making %s BNOs", self.occtype.capitalize())
        self.log.info("-------%s-----", len(self.occtype)*'-')
        self.log.changeIndentLevel(1)
        coeff, occup = self.make_bno_coeff()
        self.log_histogram(occup)
        self.log.changeIndentLevel(-1)
        self.coeff = coeff
        self.occup = occup
        return coeff, occup

    def log_histogram(self, n_bno):
        if len(n_bno) == 0:
            return
        self.log.info("%s BNO histogram:", self.occtype.capitalize())
        bins = np.hstack([-np.inf, np.logspace(-3, -10, 8)[::-1], np.inf])
        labels = '    ' + ''.join('{:{w}}'.format('E-%d' % d, w=5) for d in range(3, 11))
        self.log.info(helper.make_histogram(n_bno, bins=bins, labels=labels))

    def get_bath(self, bno_threshold=None, **kwargs):
        return self.truncate_bno(self.coeff, self.occup, bno_threshold=bno_threshold, **kwargs)

    def truncate_bno(self, coeff, occup, bno_threshold=None, verbose=True):
        """Split natural orbitals (NO) into bath and rest."""

        header = '%s BNOs:' % self.occtype

        if isinstance(bno_threshold, numbers.Number):
            bno_threshold = BNO_Threshold('occupation', bno_threshold)
        nelec_cluster = self.dmet_bath.get_cluster_electrons()
        bno_number = bno_threshold.get_number(occup, electron_total=nelec_cluster)

        # Logging
        if verbose:
            if header:
                self.log.info(header.capitalize())
            fmt = "  %4s: N= %4d  max= % 9.3g  min= % 9.3g  sum= % 9.3g ( %7.3f %%)"
            def log_space(name, n_part):
                if len(n_part) == 0:
                    self.log.info(fmt[:fmt.index('max')].rstrip(), name, 0)
                    return
                with np.errstate(invalid='ignore'): # supress 0/0 warning
                    self.log.info(fmt, name, len(n_part), max(n_part), min(n_part), np.sum(n_part),
                            100*np.sum(n_part)/np.sum(occup))
            log_space("Bath", occup[:bno_number])
            log_space("Rest", occup[bno_number:])

        c_bath, c_rest = np.hsplit(coeff, [bno_number])
        return c_bath, c_rest

    def get_active_space(self):
        dmet_bath = self.dmet_bath
        nao = self.mol.nao
        empty = np.zeros((nao, 0)) if self.spin_restricted else np.zeros((2, nao, 0))
        if self.occtype == 'occupied':
            c_active_occ = spinalg.hstack_matrices(dmet_bath.c_cluster_occ, dmet_bath.c_env_occ)
            c_frozen_occ = empty
            if self.c_buffer is not None:
                raise NotImplementedError
            c_active_vir = dmet_bath.c_cluster_vir
            c_frozen_vir = dmet_bath.c_env_vir
        elif self.occtype == 'virtual':
            if self.c_buffer is None:
                c_active_occ = dmet_bath.c_cluster_occ
                c_frozen_occ = dmet_bath.c_env_occ
            else:
                c_active_occ = spinalg.hstack_matrices(dmet_bath.c_cluster_occ, self.c_buffer)
                ovlp = self.fragment.base.get_ovlp()
                r = dot(self.c_buffer.T, ovlp, dmet_bath.c_env_occ)
                dm_frozen = np.eye(dmet_bath.c_env_occ.shape[-1]) - np.dot(r.T, r)
                e, r = np.linalg.eigh(dm_frozen)
                c_frozen_occ = np.dot(dmet_bath.c_env_occ, r[:,e>0.5])

            c_active_vir = spinalg.hstack_matrices(dmet_bath.c_cluster_vir, dmet_bath.c_env_vir)
            c_frozen_vir = empty
        actspace = Cluster.from_coeffs(c_active_occ, c_active_vir, c_frozen_occ, c_frozen_vir)
        return actspace

    def _undo_canonicalization(self, dm, rot):
        self.log.debugv("Undoing canonicalization")
        return dot(rot, dm, rot.T)

    def _dm_take_env(self, dm):
        ncluster = self.ncluster
        self.log.debugv("n(cluster)= %d", ncluster)
        self.log.debugv("tr(D)= %g", np.trace(dm))
        dm = dm[ncluster:,ncluster:]
        self.log.debugv("tr(D[env,env])= %g", np.trace(dm))
        return dm

    def _diagonalize_dm(self, dm):
        n_bno, r_bno = np.linalg.eigh(dm)
        sort = np.s_[::-1]
        n_bno = n_bno[sort]
        r_bno = r_bno[:,sort]
        return r_bno, n_bno


class BNO_Bath_UHF(BNO_Bath):

    def _undo_canonicalization(self, dm, rot):
        self.log.debugv("Undoing canonicalization")
        return (dot(rot[0], dm[0], rot[0].T),
                dot(rot[1], dm[1], rot[1].T))

    @property
    def ncluster(self):
        if self.occtype == 'occupied':
            return (self.dmet_bath.c_cluster_occ[0].shape[-1], self.dmet_bath.c_cluster_occ[1].shape[-1])
        if self.occtype == 'virtual':
            return (self.dmet_bath.c_cluster_vir[0].shape[-1], self.dmet_bath.c_cluster_vir[1].shape[-1])

    def _dm_take_env(self, dm):
        ncluster = self.ncluster
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

    def log_histogram(self, n_bno):
        if len(n_bno[0]) == len(n_bno[0]) == 0:
            return
        self.log.info("%s BNO histogram (alpha/beta):", self.occtype.capitalize())
        bins = np.hstack([-np.inf, np.logspace(-3, -10, 8)[::-1], np.inf])
        labels = '    ' + ''.join('{:{w}}'.format('E-%d' % d, w=5) for d in range(3, 11))
        ha = helper.make_histogram(n_bno[0], bins=bins, labels=labels, rstrip=False).split('\n')
        hb = helper.make_histogram(n_bno[1], bins=bins, labels=labels).split('\n')
        for i in range(len(ha)):
            self.log.info(ha[i] + '   ' + hb[i])

    def truncate_bno(self, coeff, occup, *args, **kwargs):
        c_bath_a, c_rest_a = super().truncate_bno(coeff[0], occup[0], *args, **kwargs)
        c_bath_b, c_rest_b = super().truncate_bno(coeff[1], occup[1], *args, **kwargs)
        return (c_bath_a, c_bath_b), (c_rest_a, c_rest_b)


class MP2_BNO_Bath(BNO_Bath):

    def __init__(self, *args, project_t2=False, **kwargs):
        self.project_t2 = project_t2
        super().__init__(*args, **kwargs)

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

    def make_delta_dm1(self, t2, actspace):
        """Delta MP2 density matrix"""

        if self.project_t2:
            self.log.debugv("Constructing DM from projected amplitudes (method= %s).", self.project_t2)
            eig = self.dmet_bath.n_dmet
            assert np.all(eig > -1e-10)
            assert np.all(eig-1 < 1e-10)
            eig = np.clip(eig, 0, 1)
            if self.project_t2 == 'fragment':
                weights = len(eig)*[0]
            elif self.project_t2 == 'full':
                weights = len(eig)*[1]
            elif self.project_t2 == 'linear':
                weights = 2*abs(np.fmin(eig, 1-eig))
            elif self.project_t2 == 'entropy':
                weights = 4*eig*(1-eig)
            elif self.project_t2 == 'sqrt-entropy':
                weights = 2*np.sqrt(eig*(1-eig))
            else:
                raise ValueError("Unknown value for project_t2: %s" % self.project_t2)
            assert np.all(weights > -1e-14)
            assert np.all(weights-1 < 1e-14)
            weights = hstack(self.fragment.n_frag*[1], weights)

            # Project and symmetrize:
            ovlp = self.fragment.base.get_ovlp()
            c_fragdmet = hstack(self.fragment.c_frag, self.dmet_bath.c_dmet)
            if self.occtype == 'occupied':
                rot = dot(actspace.c_active_vir.T, ovlp, c_fragdmet)
                proj = einsum('ix,x,jx->ij', rot, weights, rot)
                t2 = einsum('xa,ijab->ijxb', proj, t2)
            elif self.occtype == 'virtual':
                rot = dot(actspace.c_active_occ.T, ovlp, c_fragdmet)
                proj = einsum('ix,x,jx->ij', rot, weights, rot)
                t2 = einsum('xi,i...->x...', proj, t2)
            t2 = (t2 + t2.transpose(1,0,3,2))/2
        else:
            self.log.debugv("Constructing DM from complete amplitudes")

        # This is equivalent to:
        # do, dv = pyscf.mp.mp2._gamma1_intermediates(mp2, eris=eris)
        # do, dv = -2*do, 2*dv
        if self.occtype == 'occupied':
            dm = (2*einsum('ikab,jkab->ij', t2, t2)
                  - einsum('ikab,jkba->ij', t2, t2))
        elif self.occtype == 'virtual':
            dm = (2*einsum('ijac,ijbc->ab', t2, t2)
                  - einsum('ijac,ijcb->ab', t2, t2))
        assert np.allclose(dm, dm.T)
        return dm

    def make_bno_coeff(self, eris=None):
        """Construct MP2 bath natural orbital coefficients and occupation numbers.

        This routine works for both for spin-restricted and unrestricted.

        Parameters
        ----------
        eris: mp2._ChemistERIs

        Returns
        -------
        c_bno: (n(AO), n(BNO)) array
            Bath natural orbital coefficients.
        n_bno: (n(BNO)) array
            Bath natural orbital occupation numbers.
        """
        t_init = timer()

        actspace_orig = self.get_active_space()
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
        actspace = Cluster.from_coeffs(c_active_occ, c_active_vir,
                actspace_orig.c_frozen_occ, actspace_orig.c_frozen_vir)

        # -- Integral transformation
        if eris is None:
            eris = cderi = cderi_neg = None
            t0 = timer()
            if self.fragment.base.has_df:
                cderi, cderi_neg = self._get_cderi(actspace)
            else:
                eris = self._get_eris(actspace)
            t_ao2mo = timer()-t0
        # Reuse previously obtained integral transformation into N^2 sized quantity (rather than N^4)
        #else:
        #    self.log.debug("Transforming previous eris.")
        #    eris = transform_mp2_eris(eris, actspace.c_active_occ, actspace.c_active_vir, ovlp=self.base.get_ovlp())
        # TODO: DF-MP2
        #assert (eris.ovov is not None)
        #nocc = actspace.nocc_active
        #nvir = actspace.nvir_active

        mo_energy = self._get_mo_energy(fock, actspace)
        t0 = timer()
        t2 = self._make_t2(mo_energy, eris=eris, cderi=cderi, cderi_neg=cderi_neg)
        t_amps = timer()-t0

        dm = self.make_delta_dm1(t2, actspace)

        # --- Undo canonicalization
        if self.occtype == 'occupied' and r_occ is not None:
            dm = self._undo_canonicalization(dm, r_occ)
        elif self.occtype == 'virtual' and r_vir is not None:
            dm = self._undo_canonicalization(dm, r_vir)
        # --- Diagonalize environment-environment block
        dm = self._dm_take_env(dm)
        t0 = timer()
        r_bno, n_bno = self._diagonalize_dm(dm)
        t_diag = timer()-t0
        c_bno = spinalg.dot(self.c_env, r_bno)
        c_bno = fix_orbital_sign(c_bno)[0]

        self.log.timing("Time MP2 bath:  integrals= %s  amplitudes= %s  diagonal.= %s  total= %s",
                *map(time_string, (t_ao2mo, t_amps, t_diag, (timer()-t_init))))

        return c_bno, n_bno


class UMP2_BNO_Bath(MP2_BNO_Bath, BNO_Bath_UHF):

    def _get_mo_energy(self, fock, actspace):
        c_act_a, c_act_b = actspace.c_active
        mo_energy_a = einsum('ai,ab,bi->i', c_act_a, fock[0], c_act_a)
        mo_energy_b = einsum('ai,ab,bi->i', c_act_b, fock[1], c_act_b)
        return (mo_energy_a, mo_energy_b)

    def _get_eris(self, actspace):
        # We only need the (ov|ov) block for MP2:
        mo_a = [actspace.c_active_occ[0], actspace.c_active_vir[0]]
        mo_b = [actspace.c_active_occ[1], actspace.c_active_vir[1]]
        eris_aa = self.base.get_eris_array(mo_a + mo_a)
        eris_ab = self.base.get_eris_array(mo_a + mo_b)
        eris_bb = self.base.get_eris_array(mo_b + mo_b)
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

    def make_delta_dm1(self, t2, actspace):
        t2aa, t2ab, t2bb = t2

        if self.project_t2:
            self.log.debugv("Constructing DM from projected amplitudes (method= %s).", self.project_t2)
            eiga, eigb = self.dmet_bath.n_dmet
            assert np.all(eiga > -1e-10)
            assert np.all(eigb > -1e-10)
            assert np.all(eiga-1 < 1e-10)
            assert np.all(eigb-1 < 1e-10)
            eiga = np.clip(eiga, 0, 1)
            eigb = np.clip(eigb, 0, 1)
            if self.project_t2 == 'fragment':
                weightsa = len(eiga)*[0]
                weightsb = len(eigb)*[0]
            elif self.project_t2 == 'full':
                weightsa = len(eiga)*[1]
                weightsb = len(eigb)*[1]
            elif self.project_t2 == 'linear':
                weightsa = 2*abs(np.fmin(eiga, 1-eiga))
                weightsb = 2*abs(np.fmin(eigb, 1-eigb))
            elif self.project_t2 == 'entropy':
                weightsa = 4*eiga*(1-eiga)
                weightsb = 4*eigb*(1-eigb)
            elif self.project_t2 == 'sqrt-entropy':
                weightsa = 2*np.sqrt(eiga*(1-eiga))
                weightsb = 2*np.sqrt(eigb*(1-eigb))
            else:
                raise ValueError
                raise ValueError("Unknown value for project_t2: %s" % self.project_t2)
            assert np.all(weightsa > -1e-14)
            assert np.all(weightsb > -1e-14)
            assert np.all(weightsa-1 < 1e-14)
            assert np.all(weightsb-1 < 1e-14)
            weightsa = hstack(self.fragment.n_frag[0]*[1], weightsa)
            weightsb = hstack(self.fragment.n_frag[1]*[1], weightsb)

            # Project and symmetrize:
            ovlp = self.fragment.base.get_ovlp()
            c_fragdmet_a = hstack(self.fragment.c_frag[0], self.dmet_bath.c_dmet[0])
            c_fragdmet_b = hstack(self.fragment.c_frag[1], self.dmet_bath.c_dmet[1])
            if self.occtype == 'occupied':
                rota = dot(actspace.c_active_vir[0].T, ovlp, c_fragdmet_a)
                rotb = dot(actspace.c_active_vir[1].T, ovlp, c_fragdmet_b)
                proja = einsum('ix,x,jx->ij', rota, weightsa, rota)
                projb = einsum('ix,x,jx->ij', rotb, weightsb, rotb)
                t2aa = einsum('xa,ijab->ijxb', proja, t2aa)
                t2bb = einsum('xa,ijab->ijxb', projb, t2bb)
                t2ab = (einsum('xa,ijab->ijxb', proja, t2ab)
                      + einsum('xb,ijab->ijax', projb, t2ab))/2
            elif self.occtype == 'virtual':
                rota = dot(actspace.c_active_occ[0].T, ovlp, c_fragdmet_a)
                rotb = dot(actspace.c_active_occ[1].T, ovlp, c_fragdmet_b)
                proja = einsum('ix,x,jx->ij', rota, weightsa, rota)
                projb = einsum('ix,x,jx->ij', rotb, weightsb, rotb)
                t2aa = einsum('xi,i...->x...', proja, t2aa)
                t2bb = einsum('xi,i...->x...', projb, t2bb)
                t2ab = (einsum('xi,i...->x...', proja, t2ab)
                      + einsum('xj,ij...->ix...', projb, t2ab))/2
            t2aa = (t2aa + t2aa.transpose(1,0,3,2))/2
            t2bb = (t2bb + t2bb.transpose(1,0,3,2))/2
        else:
            self.log.debugv("Constructing DM from complete amplitudes")

        # Construct occupied-occupied DM
        if self.occtype == 'occupied':
            dma  = (einsum('imef,jmef->ij', t2aa.conj(), t2aa)/2
                  + einsum('imef,jmef->ij', t2ab.conj(), t2ab))
            dmb  = (einsum('imef,jmef->ij', t2bb.conj(), t2bb)/2
                  + einsum('mief,mjef->ij', t2ab.conj(), t2ab))
        # Construct virtual-virtual DM
        elif self.occtype == 'virtual':
            dma  = (einsum('mnae,mnbe->ba', t2aa.conj(), t2aa)/2
                  + einsum('mnae,mnbe->ba', t2ab.conj(), t2ab))
            dmb  = (einsum('mnae,mnbe->ba', t2bb.conj(), t2bb)/2
                  + einsum('mnea,mneb->ba', t2ab.conj(), t2ab))
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
