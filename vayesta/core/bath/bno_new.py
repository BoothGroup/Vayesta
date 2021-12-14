
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

class BNO_Bath(FragmentBath):
    """Bath natural orbital (BNO) bath, requires DMET bath."""

    def __init__(self, fragment, dmet_bath, *args, **kwargs):
        super().__init__(fragment, *args, **kwargs)
        self.dmet_bath = dmet_bath
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
        return self.dmet_bath.c_cluster_occ

    @property
    def c_cluster_vir(self):
        """Virtual DMET cluster orbitals."""
        return self.dmet_bath.c_cluster_vir

    def kernel(self):
        """Make bath natural orbitals."""
        self.c_bno_occ, self.n_bno_occ = self.make_bno_bath('occ')
        self.c_bno_vir, self.n_bno_vir = self.make_bno_bath('vir')

    def make_bno_coeff(self, *args, **kwargs):
        raise AbstractMethodError()

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

    def get_occupied_bath(self, bno_threshold=None, bno_number=None, **kwargs):
        return self.truncate_bno(self.c_bno_occ, self.n_bno_occ, bno_threshold=bno_threshold,
                bno_number=bno_number, header="occupied BNOs:")

    def get_virtual_bath(self, bno_threshold=None, bno_number=None, **kwargs):
        return self.truncate_bno(self.c_bno_vir, self.n_bno_vir, bno_threshold=bno_threshold,
                bno_number=bno_number, header="virtual BNOs:")

    def truncate_bno(self, c_bno, n_bno, bno_threshold=None, bno_number=None, header=None):
        """Split natural orbitals (NO) into bath and rest."""

        # For UHF, call recursively:
        if np.ndim(c_bno[0]) == 2:
            c_bno_a, c_rest_a = self.truncate_bno(c_bno[0], n_bno[0], bno_threshold=bno_threshold,
                    bno_number=bno_number, header='Alpha %s' % header)
            c_bno_b, c_rest_b = self.truncate_bno(c_bno[1], n_bno[1], bno_threshold=bno_threshold,
                    bno_number=bno_number, header='Beta %s' % header)
            return (c_bno_a, c_bno_b), (c_rest_a, c_rest_b)

        if bno_number is not None:
            pass
        elif bno_threshold is not None:
            bno_number = np.count_nonzero(n_bno >= bno_threshold)
        else:
            raise ValueError("Either `bno_threshold` or `bno_number` needs to be specified.")

        # Logging
        if header:
            self.log.info(header.capitalize())
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
        dmet_bath = self.dmet_bath
        nao = self.mol.nao
        zero_space = np.zeros((nao, 0)) if self.spin_restricted else np.zeros((2, nao, 0))
        if kind == 'occ':
            c_active_occ = stack_mo(self.c_cluster_occ, dmet_bath.c_env_occ)
            c_frozen_occ = zero_space
            c_active_vir, c_frozen_vir = self.c_cluster_vir, dmet_bath.c_env_vir
        elif kind == 'vir':
            c_active_occ, c_frozen_occ = self.c_cluster_occ, self.dmet_bath.c_env_occ
            c_active_vir = stack_mo(self.c_cluster_vir, self.dmet_bath.c_env_vir)
            c_frozen_vir = zero_space
        else:
            raise ValueError("Unknown kind: %r" % kind)
        actspace = ActiveSpace(self.mf, c_active_occ, c_active_vir, c_frozen_occ, c_frozen_vir)
        return actspace

    def _undo_canonicalization(self, dm, rot):
        self.log.debugv("Undoing canonicalization")
        return dot(rot, dm, rot.T)

    def _get_dmet_cluster_size(self):
        return (self.c_cluster_occ.shape[-1], self.c_cluster_vir.shape[-1])

    def _dm_take_env(self, dm, kind):
        if kind == 'occ':
            ncluster = self._get_dmet_cluster_size()[0]
        elif kind == 'vir':
            ncluster = self._get_dmet_cluster_size()[1]
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

    def _get_dmet_cluster_size(self):
        return ((self.c_cluster_occ[0].shape[-1], self.c_cluster_vir[0].shape[-1]),
                (self.c_cluster_occ[1].shape[-1], self.c_cluster_vir[1].shape[-1]))

    def _dm_take_env(self, dm, kind):
        if kind == 'occ':
            ncluster = self._get_dmet_cluster_size()[0]
        elif kind == 'vir':
            ncluster = self._get_dmet_cluster_size()[1]
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

    def __init__(self, *args, local_dm=False, canonicalize=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.local_dm = local_dm
        # Canonicalization can be set separately for occupied and virtual:
        if np.ndim(canonicalize) == 0:
            canonicalize = 2*[canonicalize]
        self.canonicalize = canonicalize

    def get_mp2_class(self):
        """TODO: Do not use PySCF MP2 classes."""
        if self.base.boundary_cond == 'open':
            return pyscf.mp.MP2
        return pyscf.pbc.mp.MP2

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
        if self.local_dm is False:
            self.log.debug("Constructing DM from full T2-amplitudes.")
            # This is equivalent to:
            # do, dv = pyscf.mp.mp2._gamma1_intermediates(mp2, eris=eris)
            # do, dv = -2*do, 2*dv
            if kind == 'occ':
                dm = norm*(2*einsum('ikab,jkab->ij', t2, t2)
                           - einsum('ikab,kjab->ij', t2, t2))
                # Note that this is equivalent to:
                #dm = 2*(2*einsum("kiba,kjba->ij", t2l, t2r)
                #        - einsum("kiba,kjab->ij", t2l, t2r))
            else:
                dm = norm*(2*einsum('ijac,ijbc->ab', t2, t2)
                           - einsum('ijac,ijcb->ab', t2, t2))
            assert np.allclose(dm, dm.T)
            return dm

        # Project one T-amplitude onto fragment
        self.log.debug("Constructing DM from projected T2-amplitudes.")
        #px = self.fragment.get_overlap_c2f()[0]
        #t2x = self.fragment.project_amp2_to_fragment(t2)
        ovlp = self.fragment.base.get_ovlp()
        px = dot(actspace.c_active_occ.T, ovlp, self.fragment.c_proj)

        t2x = einsum('ix,ijab->xjab', px, t2)
        #t2x = t2
        if kind == 'occ':
            #dm = norm*(2*einsum('ikab,jkab->ij', t2x, t2x)
            #           - einsum('ikab,kjab->ij', t2x, t2x))
            dm = norm*(2*einsum('kiab,kjab->ij', t2x, t2x)
                       - einsum('kiab,kjba->ij', t2x, t2x))
        else:
            dm = norm*(2*einsum('ijac,ijbc->ab', t2x, t2x)
                       - einsum('ijac,ijcb->ab', t2x, t2x))
        dm = (dm + dm.T) / 2
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

        # --- Setup PySCF MP2 object
        cls = self.get_mp2_class()
        self.log.debugv("MP2 class= %r actspace= %r", cls, actspace)
        mp2 = cls(self.mf, mo_coeff=actspace.coeff, frozen=actspace.get_frozen_indices())

        # -- Integral transformation
        if eris is None:
            with log_time(self.log.timing, "Time for AO->MO transformation: %s"):
                eris = self.base.get_eris_object(mp2, fock=fock)
        # Reuse previously obtained integral transformation into N^2 sized quantity (rather than N^4)
        #else:
        #    self.log.debug("Transforming previous eris.")
        #    eris = transform_mp2_eris(eris, actspace.c_active_occ, actspace.c_active_vir, ovlp=self.base.get_ovlp())
        # TODO: DF-MP2
        assert (eris.ovov is not None)

        # --- Kernel
        with log_time(self.log.timing, "Time for MP2 kernel: %s"):
            e_mp2_full, t2 = mp2.kernel(eris=eris)

        # --- MP2 energies
        #e_mp2_full *= self.fragment.get_energy_prefactor()
        ## Symmetrize irrelevant?
        ##t2loc = self.fragment.project_amplitudes_to_fragment(mp2, None, t2)[1]
        #t2loc = self.fragment.project_amplitude_to_fragment(t2, c_occ=actspace.c_active_occ, c_vir=False)
        #e_mp2 = self.fragment.get_energy_prefactor() * mp2.energy(t2loc, eris)
        #self.log.debug("MP2 bath energy:  E(Cluster)= %+16.8f Ha  E(Fragment)= %+16.8f Ha", e_mp2_full, e_mp2)

        e_mp2_full *= self.fragment.get_energy_prefactor()
        self.log.debug("MP2 cluster energy= %s", energy_string(e_mp2_full))

        #dm = self.make_delta_dm1(kind, t2, t2loc)
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

    def __init__(self, *args, local_dm=False, **kwargs):
        super().__init__(*args, local_dm=local_dm, **kwargs)

    def get_mp2_class(self):
        if self.base.boundary_cond == 'open':
            return pyscf.mp.UMP2
        return pyscf.pbc.mp.UMP2

    def make_delta_dm1(self, kind, t2, actspace):
        taa, tab, tbb = t2
        if self.local_dm is False:
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
