
import numpy as np

import pyscf
import pyscf.mp
import pyscf.pbc
import pyscf.pbc.mp

from vayesta.core.util import *
from vayesta.core.linalg import recursive_block_svd
from . import helper

from .dmet import DMET_Bath

class BNO_Bath(DMET_Bath):
    """DMET + Bath natural orbitals."""

    def __init__(self, fragment, *args, **kwargs):
        super().__init__(fragment, *args, **kwargs)
        # Results
        self.c_bno_occ = None
        self.c_bno_vir = None
        self.n_bno_occ = None
        self.n_bno_vir = None

    def kernel(self):
        # --- DMET bath
        super().kernel()
        # --- Natural environment orbitals:
        c_env_occ = self.c_env_occ
        c_env_vir = self.c_env_vir
        c_cluster_occ = self.c_cluster_occ
        c_cluster_vir = self.c_cluster_vir
        self.c_bno_occ, self.n_bno_occ = self.make_bno_bath('occ', c_cluster_occ, c_cluster_vir, c_env_occ, c_env_vir)
        self.c_bno_vir, self.n_bno_vir = self.make_bno_bath('vir', c_cluster_occ, c_cluster_vir, c_env_occ, c_env_vir)

    def make_bno_coeff(self, *args, **kwargs):
        raise AbstractMethodError()

    def make_bno_bath(self, kind, c_cluster_occ=None, c_cluster_vir=None, c_env_occ=None, c_env_vir=None):
        if kind not in ('occ', 'vir'):
            raise ValueError("kind not in ['occ', 'vir']: %r" % kind)
        if c_cluster_occ is None: c_cluster_occ = self.c_cluster_occ
        if c_cluster_vir is None: c_cluster_vir = self.c_cluster_vir
        if c_env_occ is None: c_env_occ = self.c_env_occ
        if c_env_vir is None: c_env_vir = self.c_env_vir

        c_env = c_env_occ if (kind == 'occ') else c_env_vir
        if c_env.shape[-1] == 0:
            return c_env, np.zeros((0,))

        name = {'occ': "occupied", 'vir': "virtual"}[kind]

        self.log.info("Making %s Bath NOs", name.capitalize())
        self.log.info("-------%s---------", len(name)*'-')
        self.log.changeIndentLevel(1)
        t0 = timer()
        c_bno, n_bno = self.make_bno_coeff(
                kind, c_cluster_occ, c_cluster_vir, c_env_occ, c_env_vir)
        self.log.debugv("BNO eigenvalues:\n%r", n_bno)
        if len(n_bno) > 0:
            self.log.info("%s Bath NO Histogram", name.capitalize())
            self.log.info("%s------------------", len(name)*'-')
            for line in helper.plot_histogram(n_bno):
                self.log.info(line)
        self.log.timing("Time for %s BNOs:  %s", name, time_string(timer()-t0))
        self.log.changeIndentLevel(-1)

        return c_bno, n_bno

    def get_occupied_bath(self, bno_threshold=None, bno_number=None):
        self.log.info("Occupied BNOs:")
        return self.truncate_bno(self.c_bno_occ, self.n_bno_occ, bno_threshold=bno_threshold, bno_number=bno_number)

    def get_virtual_bath(self, bno_threshold=None, bno_number=None):
        self.log.info("Virtual BNOs:")
        return self.truncate_bno(self.c_bno_vir, self.n_bno_vir, bno_threshold=bno_threshold, bno_number=bno_number)

    def truncate_bno(self, c_bno, n_bno, bno_threshold=None, bno_number=None):
        """Split natural orbitals (NO) into bath and rest."""
        if bno_number is not None:
            pass
        elif bno_threshold is not None:
            bno_threshold *= self.fragment.opts.bno_threshold_factor
            bno_number = np.count_nonzero(n_bno >= bno_threshold)
        else:
            raise ValueError()

        # Logging
        fmt = "  > %4s: N= %4d  max= % 9.3g  min= % 9.3g  sum= % 9.3g ( %7.3f %%)"
        def log(name, n_part):
            if len(n_part) > 0:
                with np.errstate(invalid='ignore'): # supress 0/0=nan warning
                    self.log.info(fmt, name, len(n_part), max(n_part), min(n_part), np.sum(n_part),
                            100*np.sum(n_part)/np.sum(n_bno))
            else:
                self.log.info(fmt[:fmt.index('max')].rstrip(), name, 0)
        log("Bath", n_bno[:bno_number])
        log("Rest", n_bno[bno_number:])

        c_bno, c_rest = np.hsplit(c_bno, [bno_number])
        return c_bno, c_rest


class MP2_BNO_Bath(BNO_Bath):

    def __init__(self, *args, local_dm=False, canonicalize=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.local_dm = local_dm
        # Canonicalization can be set separately for occupied and virtual:
        if canonicalize in (True, False):
            canonicalize = 2*[canonicalize]
        self.canonicalize = canonicalize

    def make_bno_coeff(self, kind, c_cluster_occ=None, c_cluster_vir=None, c_env_occ=None, c_env_vir=None, eris=None):
        """Select virtual space from MP2 natural orbitals (NOs) according to occupation number.

        Parameters
        ----------
        c_occ : ndarray
            Active occupied orbitals.
        c_vir : ndarray
            Active virtual orbitals.
        c_occ_frozen : ndarray, optional
            Frozen occupied orbitals.
        c_vir_frozen : ndarray, optional
            Frozen virtual orbitals.
        eris: TODO

        Returns
        -------
        c_bno: (n(AO), n(BNO)) array
            Bath natural orbital coefficients.
        n_bno: (n(BNO)) array
            Bath natural orbital occupation numbers.
        """

        if c_cluster_occ is None: c_cluster_occ = self.c_cluster_occ
        if c_cluster_vir is None: c_cluster_vir = self.c_cluster_vir
        if c_env_occ is None: c_env_occ = self.c_env_occ
        if c_env_vir is None: c_env_vir = self.c_env_vir

        if kind == 'occ':
            ncluster = c_cluster_occ.shape[-1]
            c_occ = np.hstack((c_cluster_occ, c_env_occ))
            c_vir = c_cluster_vir
            c_occ_frozen = None
            c_vir_frozen = c_env_vir
            c_env = c_env_occ
        elif kind == 'vir':
            ncluster = c_cluster_vir.shape[-1]
            c_occ = c_cluster_occ
            c_vir = np.hstack((c_cluster_vir, c_env_vir))
            c_occ_frozen = c_env_occ
            c_vir_frozen = None
            c_env = c_env_vir
        else:
            raise ValueError("Unknown kind: %r" % kind)
        self.log.debugv("n(cluster)= %d", ncluster)

        # Canonicalization [optional]
        self.log.debugv("canonicalize: occ= %r vir= %r", *self.canonicalize)
        if self.canonicalize[0]:
            c_occ, r_occ, e_occ = self.fragment.canonicalize_mo(c_occ, eigvals=True)
            self.log.debugv("eigenvalues (occ):\n%r", e_occ)
        if self.canonicalize[1]:
            c_vir, r_vir, e_vir = self.fragment.canonicalize_mo(c_vir, eigvals=True)
            self.log.debugv("eigenvalues (vir):\n%r", e_vir)

        # Setup MP2 object
        nao = c_occ.shape[0]
        assert (c_vir.shape[0] == nao)
        if c_occ_frozen is None:
            c_occ_frozen = np.zeros((nao, 0))
        if c_vir_frozen is None:
            c_vir_frozen = np.zeros((nao, 0))

        self.log.debugv("n(frozen occ)= %d  n(active occ)= %d  n(active vir)= %d  n(frozen vir)= %d",
                *[x.shape[-1] for x in (c_occ_frozen, c_occ, c_vir, c_vir_frozen)])
        c_active = np.hstack((c_occ, c_vir))
        c_all = np.hstack((c_occ_frozen, c_active, c_vir_frozen))
        nmo = c_all.shape[-1]
        nocc_frozen = c_occ_frozen.shape[-1]
        nvir_frozen = c_vir_frozen.shape[-1]
        frozen_indices = list(range(nocc_frozen)) + list(range(nmo-nvir_frozen, nmo))
        self.log.debugv("Frozen indices: %r", frozen_indices)
        if self.base.boundary_cond == 'open':
            cls = pyscf.mp.MP2
        else:
            cls = pyscf.pbc.mp.MP2
        mp2 = cls(self.mf, mo_coeff=c_all, frozen=frozen_indices)

        # Integral transformation
        t0 = timer()
        if eris is None:
            eris = self.base.get_eris_object(mp2)
        # Reuse previously obtained integral transformation into N^2 sized quantity (rather than N^4)
        else:
            self.log.debug("Transforming previous eris.")
            eris = transform_mp2_eris(eris, c_occ, c_vir, ovlp=self.base.get_ovlp())
        self.log.timing("Time for integral transformation:  %s", time_string(timer()-t0))
        assert (eris.ovov is not None)

        t0 = timer()
        e_mp2_full, t2 = mp2.kernel(eris=eris)
        nocc, nvir = t2.shape[0], t2.shape[2]
        assert (c_occ.shape[-1] == nocc)
        assert (c_vir.shape[-1] == nvir)
        self.log.timing("Time for MP2 kernel:  %s", time_string(timer()-t0))

        # Energies
        e_mp2_full *= self.fragment.sym_factor
        # Symmetrize irrelavant?
        #t2loc = self.project_amplitudes_to_fragment(mp2, None, t2, symmetrize=True)[1]
        t2loc = self.fragment.project_amplitudes_to_fragment(mp2, None, t2)[1]
        e_mp2 = self.fragment.sym_factor * mp2.energy(t2loc, eris)
        self.log.debug("Bath E(MP2):  Cluster= %+16.8f Ha  Fragment= %+16.8f Ha", e_mp2_full, e_mp2)

        # MP2 density matrix
        #dm_occ = dm_vir = None
        if self.local_dm is False:
            self.log.debug("Constructing DM from full T2 amplitudes.")
            t2l = t2r = t2
            # This is equivalent to:
            # do, dv = pyscf.mp.mp2._gamma1_intermediates(mp2, eris=eris)
            # do, dv = -2*do, 2*dv
        elif self.local_dm is True:
            # LOCAL DM IS NOT RECOMMENDED - USE SEMI OR FALSE
            self.log.warning("Using local_dm = True is not recommended - use 'semi' or False")
            self.log.debug("Constructing DM from local T2 amplitudes.")
            t2l = t2r = t2loc
        elif self.local_dm == "semi":
            self.log.debug("Constructing DM from semi-local T2 amplitudes.")
            t2l, t2r = t2loc, t2
        else:
            raise ValueError("Unknown value for local_dm: %r" % self.local_dm)

        if kind == 'occ':
            dm = 2*(2*einsum('ikab,jkab->ij', t2l, t2r)
                    - einsum('ikab,kjab->ij', t2l, t2r))
            # This turns out to be equivalent:
            #dm = 2*(2*einsum("kiba,kjba->ij", t2l, t2r)
            #        - einsum("kiba,kjab->ij", t2l, t2r))
        else:
            dm = 2*(2*einsum('ijac,ijbc->ab', t2l, t2r)
                    - einsum('ijac,ijcb->ab', t2l, t2r))
        if self.local_dm == 'semi':
            dm = (dm + dm.T)/2
        assert np.allclose(dm, dm.T)

        # Undo canonicalization
        if kind == 'occ' and self.canonicalize[0]:
            dm = dot(r_occ, dm, r_occ.T)
            self.log.debugv("Undoing occupied canonicalization")
        elif kind == 'vir' and self.canonicalize[1]:
            dm = dot(r_vir, dm, r_vir.T)
            self.log.debugv("Undoing virtual canonicalization")

        clt, env = np.s_[:ncluster], np.s_[ncluster:]

        self.log.debugv("Tr[D]= %r", np.trace(dm))
        self.log.debugv("Tr[D(cluster,cluster)]= %r", np.trace(dm[clt,clt]))
        self.log.debugv("Tr[D(env,env)]= %r", np.trace(dm[env,env]))
        n_bno, c_bno = np.linalg.eigh(dm[env,env])
        n_bno, c_bno = n_bno[::-1], c_bno[:,::-1]
        c_bno = np.dot(c_env, c_bno)
        c_bno = fix_orbital_sign(c_bno)[0]

        return c_bno, n_bno



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
