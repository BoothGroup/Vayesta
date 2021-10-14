"""These functions take a cluster instance as first argument ("self")."""

import numpy as np

import pyscf
import pyscf.mp
import pyscf.pbc
import pyscf.pbc.mp

from vayesta.core.util import *
from vayesta.core.linalg import recursive_block_svd
#from vayesta.ewf.psubspace import transform_mp2_eris
#from vayesta.ewf import helper
from . import helper

from .dmet import DMET_Bath

class BNO_Bath(DMET_Bath):

    def __init__(self, fragment, *args, canonicalize=True, local_dm=False, **kwargs):
        super().__init__(fragment, *args, **kwargs)
        self.canonicalize = canonicalize
        self.local_dm = local_dm
        # Output
        self.c_no_occ = None
        self.c_no_vir = None
        self.n_no_occ = None
        self.n_no_vir = None

    def kernel(self, bath_type='mp2-bno'):
        # --- DMET bath
        super().kernel()

        # Add cluster orbitals to plot
        c_env_occ = self.c_env_occ
        c_env_vir = self.c_env_vir
        c_cluster_occ = self.c_cluster_occ
        c_cluster_vir = self.c_cluster_vir

        # TODO: clean
        self.log.debugv("bath_type= %r", bath_type)
        if bath_type is None or bath_type.upper() == 'NONE':
            c_no_occ = c_env_occ
            c_no_vir = c_env_vir
            if self.base.is_rhf:
                n_no_occ = np.full((c_no_occ.shape[-1],), -np.inf)
                n_no_vir = np.full((c_no_vir.shape[-1],), -np.inf)
            else:
                n_no_occ = np.full((2, c_no_occ[0].shape[-1]), -np.inf)
                n_no_vir = np.full((2, c_no_vir[1].shape[-1]), -np.inf)
        elif bath_type.upper() == 'ALL':
            c_no_occ = c_env_occ
            c_no_vir = c_env_vir
            if self.base.is_rhf:
                n_no_occ = np.full((c_no_occ.shape[-1],), np.inf)
                n_no_vir = np.full((c_no_vir.shape[-1],), np.inf)
            else:
                n_no_occ = np.full((2, c_no_occ[0].shape[-1]), np.inf)
                n_no_vir = np.full((2, c_no_vir[1].shape[-1]), np.inf)
        elif bath_type.upper() == 'MP2-BNO':
            c_no_occ, n_no_occ = self.make_bno_bath(c_cluster_occ, c_cluster_vir, c_env_occ, c_env_vir, 'occ')
            c_no_vir, n_no_vir = self.make_bno_bath(c_cluster_occ, c_cluster_vir, c_env_occ, c_env_vir, 'vir')
        else:
            raise ValueError("Unknown bath type: '%s'" % bath_type)

        self.c_no_occ = c_no_occ
        self.c_no_vir = c_no_vir
        self.n_no_occ = n_no_occ
        self.n_no_vir = n_no_vir

    def make_bno_bath(self, c_cluster_occ, c_cluster_vir, c_env_occ, c_env_vir, kind):
        assert kind in ('occ', 'vir')
        c_env = c_env_occ if (kind == 'occ') else c_env_vir
        if c_env.shape[-1] == 0:
            return c_env, np.zeros((0,))

        name = {'occ': "occupied", 'vir': "virtual"}[kind]

        self.log.info("Making %s Bath NOs", name.capitalize())
        self.log.info("-------%s---------", len(name)*'-')
        self.log.changeIndentLevel(1)
        t0 = timer()
        c_no, n_no = self.make_mp2_bno(
                kind, c_cluster_occ, c_cluster_vir, c_env_occ, c_env_vir)
        self.log.debugv('BNO eigenvalues:\n%r', n_no)
        if len(n_no) > 0:
            self.log.info("%s Bath NO Histogram", name.capitalize())
            self.log.info("%s------------------", len(name)*'-')
            for line in helper.plot_histogram(n_no):
                self.log.info(line)
        self.log.timing("Time for %s BNOs:  %s", name, time_string(timer()-t0))
        self.log.changeIndentLevel(-1)

        return c_no, n_no

    def get_occupied_bath(self, bno_threshold=None, bno_number=None):
        self.log.info("Occupied BNOs:")
        return self.truncate_bno(self.c_no_occ, self.n_no_occ, bno_threshold=bno_threshold, bno_number=bno_number)

    def get_virtual_bath(self, bno_threshold=None, bno_number=None):
        self.log.info("Virtual BNOs:")
        return self.truncate_bno(self.c_no_vir, self.n_no_vir, bno_threshold=bno_threshold, bno_number=bno_number)

    def truncate_bno(self, c_no, n_no, bno_threshold=None, bno_number=None):
        """Split natural orbitals (NO) into bath and rest."""
        if bno_number is not None:
            pass
        elif bno_threshold is not None:
            bno_threshold *= self.fragment.opts.bno_threshold_factor
            bno_number = np.count_nonzero(n_no >= bno_threshold)
        else:
            raise ValueError()

        # Logging
        fmt = "  > %4s: N= %4d  max= % 9.3g  min= % 9.3g  sum= % 9.3g ( %7.3f %%)"
        def log(name, n_part):
            if len(n_part) > 0:
                with np.errstate(invalid='ignore'): # supress 0/0=nan warning
                    self.log.info(fmt, name, len(n_part), max(n_part), min(n_part), np.sum(n_part),
                            100*np.sum(n_part)/np.sum(n_no))
            else:
                self.log.info(fmt[:fmt.index('max')].rstrip(), name, 0)
        log("Bath", n_no[:bno_number])
        log("Rest", n_no[bno_number:])

        c_bno, c_rest = np.hsplit(c_no, [bno_number])
        return c_bno, c_rest

    def make_mp2_bno(self, kind, c_cluster_occ, c_cluster_vir,
            c_env_occ=None, c_env_vir=None, canonicalize=None, local_dm=None, eris=None):
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
        canonicalize : bool, tuple(2), optional
            Canonicalize occupied/virtual active orbitals.
        eris: TODO

        Returns
        -------
        c_no
        n_no
        """
        if canonicalize is None: canonicalize = self.canonicalize
        if local_dm is None: local_dm = self.local_dm

        if self.c_dmet is None:
            raise RuntimeError("MP2 bath requires DMET bath.")

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
        if canonicalize in (True, False):
            canonicalize = 2*[canonicalize]
        self.log.debugv("canonicalize: occ= %r vir= %r", *canonicalize)
        if canonicalize[0]:
            c_occ, r_occ, e_occ = self.fragment.canonicalize_mo(c_occ, eigvals=True)
            self.log.debugv("eigenvalues (occ):\n%r", e_occ)
        if canonicalize[1]:
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
        if local_dm is False:
            self.log.debug("Constructing DM from full T2 amplitudes.")
            t2l = t2r = t2
            # This is equivalent to:
            # do, dv = pyscf.mp.mp2._gamma1_intermediates(mp2, eris=eris)
            # do, dv = -2*do, 2*dv
        elif local_dm is True:
            # LOCAL DM IS NOT RECOMMENDED - USE SEMI OR FALSE
            self.log.warning("Using local_dm = True is not recommended - use 'semi' or False")
            self.log.debug("Constructing DM from local T2 amplitudes.")
            t2l = t2r = t2loc
        elif local_dm == "semi":
            self.log.debug("Constructing DM from semi-local T2 amplitudes.")
            t2l, t2r = t2loc, t2
        else:
            raise ValueError("Unknown value for local_dm: %r" % local_dm)

        if kind == 'occ':
            dm = 2*(2*einsum('ikab,jkab->ij', t2l, t2r)
                    - einsum('ikab,kjab->ij', t2l, t2r))
            # This turns out to be equivalent:
            #dm = 2*(2*einsum("kiba,kjba->ij", t2l, t2r)
            #        - einsum("kiba,kjab->ij", t2l, t2r))
        else:
            dm = 2*(2*einsum('ijac,ijbc->ab', t2l, t2r)
                    - einsum('ijac,ijcb->ab', t2l, t2r))
        if local_dm == 'semi':
            dm = (dm + dm.T)/2
        assert np.allclose(dm, dm.T)

        # Undo canonicalization
        if kind == 'occ' and canonicalize[0]:
            dm = dot(r_occ, dm, r_occ.T)
            self.log.debugv("Undoing occupied canonicalization")
        elif kind == 'vir' and canonicalize[1]:
            dm = dot(r_vir, dm, r_vir.T)
            self.log.debugv("Undoing virtual canonicalization")

        clt, env = np.s_[:ncluster], np.s_[ncluster:]

        # TEST SVD BATH:
        #coeff, sv, order = recursive_block_svd(dm, n=ncluster)
        #c_no = np.dot(c_env, coeff)
        #n_no = 1/order
        #self.log.debugv("n_no= %r", n_no)

        self.log.debugv("Tr[D]= %r", np.trace(dm))
        self.log.debugv("Tr[D(cluster,cluster)]= %r", np.trace(dm[clt,clt]))
        self.log.debugv("Tr[D(env,env)]= %r", np.trace(dm[env,env]))
        n_no, c_no = np.linalg.eigh(dm[env,env])
        n_no, c_no = n_no[::-1], c_no[:,::-1]
        c_no = np.dot(c_env, c_no)

        return c_no, n_no

#    def get_mp2_correction(self, Co1, Cv1, Co2, Cv2):
#        """Calculate delta MP2 correction."""
#        e_mp2_all, eris = self.run_mp2(Co1, Cv1)[:2]
#        e_mp2_act = self.run_mp2(Co2, Cv2, eris=eris)[0]
#        e_delta_mp2 = e_mp2_all - e_mp2_act
#        self.log.debug("MP2 correction: all=%.4g, active=%.4g, correction=%+.4g",
#                e_mp2_all, e_mp2_act, e_delta_mp2)
#        return e_delta_mp2
#
#

# ================================================================================================ #

#def make_mp2_bath(self, c_cluster_occ, c_cluster_vir, c_env_occ, c_env_vir,
#        kind, mp2_correction=None):
#    """Select occupied or virtual bath space from MP2 natural orbitals.
#
#    The natural orbitals are calculated only including the local virtual (occupied)
#    cluster orbitals when calculating occupied (virtual) bath orbitals, i.e. they do not correspond
#    to the full system MP2 natural orbitals and are different for every cluster.
#
#    Parameters
#    ----------
#    c_cluster_occ : ndarray
#        Occupied cluster (fragment + DMET bath) orbitals.
#    c_cluster_vir : ndarray
#        Virtual cluster (fragment + DMET bath) orbitals.
#    C_env : ndarray
#        Environment orbitals. These need to be off purely occupied character if kind=="occ"
#        and of purely virtual character if kind=="vir".
#    kind : str ["occ", "vir"]
#        Calculate occupied or virtual bath orbitals.
#
#    Returns
#    -------
#    c_no : ndarray
#        MP2 natural orbitals.
#    n_no : ndarray
#        Natural occupation numbers.
#    e_delta_mp2 : float
#        MP2 correction energy (0 if self.mp2_correction == False)
#    """
#    if kind not in ("occ", "vir"):
#        raise ValueError("Unknown kind: %s", kind)
#    if mp2_correction is None: mp2_correction = self.mp2_correction[0 if kind == "occ" else 1]
#    kindname = {"occ": "occupied", "vir" : "virtual"}[kind]
#
#    # All occupied and virtual orbitals
#    c_all_occ = np.hstack((c_cluster_occ, c_env_occ))
#    c_all_vir = np.hstack((c_cluster_vir, c_env_vir))
#
#    # All occupied orbitals, only cluster virtual orbitals
#    if kind == "occ":
#        c_occ = np.hstack((c_cluster_occ, c_env_occ))
#        c_vir = c_cluster_vir
#        ... = self.run_mp2(c_occ=c_occ, c_vir=c_vir, c_vir_frozen=c_env_vir)
#        ncluster = c_occclst.shape[-1]
#        nenv = c_occ.shape[-1] - ncluster
#
#    # All virtual orbitals, only cluster occupied orbitals
#    elif kind == "vir":
#        c_occ = c_cluster_occ
#        c_vir = np.hstack((c_cluster_vir, c_env_vir))
#        ... = self.run_mp(c_occ, c_vir, c_occenv=c_occenv, c_virenv=None)
#        ncluster = c_virclst.shape[-1]
#        nenv = c_vir.shape[-1] - ncluster
#
#    # Diagonalize environment-environment block of MP2 DM correction
#    # and rotate into natural orbital basis, with the orbitals sorted
#    # with decreasing (absolute) occupation change
#    # [Note that dm_occ is minus the change of the occupied DM]
#
#    env = np.s_[ncluster:]
#    dm = dm[env,env]
#    dm_occ, dm_rot = np.linalg.eigh(dm)
#    assert (len(dm_occ) == nenv)
#    if np.any(dm_occ < -1e-12):
#        raise RuntimeError("Negative occupation values detected: %r" % dm_occ[dm_occ < -1e-12])
#    dm_occ, dm_rot = dm_occ[::-1], dm_rot[:,::-1]
#    c_rot = np.dot(c_env, dm_rot)
#
#    with open("mp2-bath-occupation.txt", "ab") as f:
#        #np.savetxt(f, dm_occ[np.newaxis], header="MP2 bath orbital occupation of cluster %s" % self.name)
#        np.savetxt(f, dm_occ, fmt="%.10e", header="%s MP2 bath orbital occupation of cluster %s" % (kindname.title(), self.name))
#
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
#
#    return c_no, n_no
