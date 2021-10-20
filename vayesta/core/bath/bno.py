
import numpy as np

import pyscf
import pyscf.mp
import pyscf.pbc
import pyscf.pbc.mp

from vayesta.core.util import *
from vayesta.core.actspace import ActiveSpace
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
        self.c_bno_occ, self.n_bno_occ = self.make_bno_bath('occ')
        self.c_bno_vir, self.n_bno_vir = self.make_bno_bath('vir')

    def make_bno_coeff(self, *args, **kwargs):
        raise AbstractMethodError()

    def make_bno_bath(self, kind):
        if kind not in ('occ', 'vir'):
            raise ValueError("kind not in ['occ', 'vir']: %r" % kind)

        c_env = self.c_env_occ if (kind == 'occ') else self.c_env_vir
        if c_env.shape[-1] == 0:
            return c_env, np.zeros((0,))

        name = {'occ': "occupied", 'vir': "virtual"}[kind]

        self.log.info("Making %s Bath NOs", name.capitalize())
        self.log.info("-------%s---------", len(name)*'-')
        self.log.changeIndentLevel(1)
        t0 = timer()
        c_bno, n_bno = self.make_bno_coeff(kind)
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

    def get_mp2_class(self):
        if self.base.boundary_cond == 'open':
            return pyscf.mp.MP2
        return pyscf.pbc.mp.MP2

    def make_dm1(self, kind, t2, t2loc):
        """MP2 density matrix"""
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

        if kind == 'occ':
            dm = 2*(2*einsum('ikab,jkab->ij', t2l, t2r)
                    - einsum('ikab,kjab->ij', t2l, t2r))
            # Note that this is equivalent to:
            #dm = 2*(2*einsum("kiba,kjba->ij", t2l, t2r)
            #        - einsum("kiba,kjab->ij", t2l, t2r))
        else:
            dm = 2*(2*einsum('ijac,ijbc->ab', t2l, t2r)
                    - einsum('ijac,ijcb->ab', t2l, t2r))
        if self.local_dm == 'semi':
            dm = (dm + dm.T)/2
        assert np.allclose(dm, dm.T)
        return dm

    def get_active_space(self, kind):
        nao = self.mol.nao
        if kind == 'occ':
            c_active_occ, c_frozen_occ = stack_mo_coeffs(self.c_cluster_occ, self.c_env_occ), 0#np.zeros((nao, 0))
            c_active_vir, c_frozen_vir = self.c_cluster_vir, self.c_env_vir
        elif kind == 'vir':
            c_active_occ, c_frozen_occ = self.c_cluster_occ, self.c_env_occ
            c_active_vir, c_frozen_vir = stack_mo_coeffs(self.c_cluster_vir, self.c_env_vir), 0#np.zeros((nao, 0))
        else:
            raise ValueError("Unknown kind: %r" % kind)
        actspace = ActiveSpace(self.mf, c_active_occ, c_active_vir, c_frozen_occ, c_frozen_vir)
        return actspace

    def make_bno_coeff(self, kind, eris=None):
        """Construct MP2 bath natural orbital coefficients and occupation numbers.

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
        if kind == 'occ':
            c_env = self.c_env_occ
            ncluster = self.c_cluster_occ.shape[-1]
        elif kind == 'vir':
            c_env = self.c_env_vir
            ncluster = self.c_cluster_vir.shape[-1]

        self.log.debugv("n(cluster)= %d", ncluster)

        # --- Canonicalization [optional]
        if self.canonicalize[0]:
            self.log.debugv("Canonicalizing occupied orbitals")
            c_active_occ, r_occ, e_occ = self.fragment.canonicalize_mo(actspace_orig.c_active_occ, eigvals=True)
            self.log.debugv("Occupied eigenvalues:\n%r", e_occ)
        else:
            c_active_occ = actspace_orig.c_active_occ
        if self.canonicalize[1]:
            self.log.debugv("Canonicalizing virtual orbitals")
            c_active_vir, r_vir, e_vir = self.fragment.canonicalize_mo(actspace_orig.c_active_vir, eigvals=True)
            self.log.debugv("Virtual eigenvalues:\n%r", e_vir)
        else:
            c_active_vir = actspace_orig.c_active_vir
        actspace = ActiveSpace(self.mf, c_active_occ, c_active_vir,
                actspace_orig.c_frozen_occ, actspace_orig.c_frozen_vir)

        # --- Setup PySCF MP2 object
        cls = self.get_mp2_class()
        mo_coeff = actspace.coeff
        self.log.debugv('%r', actspace)
        frozen_indices = actspace.get_frozen_indices()
        self.log.debugv("Frozen indices: %r", frozen_indices)
        mp2 = cls(self.mf, mo_coeff=mo_coeff, frozen=frozen_indices)

        # -- Integral transformation
        t0 = timer()
        if eris is None:
            eris = self.base.get_eris_object(mp2)
        # Reuse previously obtained integral transformation into N^2 sized quantity (rather than N^4)
        else:
            self.log.debug("Transforming previous eris.")
            eris = transform_mp2_eris(eris, actspace.c_active_occ, actspace.c_active_vir, ovlp=self.base.get_ovlp())
        self.log.timing("Time for integral transformation:  %s", time_string(timer()-t0))
        assert (eris.ovov is not None)

        # --- Kernel
        t0 = timer()
        e_mp2_full, t2 = mp2.kernel(eris=eris)
        nocc, nvir = t2.shape[0], t2.shape[2]
        assert (actspace.nocc_active == nocc)
        assert (actspace.nvir_active == nvir)
        self.log.timing("Time for MP2 kernel:  %s", time_string(timer()-t0))

        # --- MP2 energies
        e_mp2_full *= self.fragment.get_energy_prefactor()
        # Symmetrize irrelevant?
        t2loc = self.fragment.project_amplitudes_to_fragment(mp2, None, t2)[1]
        e_mp2 = self.fragment.get_energy_prefactor() * mp2.energy(t2loc, eris)
        self.log.debug("MP2 bath energy:  E(Cluster)= %+16.8f Ha  E(Fragment)= %+16.8f Ha", e_mp2_full, e_mp2)

        dm = self.make_dm1(kind, t2, t2loc)

        # --- Undo canonicalization
        if kind == 'occ' and self.canonicalize[0]:
            dm = dot(r_occ, dm, r_occ.T)
            self.log.debugv("Undoing occupied canonicalization")
        elif kind == 'vir' and self.canonicalize[1]:
            dm = dot(r_vir, dm, r_vir.T)
            self.log.debugv("Undoing virtual canonicalization")

        # --- Diagonalize
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
