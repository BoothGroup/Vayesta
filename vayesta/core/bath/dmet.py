import numpy as np
import scipy
import scipy.linalg

from vayesta.core.util import *
from vayesta.core.bath.bath import Bath

DEFAULT_DMET_THRESHOLD = 1e-6

class DMET_Bath_RHF(Bath):

    def __init__(self, fragment, dmet_threshold=DEFAULT_DMET_THRESHOLD):
        super().__init__(fragment)
        self.dmet_threshold = dmet_threshold
        # Output
        self.c_dmet = None
        self.n_dmet = None
        self.c_cluster_occ = None
        self.c_cluster_vir = None
        self.c_env_occ = None
        self.c_env_vir = None

        self.dmet_bath = self

    def get_cluster_electrons(self):
        """Number of cluster electrons."""
        return 2*self.c_cluster_occ.shape[-1]

    def get_occupied_bath(self, *args, **kwargs):
        """Inherited bath classes can overwrite this to return additional occupied bath orbitals."""
        nao = self.mol.nao_nr()
        return np.zeros((nao, 0)), self.c_env_occ

    def get_virtual_bath(self, *args, **kwargs):
        """Inherited bath classes can overwrite this to return additional virtual bath orbitals."""
        nao = self.mol.nao_nr()
        return np.zeros((nao, 0)), self.c_env_vir

    def kernel(self):
        # --- DMET bath
        self.log.info("Making DMET Bath")
        self.log.info("----------------")
        self.log.changeIndentLevel(1)
        t0 = timer()
        c_dmet, n_dmet, c_env_occ, c_env_vir = self.make_dmet_bath(self.fragment.c_env)

        # --- Separate occupied and virtual in cluster
        cluster = [self.c_frag, c_dmet]
        self.base._check_orthonormal(*cluster, mo_name='cluster MO')
        c_cluster_occ, c_cluster_vir = self.fragment.diagonalize_cluster_dm(*cluster, tol=2*self.dmet_threshold)
        # Canonicalize
        c_cluster_occ = self.fragment.canonicalize_mo(c_cluster_occ)[0]
        c_cluster_vir = self.fragment.canonicalize_mo(c_cluster_vir)[0]
        if self.base.is_rhf:
            self.log.info("Cluster orbitals:  n(occ)= %3d  n(vir)= %3d", c_cluster_occ.shape[-1], c_cluster_vir.shape[-1])
        else:
            self.log.info("Alpha-cluster orbitals:  n(occ)= %3d  n(vir)= %3d", c_cluster_occ[0].shape[-1], c_cluster_vir[0].shape[-1])
            self.log.info(" Beta-cluster orbitals:  n(occ)= %3d  n(vir)= %3d", c_cluster_occ[1].shape[-1], c_cluster_vir[1].shape[-1])
        self.log.timing("Time for DMET bath:  %s", time_string(timer()-t0))
        self.log.changeIndentLevel(-1)

        self.c_dmet = c_dmet
        self.n_dmet = n_dmet
        self.c_env_occ = c_env_occ
        self.c_env_vir = c_env_vir
        self.c_cluster_occ = c_cluster_occ
        self.c_cluster_vir = c_cluster_vir

    def get_environment(self):
        return self.c_env_occ, self.c_env_vir

    def make_dmet_bath(self, c_env, dm1=None, c_ref=None, nbath=None, verbose=True, reftol=0.8):
        """Calculate DMET bath, occupied environment and virtual environment orbitals.

        If c_ref is not None, complete DMET orbital space using active transformation of reference orbitals.

        TODO:
        * reftol should not be necessary - just determine how many DMET bath orbital N are missing
        from C_ref and take the N largest eigenvalues over the combined occupied and virtual
        eigenvalues.

        Parameters
        ----------
        c_env : (n(AO), n(env)) array
            MO-coefficients of environment orbitals.
        dm1 : (n(AO), n(AO)) array, optional
            Mean-field one-particle reduced density matrix in AO representation. If None, `self.mf.make_rdm1()` is used.
            Default: None.
        c_ref : ndarray, optional
            Reference DMET bath orbitals from previous calculation.
        nbath : int, optional
            Number of DMET bath orbitals. If set, the parameter `tol` is ignored. Default: None.
        tol : float, optional
            Tolerance for DMET orbitals in eigendecomposition of density-matrix. Default: 1e-5.
        reftol : float, optional
            Tolerance for DMET orbitals in projection of reference orbitals.

        Returns
        -------
        c_bath : (n(AO), n(bath)) array
            DMET bath orbitals.
        eig : n(bath) array
            DMET orbital occupation numbers (in [0,1]).
        c_occenv : (n(AO), n(occ. env)) array
            Occupied environment orbitals.
        c_virenv : (n(AO), n(vir. env)) array
            Virtual environment orbitals.
        """

        # No environemnt -> no bath/environment orbitals
        if c_env.shape[-1] == 0:
            nao = c_env.shape[0]
            return np.zeros((nao, 0)), np.zeros((0,)), np.zeros((nao, 0)), np.zeros((nao, 0))

        tol = self.dmet_threshold

        # Divide by 2 to get eigenvalues in [0,1]
        sc = np.dot(self.base.get_ovlp(), c_env)
        if dm1 is None: dm1 = self.mf.make_rdm1()
        dm_env = dot(sc.T, dm1, sc) / 2
        try:
            eig, r = np.linalg.eigh(dm_env)
        except np.linalg.LinAlgError:
            eig, r = scipy.linalg.eigh(dm_env)
        # Sort: occ. env -> DMET bath -> vir. env
        eig, r = eig[::-1], r[:,::-1]
        if (eig.min() < -1e-8):
            self.log.error("Smallest eigenvalue of environment 1-DM below 0:  n= %.10e !", eig.min())
        if ((eig.max()-1) > 1e-8):
            self.log.error("Largest eigenvalue of environment 1-DM above 1:  n= %.10e !", eig.max())
        c_env = np.dot(c_env, r)
        c_env = fix_orbital_sign(c_env)[0]

        if nbath is not None:
            # FIXME
            raise NotImplementedError()
            # Work out tolerance which leads to nbath bath orbitals. This overwrites `tol`.
            abseig = abs(eig[np.argsort(abs(eig-0.5))])
            low, up = abseig[nbath-1], abseig[nbath]
            if abs(low - up) < 1e-14:
                raise RuntimeError("Degeneracy in env. DM does not allow for clear identification of %d bath orbitals!\nabs(eig)= %r"
                        % (nbath, abseig[:nbath+5]))
            tol = (low + up)/2
            self.log.debugv("Tolerance for %3d bath orbitals= %.8g", nbath, tol)

        mask_bath = np.logical_and(eig >= tol, eig <= 1-tol)
        mask_occenv = (eig > 1-tol)
        mask_virenv = (eig < tol)
        nbath = sum(mask_bath)

        noccenv = sum(mask_occenv)
        nvirenv = sum(mask_virenv)
        self.log.info("DMET bath:  n(Bath)= %4d  n(occ-Env)= %4d  n(vir-Env)= %4d", nbath, noccenv, nvirenv)
        assert (nbath + noccenv + nvirenv == c_env.shape[-1])
        c_bath = c_env[:,mask_bath].copy()
        c_occenv = c_env[:,mask_occenv].copy()
        c_virenv = c_env[:,mask_virenv].copy()

        if verbose:
            self.log_info(eig, c_env)
        n_dmet = eig[mask_bath]
        # Complete DMET orbital space using reference orbitals
        # NOT MAINTAINED!
        if c_ref is not None:
            c_bath, c_occenv, c_virenv = self.use_ref_orbitals(c_bath, c_occenv, c_virenv, c_ref, reftol)
        return c_bath, n_dmet, c_occenv, c_virenv

    def make_dmet_bath_fast(self, c_env, dm1=None):
        """Fast DMET orbitals.
        from Ref. J. Chem. Phys. 151, 064108 (2019); https://doi.org/10.1063/1.5108818

        Problem: How to get C_occenv and C_virenv without N^3 diagonalization?
        """

        ovlp = self.base.get_ovlp()
        c_occ = self.base.mo_coeff_occ
        ca, cb = self.c_frag, c_env
        ra = dot(c_occ.T, ovlp, ca)
        rb = dot(c_occ.T, ovlp, cb)
        d11 = np.dot(ra.T, ra)
        ea, ua = np.linalg.eigh(d11)
        if (ea.min() < -1e-9):
            self.log.error("Min eigenvalue of frag. DM = %.6e !", ea.min())
        if ((ea.max()-1) > 1e-9):
            self.log.error("Max eigenvalue of frag. DM = %.6e !", ea.max())
        # Fragment singular values:
        ea = np.clip(ea, 0, 1)
        sa = np.sqrt(ea)
        d21 = np.dot(rb.T, ra)
        ub = np.dot(d21, ua)
        sab = np.linalg.norm(ub, axis=0)
        sb = sab/sa
        mask_bath = (sb**2 >= self.dmet_threshold)
        ub = ub[:,mask_bath]
        # In AO basis
        c_bath = np.dot(cb, ub/sab[mask_bath])
        return c_bath

    def log_info(self, eig, c_env, threshold=1e-10):
        tol = self.dmet_threshold
        mask = np.logical_and(eig >= threshold, eig <= 1-threshold)
        ovlp = self.base.get_ovlp()
        maxocc = 2 if self.base.spinsym == 'restricted' else 1
        if np.any(mask):
            self.log.info("Mean-field entangled orbitals:")
            self.log.info("      Bath  Occupation  Entanglement  Character")
            self.log.info("      ----  ----------  ------------  ------------------------------------------------------")
            for idx, e in enumerate(eig[mask]):
                bath = 'Yes' if (tol <= e <= 1-tol) else 'No'
                entang = 4*e*(1-e)
                # Mulliken population of DMET orbital:
                pop = einsum('a,b,ba->a', c_env[:,mask][:,idx], c_env[:,mask][:,idx], ovlp)
                sort = np.argsort(-pop)
                pop = pop[sort]
                labels = np.asarray(self.mol.ao_labels(None))[sort][:min(len(pop), 4)]
                char = ', '.join('%s %s%s (%.0f%%)' % (*(l[1:]), 100*pop[i]) for (i,l) in enumerate(labels))
                self.log.info("  %2d  %4s  %10.3g  %12.3g  %s", idx+1, bath, e*maxocc, entang, char)
        # Calculate entanglement entropy
        mask_bath = np.logical_and(eig >= tol, eig <= 1-tol)
        entropy = np.sum(eig * (1-eig))
        entropy_bath = np.sum(eig[mask_bath] * (1-eig[mask_bath]))
        self.log.info("Entanglement entropy: total= %.3e  bath= %.3e (%.2f %%)",
                entropy, entropy_bath, 100*entropy_bath/entropy)

    def use_ref_orbitals(self, c_bath, c_occenv, c_virenv, c_ref, reftol=0.8):
        """Not maintained!"""
        nref = c_ref.shape[-1]
        self.log.debug("%d reference DMET orbitals given.", nref)
        nmissing = nref - nbath

        # DEBUG
        _, eig = self.project_ref_orbitals(c_ref, c_bath)
        self.log.debug("Eigenvalues of reference orbitals projected into DMET bath:\n%r", eig)

        if nmissing == 0:
            self.log.debug("Number of DMET orbitals equal to reference.")
        elif nmissing > 0:
            # Perform the projection separately for occupied and virtual environment space
            # Otherwise, it is not guaranteed that the additional bath orbitals are
            # fully (or very close to fully) occupied or virtual.
            # --- Occupied
            C_occenv, eig = self.project_ref_orbitals(c_ref, c_occenv)
            mask_occref = eig >= reftol
            mask_occenv = eig < reftol
            self.log.debug("Eigenvalues of projected occupied reference: %s", eig[mask_occref])
            if np.any(mask_occenv):
                self.log.debug("Largest remaining: %s", max(eig[mask_occenv]))
            # --- Virtual
            c_virenv, eig = self.project_ref_orbitals(c_ref, c_virenv)
            mask_virref = eig >= reftol
            mask_virenv = eig < reftol
            self.log.debug("Eigenvalues of projected virtual reference: %s", eig[mask_virref])
            if np.any(mask_virenv):
                self.log.debug("Largest remaining: %s", max(eig[mask_virenv]))
            # -- Update coefficient matrices
            c_bath = np.hstack((c_bath, c_occenv[:,mask_occref], c_virenv[:,mask_virref]))
            c_occenv = c_occenv[:,mask_occenv].copy()
            c_virenv = c_virenv[:,mask_virenv].copy()
            nbath = C_bath.shape[-1]
            self.log.debug("New number of occupied environment orbitals: %d", c_occenv.shape[-1])
            self.log.debug("New number of virtual environment orbitals: %d", c_virenv.shape[-1])
            if nbath != nref:
                err = "Number of DMET bath orbitals=%d not equal to reference=%d" % (nbath, nref)
                self.log.critical(err)
                raise RuntimeError(err)
        else:
            err = "More DMET bath orbitals found than in reference!"
            self.log.critical(err)
            raise RuntimeError(err)
        return c_bath, c_occenv, c_virenv


class DMET_Bath_UHF(DMET_Bath_RHF):

    def get_cluster_electrons(self):
        """Number of (alpha, beta) cluster electrons."""
        return (self.c_cluster_occ[0].shape[-1] + self.c_cluster_occ[1].shape[-1])

    def get_occupied_bath(self, *args, **kwargs):
        """Inherited bath classes can overwrite this to return additional occupied bath orbitals."""
        nao = self.mol.nao_nr()
        return np.zeros((2, nao, 0)), self.c_env_occ

    def get_virtual_bath(self, *args, **kwargs):
        """Inherited bath classes can overwrite this to return additional virtual bath orbitals."""
        nao = self.mol.nao_nr()
        return np.zeros((2, nao, 0)), self.c_env_vir

    def make_dmet_bath(self, c_env, dm1=None, **kwargs):
        if dm1 is None: dm1 = self.mf.make_rdm1()
        results = []
        for s, spin in enumerate(('alpha', 'beta')):
            self.log.info("Making %s-DMET bath", spin)
            # Use restricted DMET bath routine for each spin:
            results.append(super().make_dmet_bath(c_env[s], dm1=2*dm1[s], **kwargs))
        return tuple(zip(*results))
