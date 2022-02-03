import numpy as np
import scipy
import scipy.linalg

from vayesta.core.util import *

DEFAULT_DMET_THRESHOLD = 1e-6

class DMET_Bath:

    spin_restricted = True

    def __init__(self, fragment, dmet_threshold=DEFAULT_DMET_THRESHOLD):
        self.fragment = fragment
        self.dmet_threshold = dmet_threshold
        # Output
        self.c_dmet = None
        self.c_cluster_occ = None
        self.c_cluster_vir = None
        self.c_env_occ = None
        self.c_env_vir = None

    @property
    def spin_unrestricted(self):
        return not self.spin_restricted

    @property
    def mf(self):
        return self.fragment.mf

    @property
    def mol(self):
        return self.fragment.mol

    @property
    def log(self):
        return self.fragment.log

    @property
    def base(self):
        return self.fragment.base

    @property
    def c_frag(self):
        return self.fragment.c_frag

    def get_dmet_bath(self):
        return self.c_dmet

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
        c_dmet, c_env_occ, c_env_vir = self.make_dmet_bath(self.fragment.c_env)

        # --- Separate occupied and virtual in cluster
        cluster = [self.c_frag, c_dmet]
        self.base.check_orthonormal(*cluster, mo_name='cluster MO')
        c_cluster_occ, c_cluster_vir = self.fragment.diagonalize_cluster_dm(*cluster, tol=2*self.dmet_threshold)
        if self.base.is_rhf:
            self.log.info("Cluster orbitals:  n(occ)= %3d  n(vir)= %3d", c_cluster_occ.shape[-1], c_cluster_vir.shape[-1])
        else:
            self.log.info("Alpha-cluster orbitals:  n(occ)= %3d  n(vir)= %3d", c_cluster_occ[0].shape[-1], c_cluster_vir[0].shape[-1])
            self.log.info(" Beta-cluster orbitals:  n(occ)= %3d  n(vir)= %3d", c_cluster_occ[1].shape[-1], c_cluster_vir[1].shape[-1])
        self.log.timing("Time for DMET bath:  %s", time_string(timer()-t0))
        self.log.changeIndentLevel(-1)

        self.c_dmet = c_dmet
        self.c_env_occ = c_env_occ
        self.c_env_vir = c_env_vir
        self.c_cluster_occ = c_cluster_occ
        self.c_cluster_vir = c_cluster_vir

    def get_environment(self):
        return self.c_env_occ, self.c_env_vir

    def make_dmet_bath(self, c_env, dm1=None, c_ref=None, nbath=None, dmet_threshold=None, verbose=True, reftol=0.8):
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
        c_occenv : (n(AO), n(occ. env)) array
            Occupied environment orbitals.
        c_virenv : (n(AO), n(vir. env)) array
            Virtual environment orbitals.
        """

        # No environemnt -> no bath/environment orbitals
        if c_env.shape[-1] == 0:
            nao = c_env.shape[0]
            return np.zeros((nao, 0)), np.zeros((nao, 0)), np.zeros((nao, 0))

        if dmet_threshold is None: dmet_threshold = self.dmet_threshold
        tol = dmet_threshold

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
        if (eig.min() < -1e-9):
            self.log.error("Min eigenvalue of env. DM = %.6e !", eig.min())
        if ((eig.max()-1) > 1e-9):
            self.log.error("Max eigenvalue of env. DM = %.6e !", eig.max())
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
            # Orbitals in [print_tol, 1-print_tol] will be printed (even if they don't fall in the DMET tol range)
            print_tol = 1e-10
            # DMET bath orbitals with eigenvalue in [strong_tol, 1-strong_tol] are printed as strongly entangled
            strong_tol = 0.1
            limits = [print_tol, tol, strong_tol, 1-strong_tol, 1-tol, 1-print_tol]
            if np.any(np.logical_and(eig > limits[0], eig <= limits[-1])):
                names = [
                        "Unentangled vir. env. orbital",
                        "Weakly-entangled vir. bath orbital",
                        "Strongly-entangled bath orbital",
                        "Weakly-entangled occ. bath orbital",
                        "Unentangled occ. env. orbital",
                        ]
                self.log.info("Non-(0 or 1) eigenvalues (n) of environment DM:")
                for i, e in enumerate(eig):
                    name = None
                    for j, llim in enumerate(limits[:-1]):
                        ulim = limits[j+1]
                        if (llim < e and e <= ulim):
                            name = names[j]
                            break
                    if name:
                        self.log.info("  > %-34s  n= %12.6g  1-n= %12.6g  n*(1-n)= %12.6g", name, e, 1-e, e*(1-e))

            # DMET bath analysis
            self.log.info("DMET bath character:")
            for i in range(c_bath.shape[-1]):
                ovlp = einsum('a,b,ba->a', c_bath[:,i], c_bath[:,i], self.base.get_ovlp())
                sort = np.argsort(-ovlp)
                ovlp = ovlp[sort]
                n = np.amin((len(ovlp), 6))     # Get the six largest overlaps
                labels = np.asarray(self.mol.ao_labels())[sort][:n]
                lines = [('%s= %.5f' % (labels[i].strip(), ovlp[i])) for i in range(n)]
                self.log.info("  > %2d:  %s", i+1, '  '.join(lines))

        # Calculate entanglement entropy
        entropy = np.sum(eig * (1-eig))
        entropy_bath = np.sum(eig[mask_bath] * (1-eig[mask_bath]))
        self.log.info("Entanglement entropy: total= %.6e  bath= %.6e (%.2f %%)",
                entropy, entropy_bath, 100.0*entropy_bath/entropy)

        # Complete DMET orbital space using reference orbitals
        # NOT MAINTAINED!
        if c_ref is not None:  # pragma: no cover
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


class CompleteBath(DMET_Bath):
    """Complete bath for testing purposes."""

    def get_occupied_bath(self, *args, **kwargs):
        nao = self.c_env_occ.shape[0]
        return self.c_env_occ, np.zeros((nao, 0))

    def get_virtual_bath(self, *args, **kwargs):
        nao = self.c_env_vir.shape[0]
        return self.c_env_vir, np.zeros((nao, 0))

# --- Spin unrestricted

class UDMET_Bath(DMET_Bath):

    spin_restricted = False

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

class UCompleteBath(UDMET_Bath):
    """Complete bath for testing purposes."""

    def get_occupied_bath(self, *args, **kwargs):
        nao = self.c_env_occ[0].shape[0]
        return self.c_env_occ, tuple(2*[np.zeros((nao, 0))])

    def get_virtual_bath(self, *args, **kwargs):
        nao = self.c_env_vir[0].shape[0]
        return self.c_env_vir, tuple(2*[np.zeros((nao, 0))])
