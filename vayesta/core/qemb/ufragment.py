import numpy as np

from vayesta.core.util import *

from .fragment import QEmbeddingFragment

class UQEmbeddingFragment(QEmbeddingFragment):

    @staticmethod
    def stack_mo(*mo_coeff):
        mo_coeff = (np.hstack([c[0] for c in mo_coeff]),
                    np.hstack([c[1] for c in mo_coeff]))
        return mo_coeff

    @property
    def size(self):
        """Number of fragment orbitals."""
        return (self.c_frag[0].shape[-1],
                self.c_frag[1].shape[-1])

    @property
    def nelectron(self):
        """Number of mean-field electrons."""
        ovlp = self.base.get_ovlp()
        dm_ao = self.mf.make_rdm1()
        ne = []
        for s in range(2):
            sc = einsum('ab,bi->ai', ovlp, self.c_frag[s])
            ne.append(dot(sc.T, dm_ao[s], sc))
        return tuple(ne)

    @property
    def n_active(self):
        """Number of active orbitals."""
        return (self.n_active_occ[0] + self.n_active_vir[0],
                self.n_active_occ[1] + self.n_active_vir[1])

    @property
    def n_active_occ(self):
        """Number of active occupied orbitals."""
        return (self.c_active_occ[0].shape[-1],
                self.c_active_occ[1].shape[-1])

    @property
    def n_active_vir(self):
        """Number of active virtual orbitals."""
        return (self.c_active_vir[0].shape[-1],
                self.c_active_vir[1].shape[-1])

    @property
    def n_frozen(self):
        """Number of frozen orbitals."""
        return (self.n_frozen_occ[0] + self.n_frozen_vir[0],
                self.n_frozen_occ[1] + self.n_frozen_vir[1])

    @property
    def n_frozen_occ(self):
        """Number of frozen occupied orbitals."""
        return (self.c_frozen_occ[0].shape[-1],
                self.c_frozen_occ[1].shape[-1])

    @property
    def n_frozen_vir(self):
        """Number of frozen virtual orbitals."""
        return (self.c_frozen_vir[0].shape[-1],
                self.c_frozen_vir[1].shape[-1])

    def get_mo_occupation(self, *mo_coeff):
        """Get mean-field occupation numbers (diagonal of 1-RDM) of orbitals.

        Parameters
        ----------
        mo_coeff : ndarray, shape(N, M)
            Orbital coefficients.

        Returns
        -------
        occ : ndarray, shape(M)
            Occupation numbers of orbitals.
        """
        mo_coeff = self.stack_mo(*mo_coeff)
        ovlp = self.base.get_ovlp()
        dm_ao = self.mf.make_rdm1()
        occ = []
        for s in range(2):
            sc = np.dot(ovlp, mo_coeff[s])
            occ.append(einsum('ai,ab,bi->i', sc, dm[s], sc))
        return tuple(occ)

    def canonicalize_mo(self, *mo_coeff, eigvals=False, sign_convention=True):
        """Diagonalize Fock matrix within subspace.

        Parameters
        ----------
        *mo_coeff : ndarrays
            Orbital coefficients.
        eigenvalues : ndarray
            Return MO energies of canonicalized orbitals.

        Returns
        -------
        mo_canon : ndarray
            Canonicalized orbital coefficients.
        rot : ndarray
            Rotation matrix: np.dot(mo_coeff, rot) = mo_canon.
        """
        mo_coeff = self.stack_mo(*mo_coeff)
        f_ao = self.base.get_fock()
        results = []
        for s in range(2):
            fock = dot(mo_coeff[s].T, f_ao[s], mo_coeff[s])
            mo_energy, rot = np.linalg.eigh(fock)
            mo_can = np.dot(mo_coeff[s], rot)
            if sign_convention:
                mo_can, signs = helper.orbital_sign_convention(mo_can)
                rot = rot*signs[np.newaxis]
            assert np.allclose(np.dot(mo_coeff, rot), mo_can)
            assert np.allclose(np.dot(mo_can, rot.T), mo_coeff)
            if eigvals:
                results.append((mo_can, rot, mo_energy))
            else:
                results.append((mo_can, rot))
            resu
        return tuple(zip(*results))

    def diagonalize_cluster_dm(self, *mo_coeff, tol=1e-4):
        """Diagonalize cluster (fragment+bath) DM to get fully occupied and virtual orbitals.

        Parameters
        ----------
        *mo_coeff : ndarrays
            Orbital coefficients.
        tol : float, optional
            If set, check that all eigenvalues of the cluster DM are close
            to 0 or 1, with the tolerance given by tol. Default= 1e-4.

        Returns
        -------
        c_cluster_occ : ndarray
            Occupied cluster orbitals.
        c_cluster_vir : ndarray
            Virtual cluster orbitals.
        """
        mo_coeff = self.stack_mo(*mo_coeff)
        ovlp = self.base.get_ovlp()
        dm_ao = self.mf.make_rdm1()
        c_cluster_occ = []
        c_cluster_vir = []
        for s in range(2):
            sc = np.dot(ovlp, mo_coeff[s])
            dm = dot(sc.T, dm_ao[s], sc) 
            e, v = np.linalg.eigh(dm)
            if tol and not np.allclose(np.fmin(abs(e), abs(e-1)), 0, atol=tol, rtol=0):
                raise RuntimeError("Error while diagonalizing cluster DM: eigenvalues not all close to 0 or 1:\n%s" % e)
            e, v = e[::-1], v[:,::-1]
            c_cluster = np.dot(mo_coeff, v)
            nocc = sum(e >= 0.5)
            c_occ_s, c_vir_s = np.hsplit(c_cluster, [nocc])
            c_cluster_occ.append(c_occ_s)
            c_cluster_vir.append(c_vir_s)
        return tuple(c_cluster_occ), tuple(c_cluster_vir)

    def make_dmet_bath(self, c_env, dm1=None, **kwargs):
        if dm1 is None: dm1 = self.mf.make_rdm1()
        results = []
        for s, spin in enumerate(('alpha', 'beta')):
            self.log("Making %s-DMET bath", spin)
            # Use restricted DMET bath routine for each spin:
            results.append(super().make_dmet_bath(c_env[s], dm1=2*dm1[s], **kwargs))
        return tuple(zip(*results))

    def get_fragment_projector(self, coeff, **kwargs):
        projectors = []
        for s in range(2):
            projectors.append(super().get_fragment_projector(coeff[s], **kwargs))
        return tuple(projectors)
