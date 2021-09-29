import dataclasses

import numpy as np

from vayesta.core.util import *

from .fragment import Fragment

class UFragment(Fragment):

    @dataclasses.dataclass
    class Result(Fragment.Results):
        pass

    def log_info(self):
        # Some output
        fmt = '  > %-24s     '
        self.log.info(fmt+'%d , %d', "Fragment orbitals:", *self.n_frag)
        self.log.info(fmt+'%f', "Symmetry factor:", self.sym_factor)
        self.log.info(fmt+'%.10f , %.10f', "Number of electrons:", *self.nelectron)
        if self.atoms is not None:
            self.log.info(fmt+'%r', "Associated atoms:", self.atoms)
        if self.aos is not None:
            self.log.info(fmt+'%r', "Associated AOs:", self.aos)

    def __repr__(self):
        return '%s(id= %d, name= %s, n_frag= (%d, %d), n_elec= (%d, %d), sym_factor= %f)' % (self.__class__.__name__,
                self.id, self.name, *self.n_frag, *self.nelectron, self.sym_factor)

    @staticmethod
    def stack_mo(*mo_coeff):
        mo_coeff = (hstack(*[c[0] for c in mo_coeff]),
                    hstack(*[c[1] for c in mo_coeff]))
        return mo_coeff

    @property
    def n_frag(self):
        """Number of fragment orbitals."""
        return (self.c_frag[0].shape[-1],
                self.c_frag[1].shape[-1])

    @property
    def nelectron(self):
        """Number of mean-field electrons."""
        ovlp = self.base.get_ovlp()
        dm = self.mf.make_rdm1()
        ne = []
        for s in range(2):
            sc = np.dot(ovlp, self.c_frag[s])
            ne.append(einsum('ai,ab,bi->', sc, dm[s], sc))
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

    def get_mo_occupation(self, *mo_coeff, dm1=None, **kwargs):
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
        if dm1 is None: dm1 = self.mf.make_rdm1()
        results = []
        for s, spin in enumerate(('alpha', 'beta')):
            results.append(super().get_mo_occupation(mo_coeff[s], dm1=dm1[s], **kwargs))
        return results

    def canonicalize_mo(self, *mo_coeff, fock=None, **kwargs):
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
        if fock is None: fock = self.base.get_fock()
        mo_coeff = self.stack_mo(*mo_coeff)
        results = []
        for s, spin in enumerate(('alpha', 'beta')):
            results.append(super().canonicalize_mo(mo_coeff[s], fock=fock[s], **kwargs))
        return tuple(zip(*results))

    def diagonalize_cluster_dm(self, *mo_coeff, dm1=None, norm=1, **kwargs):
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
        if dm1 is None: dm1 = self.mf.make_rdm1()
        results = []
        for s, spin in enumerate(('alpha', 'beta')):
            res_s = super().diagonalize_cluster_dm(mo_coeff[s], dm1=dm1[s], norm=norm, **kwargs)
            results.append(res_s)
        return tuple(zip(*results))

    #def make_dmet_bath(self, c_env, dm1=None, **kwargs):
    #    if dm1 is None: dm1 = self.mf.make_rdm1()
    #    results = []
    #    for s, spin in enumerate(('alpha', 'beta')):
    #        self.log.info("Making %s-DMET bath", spin)
    #        # Use restricted DMET bath routine for each spin:
    #        results.append(super().make_dmet_bath(c_env[s], dm1=2*dm1[s], **kwargs))
    #    return tuple(zip(*results))

    def get_fragment_projector(self, coeff, c_proj=None, **kwargs):
        if c_proj is None: c_proj = self.c_proj
        projectors = []
        for s in range(2):
            projectors.append(super().get_fragment_projector(coeff[s], c_proj=c_proj[s], **kwargs))
        return tuple(projectors)

    def project_amplitude_to_fragment(self, c, c_occ=None, c_vir=None, partition=None, symmetrize=True):
        if c_occ is None: c_occ = self.c_active_occ
        if c_vir is None: c_vir = self.c_active_vir
        if partition is None: partition = self.opts.wf_partition
        if partition != 'first-occ': raise NotImplementedError()

        p_occ = self.get_fragment_projector(c_occ)

        if np.ndim(c[0]) == 2:
            ca = np.dot(p_occ[0], c[0])
            cb = np.dot(p_occ[1], c[1])
            return (ca, cb)
        if np.ndim(c[0]) == 4:
            caa = np.tensordot(p_occ[0], c[0], axes=1)
            cab = np.tensordot(p_occ[0], c[1], axes=1)
            #cab = (np.tensordot(p_occ[0], c[1], axes=1) + einsum('xj,ijab->ixab', p_occ[1], c[1]))/2
            cbb = np.tensordot(p_occ[1], c[1], axes=1)
            if symmetrize:
                caa = (caa + caa.transpose(1,0,3,2))/2
                cbb = (cbb + cbb.transpose(1,0,3,2))/2
            return (caa, cab, cbb)

    def get_fragment_mf_energy(self):
        """Calculate the part of the mean-field energy associated with the fragment.

        Does not include nuclear-nuclear repulsion!
        """
        pa, pb = self.get_fragment_projector(self.base.mo_coeff)
        hveff = (dot(pa, self.base.mo_coeff[0].T, self.base.get_hcore()+self.base.get_veff()[0]/2, self.base.mo_coeff[0]),
                 dot(pb, self.base.mo_coeff[1].T, self.base.get_hcore()+self.base.get_veff()[1]/2, self.base.mo_coeff[1]))
        occ = ((self.base.mo_occ[0] > 0), (self.base.mo_occ[1] > 0))
        e_mf = np.sum(np.diag(hveff[0])[occ[0]]) + np.sum(np.diag(hveff[1])[occ[1]])
        return e_mf
