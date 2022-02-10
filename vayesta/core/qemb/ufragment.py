import dataclasses

import numpy as np

from vayesta.core.util import *

from .fragment import Fragment
from vayesta.core.symmetry import tsymmetry

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
        return '%s(id= %d, name= %s, n_frag= (%d, %d), n_elec= (%.8f, %.8f), sym_factor= %f)' % (self.__class__.__name__,
                self.id, self.name, *self.n_frag, *self.nelectron, self.sym_factor)

    @staticmethod
    def stack_mo(*mo_coeff):
        """Stack MOs in each spin channel."""
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

    # Amplitude projection
    # --------------------

    # NEW:

    def get_occ2frag_projector(self):
        ovlp = self.base.get_ovlp()
        projector = (dot(self.c_proj[0].T, ovlp, self.cluster.c_active_occ[0]),
                     dot(self.c_proj[1].T, ovlp, self.cluster.c_active_occ[1]))
        return projector

    def project_amp1_to_fragment(self, amp1, projector=None):
        """Can be used to project C1, T1, or L1 amplitudes."""
        if projector is None:
            projector = self.get_occ2frag_projector()
        return (np.dot(projector[0], amp1[0]),
                np.dot(projector[1], amp1[1]))

    def project_amp2_to_fragment(self, amp2, projector=None, axis=0):
        """Can be used to project C2, T2, or L2 amplitudes."""
        if projector is None:
            projector = self.get_occ2frag_projector()
        if axis == 0:
            caa, cab, cbb = amp2
            caax = einsum('xi,i...->x...', projector[0], caa)
            cabx = einsum('xi,i...->x...', projector[0], cab)
            cbax = einsum('xj,ij...->ix...', projector[1], cab)
            cbbx = einsum('xi,i...->x...', projector[1], cbb)
            return (caax, cabx, cbax, cbbx)
        if axis == 1:
            raise NotImplementedError()
        raise ValueError("axis needs to be 0 or 1")

    # OLD:

    def get_fragment_projector(self, coeff, c_proj=None, **kwargs):
        if c_proj is None: c_proj = self.c_proj
        projectors = []
        for s in range(2):
            projectors.append(super().get_fragment_projector(coeff[s], c_proj=c_proj[s], **kwargs))
        return tuple(projectors)

    def project_amplitude_to_fragment(self, c, c_occ=None, c_vir=None, partition=None, symmetrize=True):
        if c_occ is None: c_occ = self.cluster.c_active_occ
        if c_vir is None: c_vir = self.cluster.c_active_vir
        if partition is None: partition = self.opts.wf_partition
        if partition != 'first-occ': raise NotImplementedError()

        # Two projectors: (alpha, beta)
        p_occ = self.get_fragment_projector(c_occ)

        if np.ndim(c[0]) == 2:
            ca, cb = c
            ca = np.dot(p_occ[0], ca)
            cb = np.dot(p_occ[1], cb)
            return (ca, cb)
        if np.ndim(c[0]) == 4:
            caa, cab, cbb = c
            caa = np.tensordot(p_occ[0], caa, axes=1)
            cab = np.tensordot(p_occ[0], cab, axes=1)
            cbb = np.tensordot(p_occ[1], cbb, axes=1)
            if symmetrize:
                # Symmetrize projection between first index (alpha) and second index (beta)
                cab = (cab + einsum('xj,ijab->ixab', p_occ[1], c[1])) / 2
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

    def get_fragment_dmet_energy(self, dm1=None, dm2=None, h1e_eff=None, eris=None):
        """Get fragment contribution to whole system DMET energy.

        After fragment summation, the nuclear-nuclear repulsion must be added to get the total energy!

        Parameters
        ----------
        dm1: array, optional
            Cluster one-electron reduced density-matrix in cluster basis. If `None`, `self.results.dm1` is used. Default: None.
        dm2: array, optional
            Cluster two-electron reduced density-matrix in cluster basis. If `None`, `self.results.dm2` is used. Default: None.
        eris: array, optional
            Cluster electron-repulsion integrals in cluster basis. If `None`, the ERIs are reevaluated. Default: None.

        Returns
        -------
        e_dmet: float
            Electronic fragment DMET energy.
        """
        if dm1 is None: dm1 = self.results.dm1
        if dm2 is None: dm2 = self.results.dm2
        if dm1 is None: raise RuntimeError("DM1 not found for %s" % self)
        if dm2 is None: raise RuntimeError("DM2 not found for %s" % self)
        c_act = self.cluster.c_active
        t0 = timer()
        if eris is None:
            with log_time(self.log.timing, "Time for AO->MO transformation: %s"):
                eris = self.base.get_eris_array(c_act)
        elif isinstance(eris, tuple) and len(eris) == 3:
            pass
        else:
            #TODO
            raise NotImplementedError()
            #elif not isinstance(eris, np.ndarray):
            #    self.log.debugv("Extracting ERI array from CCSD ERIs object.")
            #    eris = vayesta.core.ao2mo.helper.get_full_array(eris, c_act)
        dm1a, dm1b = dm1
        dm2aa, dm2ab, dm2bb = dm2
        gaa, gab, gbb = eris

        # Get effective core potential
        if h1e_eff is None:
            # Use the original Hcore (without chemical potential modifications), but updated mf-potential!
            h1e_eff = self.base.get_hcore_orig() + self.base.get_veff(with_exxdiv=False)
            h1e_eff = (dot(c_act[0].T, h1e_eff[0], c_act[0]),
                       dot(c_act[1].T, h1e_eff[1], c_act[1]))
            oa = np.s_[:self.cluster.nocc_active[0]]
            ob = np.s_[:self.cluster.nocc_active[1]]
            va = (einsum('iipq->pq', gaa[oa,oa,:,:]) + einsum('pqii->pq', gab[:,:,ob,ob])
                - einsum('ipqi->pq', gaa[oa,:,:,oa]))
            vb = (einsum('iipq->pq', gbb[ob,ob,:,:]) + einsum('iipq->pq', gab[oa,oa,:,:])
                - einsum('ipqi->pq', gbb[ob,:,:,ob]))
            h1e_eff = (h1e_eff[0]-va, h1e_eff[1]-vb)

        p_frag = self.get_fragment_projector(c_act)
        # Check number of electrons
        nea = einsum('ix,ij,jx->', p_frag[0], dm1a, p_frag[0])
        neb = einsum('ix,ij,jx->', p_frag[1], dm1b, p_frag[1])
        self.log.info("Number of local electrons for DMET energy: %.8f %.8f", nea, neb)

        # Evaluate energy
        e1b = (einsum('xj,xi,ij->', h1e_eff[0], p_frag[0], dm1a)
             + einsum('xj,xi,ij->', h1e_eff[1], p_frag[1], dm1b))
        e2b = (einsum('xjkl,xi,ijkl->', gaa, p_frag[0], dm2aa)
             + einsum('xjkl,xi,ijkl->', gbb, p_frag[1], dm2bb)
             + einsum('xjkl,xi,ijkl->', gab, p_frag[0], dm2ab)/2
             + einsum('ijxl,xk,ijkl->', gab, p_frag[1], dm2ab)/2)
        e_dmet = self.opts.sym_factor*(e1b + e2b)
        self.log.debug("Fragment E(DMET)= %+16.8f Ha", e_dmet)
        self.log.timing("Time for DMET energy: %s", time_string(timer()-t0))
        return e_dmet

    # --- Symmetry
    # ============

    def add_tsymmetric_fragments(self, tvecs, unit='Ang', charge_tol=1e-6, spin_tol=1e-6):
        """

        Parameters
        ----------
        tvecs: (3,3) float array or (3,) integer array
            Translational symmetry vectors. If an array with shape (3,3) is passed, each row represents
            a translation vector in cartesian coordinates, in units defined by the parameter `unit`.
            If an array with shape (3,) is passed, each element represent the number of
            translation vector corresponding to the a0, a1, and a2 lattice vectors of the cell.
        unit: ['Ang', 'Bohr'], optional
            Units of translation vectors. Only used if a (3, 3) array is passed. Default: 'Ang'.
        charge_tol: float, optional
            Tolerance for the error of the mean-field density matrix between symmetry related fragments.
            If the largest absolute difference in the density-matrix is above this value,
            and exception will be raised. Default: 1e-6.
        spin_tol: float, optional
            Tolerance for the error of the mean-field density matrix between symmetry related fragments.
            If the largest absolute difference in the density-matrix is above this value,
            and exception will be raised. Default: 1e-6.

        Returns
        -------
        fragments: list
            List of T-symmetry related fragments. These will be automatically added to base.fragments and
            have the attributes `sym_parent` and `sym_op` set.
        """
        #if self.boundary_cond == 'open': return []

        ovlp = self.base.get_ovlp()
        dm1 = self.mf.make_rdm1()

        fragments = []
        for (dx, dy, dz), tvec in tsymmetry.loop_tvecs(self.mol, tvecs, unit=unit):

            sym_op = tsymmetry.get_tsymmetry_op(self.mol, tvec, unit='Bohr')
            if sym_op is None:
                self.log.error("No T-symmetric fragment found for translation (%d,%d,%d) of fragment %s", dx, dy, dz, self.name)
                continue
            # Name for translationally related fragments
            name = '%s_T(%d,%d,%d)' % (self.name, dx, dy, dz)
            # Translated coefficients
            c_frag_t = sym_op(self.c_frag)
            c_env_t = sym_op(self.c_env)

            # Check that translated fragment does not overlap with current fragment:
            fragovlp = max(abs(dot(self.c_frag[0].T, ovlp, c_frag_t[0])).max(),
                           abs(dot(self.c_frag[1].T, ovlp, c_frag_t[1])).max())
            if (fragovlp > 1e-8):
                self.log.error("Translation (%d,%d,%d) of fragment %s not orthogonal to original fragment (overlap= %.3e)!",
                            dx, dy, dz, self.name, fragovlp)
            # Deprecated:
            if hasattr(self.base, 'add_fragment'):  # pragma: no cover
                frag = self.base.add_fragment(name, c_frag_t, c_env_t, options=self.opts,
                        sym_parent=self, sym_op=sym_op)
            else:
                frag_id = self.base.register.get_next_id()
                frag = self.base.Fragment(self.base, frag_id, name, c_frag_t, c_env_t, options=self.opts,
                        sym_parent=self, sym_op=sym_op, mpi_rank=self.mpi_rank)
                self.base.fragments.append(frag)

            # Check symmetry
            charge_err, spin_err = self.get_tsymmetry_error(frag, dm1=dm1)
            if charge_err > charge_tol or spin_err > spin_tol:
                self.log.critical("Mean-field DM not symmetric for translation (%d,%d,%d) of %s (charge error= %.3e spin error= %.3e)!",
                    dx, dy, dz, self.name, charge_err, spin_err)
                raise RuntimeError("MF not symmetric under translation (%d,%d,%d)" % (dx, dy, dz))
            else:
                self.log.debugv("Mean-field DM symmetry error for translation (%d,%d,%d) of %s charge= %.3e spin= %.3e",
                    dx, dy, dz, self.name, charge_err, spin_err)

            fragments.append(frag)

        return fragments

    def get_tsymmetry_error(self, frag, dm1=None):
        """Get translational symmetry error between two fragments."""
        if dm1 is None: dm1 = self.mf.make_rdm1()
        dma, dmb = dm1
        ovlp = self.base.get_ovlp()
        # This fragment (x)
        cxa, cxb = self.stack_mo(self.c_frag, self.c_env)
        dmxa = dot(cxa.T, ovlp, dma, ovlp, cxa)
        dmxb = dot(cxb.T, ovlp, dmb, ovlp, cxb)
        # Other fragment (y)
        cya, cyb = self.stack_mo(frag.c_frag, frag.c_env)
        dmya = dot(cya.T, ovlp, dma, ovlp, cya)
        dmyb = dot(cyb.T, ovlp, dmb, ovlp, cyb)
        charge_err = abs(dmxa+dmxb-dmya-dmyb).max()
        spin_err = abs(dmxa-dmxb-dmya+dmyb).max()
        return (charge_err, spin_err)


    # --- Overlap matrices
    # --------------------

    def get_overlap_m2c(self):
        """Get rotation matrices from occupied/virtual active space to MF orbitals."""
        ovlp = self.base.get_ovlp()
        r_occ_a = dot(self.base.mo_coeff_occ[0].T, ovlp, self.cluster.c_active_occ[0])
        r_occ_b = dot(self.base.mo_coeff_occ[1].T, ovlp, self.cluster.c_active_occ[1])
        r_vir_a = dot(self.base.mo_coeff_vir[0].T, ovlp, self.cluster.c_active_vir[0])
        r_vir_b = dot(self.base.mo_coeff_vir[1].T, ovlp, self.cluster.c_active_vir[1])
        return (r_occ_a, r_occ_b), (r_vir_a, r_vir_b)

    def get_overlap_m2f(self):
        """Get overlap matrices from mean-field to fragment orbitals."""
        ovlp = self.base.get_ovlp()
        r_occ_a = dot(self.base.mo_coeff_occ[0].T, ovlp, self.c_proj[0])
        r_occ_b = dot(self.base.mo_coeff_occ[1].T, ovlp, self.c_proj[1])
        r_vir_a = dot(self.base.mo_coeff_vir[0].T, ovlp, self.c_proj[0])
        r_vir_b = dot(self.base.mo_coeff_vir[1].T, ovlp, self.c_proj[1])
        return (r_occ_a, r_occ_b), (r_vir_a, r_vir_b)
