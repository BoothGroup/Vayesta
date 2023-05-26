
import numpy as np

from vayesta.core.util import dot, einsum, energy_string, log_time, time_string, timer, with_doc
from vayesta.core import spinalg
from vayesta.core.qemb.fragment import Fragment


class UFragment(Fragment):

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
        mo_coeff = spinalg.hstack_matrices(*mo_coeff)
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
        mo_coeff = spinalg.hstack_matrices(*mo_coeff)
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
        mo_coeff = spinalg.hstack_matrices(*mo_coeff)
        if dm1 is None: dm1 = self.mf.make_rdm1()
        results = []
        for s, spin in enumerate(('alpha', 'beta')):
            res_s = super().diagonalize_cluster_dm(mo_coeff[s], dm1=dm1[s], norm=norm, **kwargs)
            results.append(res_s)
        return tuple(zip(*results))

    # Amplitude projection
    # --------------------

    def get_fragment_projector(self, coeff, c_proj=None, **kwargs):
        if c_proj is None: c_proj = self.c_proj
        projectors = []
        for s in range(2):
            projectors.append(super().get_fragment_projector(coeff[s], c_proj=c_proj[s], **kwargs))
        return tuple(projectors)

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

    def get_fragment_mo_energy(self, c_active=None, fock=None):
        """Returns approximate MO energies, using the the diagonal of the Fock matrix.

        Parameters
        ----------
        c_active: array, optional
        fock: array, optional
        """
        if c_active is None: c_active = self.cluster.c_active
        if fock is None: fock = self.base.get_fock()
        mo_energy_a = einsum('ai,ab,bi->i', c_active[0], fock[0], c_active[0])
        mo_energy_b = einsum('ai,ab,bi->i', c_active[1], fock[1], c_active[1])
        return (mo_energy_a, mo_energy_b)

    @with_doc(Fragment.get_fragment_dmet_energy)
    def get_fragment_dmet_energy(self, dm1=None, dm2=None, h1e_eff=None, eris=None, part_cumulant=True, approx_cumulant=True):
        if dm1 is None: dm1 = self.results.dm1
        if dm1 is None: raise RuntimeError("DM1 not found for %s" % self)
        c_act = self.cluster.c_active
        t0 = timer()
        if eris is None:
            eris = self._eris
        if eris is None:
            with log_time(self.log.timingv, "Time for AO->MO transformation: %s"):
                gaa, gab, gbb = self.base.get_eris_array_uhf((c_act[0], c_act[1]))
        elif isinstance(eris, tuple) and len(eris) == 3:
            gaa, gab, gbb = eris
        else:
            # TODO: Extract integrals from CCSD ERIs object
            # Temporary solution:
            with log_time(self.log.timingv, "Time for AO->MO transformation: %s"):
                gaa, gab, gbb = self.base.get_eris_array_uhf((c_act[0], c_act[1]))
        if dm2 is None:
            dm2 = self.results.wf.make_rdm2(with_dm1=not part_cumulant, approx_cumulant=approx_cumulant)
        dm1a, dm1b = dm1
        dm2aa, dm2ab, dm2bb = dm2

        # Get effective core potential
        if h1e_eff is None:
            if part_cumulant:
                h1e_eff = self.base.get_hcore_for_energy()
                h1e_eff = (dot(c_act[0].T, h1e_eff, c_act[0]),
                           dot(c_act[1].T, h1e_eff, c_act[1]))
            else:
                # Use the original Hcore (without chemical potential modifications), but updated mf-potential!
                h1e_eff = self.base.get_hcore_for_energy() + self.base.get_veff_for_energy(with_exxdiv=False)/2
                h1e_eff = (dot(c_act[0].T, h1e_eff[0], c_act[0]),
                           dot(c_act[1].T, h1e_eff[1], c_act[1]))
                oa = np.s_[:self.cluster.nocc_active[0]]
                ob = np.s_[:self.cluster.nocc_active[1]]
                va = (einsum('iipq->pq', gaa[oa,oa,:,:]) + einsum('pqii->pq', gab[:,:,ob,ob])
                    - einsum('ipqi->pq', gaa[oa,:,:,oa]))/2
                vb = (einsum('iipq->pq', gbb[ob,ob,:,:]) + einsum('iipq->pq', gab[oa,oa,:,:])
                    - einsum('ipqi->pq', gbb[ob,:,:,ob]))/2
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
             + einsum('xjkl,xi,ijkl->', gab, p_frag[0], dm2ab)
             + einsum('ijxl,xk,ijkl->', gab, p_frag[1], dm2ab))/2
        self.log.debugv("E(DMET): E(1)= %s E(2)= %s", energy_string(e1b), energy_string(e2b))
        e_dmet = self.opts.sym_factor*(e1b + e2b)
        self.log.debug("Fragment E(DMET)= %+16.8f Ha", e_dmet)
        self.log.timing("Time for DMET energy: %s", time_string(timer()-t0))
        return e_dmet

    # --- Symmetry
    # ============

    def get_symmetry_error(self, frag, dm1=None):
        """Get translational symmetry error between two fragments."""
        if dm1 is None:
            dm1 = self.mf.make_rdm1()
        dma, dmb = dm1
        ovlp = self.base.get_ovlp()
        # This fragment (x)
        cxa, cxb = spinalg.hstack_matrices(self.c_frag, self.c_env)
        dmxa = dot(cxa.T, ovlp, dma, ovlp, cxa)
        dmxb = dot(cxb.T, ovlp, dmb, ovlp, cxb)
        # Other fragment (y)
        if frag.c_env is None:
            cy_env = frag.sym_op(self.c_env)
        else:
            cy_env = frag.c_env
        cya, cyb = spinalg.hstack_matrices(frag.c_frag, cy_env)
        dmya = dot(cya.T, ovlp, dma, ovlp, cya)
        dmyb = dot(cyb.T, ovlp, dmb, ovlp, cyb)
        charge_err = abs(dmxa+dmxb-dmya-dmyb).max()
        spin_err = abs(dmxa-dmxb-dmya+dmyb).max()
        return charge_err, spin_err

    # --- Overlap matrices
    # --------------------

    def _csc_dot(self, c1, c2, ovlp=True, transpose_left=True, transpose_right=False):
        if transpose_left:
            c1 = (c1[0].T, c1[1].T)
        if transpose_right:
            c2 = (c2[0].T, c2[1].T)
        if ovlp is True:
            ovlp = self.base.get_ovlp()
        if ovlp is None:
            outa = dot(c1[0], c2[0])
            outb = dot(c1[1], c2[1])
            return (outa, outb)
        outa = dot(c1[0], ovlp, c2[0])
        outb = dot(c1[1], ovlp, c2[1])
        return (outa, outb)
