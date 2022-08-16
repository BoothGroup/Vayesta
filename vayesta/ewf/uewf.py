import numpy as np

from vayesta.core.qemb import UEmbedding
from vayesta.core.util import *

from vayesta.ewf import REWF
from vayesta.ewf.ufragment import Fragment
from vayesta.core.fragmentation import SAO_Fragmentation
from vayesta.core.fragmentation import IAOPAO_Fragmentation
from vayesta.misc import corrfunc
from vayesta.mpi import mpi

# Amplitudes
from .amplitudes import get_global_t1_uhf
from .amplitudes import get_global_t2_uhf
# Density-matrices
from .urdm import make_rdm1_ccsd
from .urdm import make_rdm1_ccsd_global_wf
from .urdm import make_rdm2_ccsd_global_wf
from .urdm import make_rdm2_ccsd_proj_lambda
from .icmp2 import get_intercluster_mp2_energy_uhf


class UEWF(REWF, UEmbedding):

    Fragment = Fragment

    # --- CC Amplitudes
    # -----------------

    # T-amplitudes
    get_global_t1 = get_global_t1_uhf
    get_global_t2 = get_global_t2_uhf

    def t1_diagnostic(self, warn_tol=0.02):
        # Per cluster
        for f in self.get_fragments(active=True, mpi_rank=mpi.rank):
            t1 = f.results.t1
            if t1 is None:
                self.log.error("No T1 amplitudes found for %s.", f)
                continue
            nelec = t1[0].shape[0] + t1[1].shape[0]
            t1diag = (np.linalg.norm(t1[0]) / np.sqrt(nelec),
                      np.linalg.norm(t1[1]) / np.sqrt(nelec))
            if max(t1diag) > warn_tol:
                self.log.warning("T1 diagnostic for %-20s alpha= %.5f beta= %.5f", str(f)+':', *t1diag)
            else:
                self.log.info("T1 diagnostic for %-20s alpha= %.5f beta= %.5f", str(f)+':', *t1diag)
        # Global
        t1 = self.get_global_t1(mpi_target=0)
        if mpi.is_master:
            nelec = t1[0].shape[0] + t1[1].shape[0]
            t1diag = (np.linalg.norm(t1[0]) / np.sqrt(nelec),
                      np.linalg.norm(t1[1]) / np.sqrt(nelec))
            if max(t1diag) > warn_tol:
                self.log.warning("Global T1 diagnostic: alpha= %.5f beta= %.5f", *t1diag)
            else:
                self.log.info("Global T1 diagnostic: alpha= %.5f beta= %.5f", *t1diag)


    # --- Density-matrices
    # --------------------

    # DM1

    @log_method()
    def _make_rdm1_mp2(self, *args, **kwargs):
        return make_rdm1_ccsd(self, *args, mp2=True, **kwargs)

    @log_method()
    def _make_rdm1_ccsd(self, *args, **kwargs):
        return make_rdm1_ccsd(self, *args, mp2=False, **kwargs)

    @log_method()
    def _make_rdm1_ccsd_global_wf(self, *args, ao_basis=False, with_mf=True, **kwargs):
        dm1a, dm1b = self._make_rdm1_ccsd_global_wf_cached(*args, **kwargs)
        if with_mf:
            dm1a[np.diag_indices(self.nocc[0])] += 1
            dm1b[np.diag_indices(self.nocc[1])] += 1
        if ao_basis:
            dm1a = dot(self.mo_coeff[0], dm1a, self.mo_coeff[0].T)
            dm1b = dot(self.mo_coeff[1], dm1b, self.mo_coeff[1].T)
        return (dm1a, dm1b)

    @cache(copy=True)
    def _make_rdm1_ccsd_global_wf_cached(self, *args, **kwargs):
        return make_rdm1_ccsd_global_wf(self, *args, **kwargs)

    @log_method()
    def _make_rdm1_ccsd_proj_lambda(self, *args, **kwargs):
        raise NotImplementedError()

    # DM2

    @log_method()
    def _make_rdm2_ccsd_global_wf(self, *args, **kwargs):
        return make_rdm2_ccsd_global_wf(self, *args, **kwargs)

    @log_method()
    def _make_rdm2_ccsd_proj_lambda(self, *args, **kwargs):
        return make_rdm2_ccsd_proj_lambda(self, *args, **kwargs)

    get_intercluster_mp2_energy = get_intercluster_mp2_energy_uhf

    # --- Other expectation values

    @deprecated(replacement='get_corrfunc_mf')
    def get_atomic_ssz_mf(self, dm1=None, atoms=None, projection='sao'):
        """TODO: update similar to restricted code
            dm1 in MO basis"""
        if dm1 is None:
            dm1a = np.zeros((self.nmo[0], self.nmo[0]))
            dm1b = np.zeros((self.nmo[1], self.nmo[1]))
            dm1a[np.diag_indices(self.nocc[0])] = 1
            dm1b[np.diag_indices(self.nocc[1])] = 1
            dm1 = (dm1a, dm1b)
        c_atom = self._get_atomic_coeffs(atoms=atoms, projection=projection)
        natom = len(c_atom)
        ovlp = self.get_ovlp()
        # Get projectors
        proj = []
        for a in range(natom):
            ra = dot(self.mo_coeff[0].T, ovlp, c_atom[a][0])
            rb = dot(self.mo_coeff[1].T, ovlp, c_atom[a][1])
            pa = np.dot(ra, ra.T)
            pb = np.dot(rb, rb.T)
            proj.append((pa, pb))
        ssz = np.zeros((natom, natom))
        for a in range(natom):
            for b in range(natom):
                ssz[a,b] = corrfunc.spinspin_z_unrestricted(dm1, None, proj1=proj[a], proj2=proj[b])
        return ssz

    @log_method()
    @deprecated(replacement='get_corrfunc')
    def get_atomic_ssz(self, dm1=None, dm2=None, atoms=None, projection='sao', dm2_with_dm1=None):
        """Get expectation values <P(A) S_z^2 P(B)>, where P(X) are projectors onto atoms.

        TODO: MPI"""
        # --- Setup
        if dm2_with_dm1 is None:
            dm2_with_dm1 = False
            if dm2 is not None:
                # Determine if DM2 contains DM1 by calculating norm
                norm_aa = einsum('iikk->', dm2[0])
                norm_ab = einsum('iikk->', dm2[1])
                norm_bb = einsum('iikk->', dm2[2])
                norm = norm_aa + norm_bb + 2*norm_ab
                ne2 = self.mol.nelectron*(self.mol.nelectron-1)
                dm2_with_dm1 = (norm > ne2/2)
        if atoms is None:
            atoms = list(range(self.mol.natm))
        natom = len(atoms)
        c_atom = self._get_atomic_coeffs(atoms=atoms, projection=projection)
        ovlp = self.get_ovlp()

        proj = []
        for a in range(natom):
            rxa = dot(self.mo_coeff[0].T, ovlp, c_atom[a][0])
            rxb = dot(self.mo_coeff[1].T, ovlp, c_atom[a][1])
            pxa = np.dot(rxa, rxa.T)
            pxb = np.dot(rxb, rxb.T)
            proj.append((pxa, pxb))
        # Fragment dependent projection operator:
        if dm2 is None:
            proj_x = []
            for x in self.get_fragments(active=True):
                tmpa = np.dot(x.cluster.c_active[0].T, ovlp)
                tmpb = np.dot(x.cluster.c_active[1].T, ovlp)
                proj_x.append([])
                for a in range(natom):
                    rxa = np.dot(tmpa, c_atom[a][0])
                    rxb = np.dot(tmpb, c_atom[a][1])
                    pxa = np.dot(rxa, rxa.T)
                    pxb = np.dot(rxb, rxb.T)
                    proj_x[-1].append((pxa, pxb))

        ssz = np.zeros((natom, natom))
        # 1-DM contribution:
        if dm1 is None:
            dm1 = self.make_rdm1()
        dm1a, dm1b = dm1
        for a in range(natom):
            tmpa = np.dot(proj[a][0], dm1a)
            tmpb = np.dot(proj[a][1], dm1b)
            for b in range(natom):
                ssz[a,b] = (np.sum(tmpa*proj[b][0])
                          + np.sum(tmpb*proj[b][1]))/4

        occa = np.s_[:self.nocc[0]]
        occb = np.s_[:self.nocc[1]]
        occdiaga = np.diag_indices(self.nocc[0])
        occdiagb = np.diag_indices(self.nocc[1])
        # Non-cumulant DM2 contribution:
        if not dm2_with_dm1:
            ddm1a, ddm1b = dm1a.copy(), dm1b.copy()
            ddm1a[occdiaga] -= 0.5
            ddm1b[occdiagb] -= 0.5
            # Traces of projector*DM(HF)
            trpa = [np.trace(p[0][occa,occa]) for p in proj]
            trpb = [np.trace(p[1][occb,occb]) for p in proj]
            # Traces of projector*[DM(CC) + DM(HF)/2]
            trda = [np.sum(p[0] * ddm1a) for p in proj]
            trdb = [np.sum(p[1] * ddm1b) for p in proj]
            for a in range(natom):
                tmpa = np.dot(proj[a][0], ddm1a)
                tmpb = np.dot(proj[a][1], ddm1b)
                for b in range(natom):
                    ssz[a,b] -= (np.sum(tmpa[occa] * proj[b][0][occa])
                               + np.sum(tmpb[occb] * proj[b][1][occb]))/2
                    # Note that this contribution cancel to 0 in RHF,
                    # since trpa == trpb and trda == trdb:
                    ssz[a,b] += ((trpa[a]*trda[b] + trpa[b]*trda[a])      # alpha-alpha
                               - (trpa[a]*trdb[b] + trpb[b]*trda[a])      # alpha-beta
                               - (trpb[a]*trda[b] + trpa[b]*trdb[a])      # beta-alpha
                               + (trpb[a]*trdb[b] + trpb[b]*trdb[a]))/4   # beta-beta

        if dm2 is not None:
            dm2aa, dm2ab, dm2bb = dm2
            for a in range(natom):
                pa = proj[a]
                tmpa = (np.tensordot(pa[0], dm2aa) - np.tensordot(dm2ab, pa[1]))
                tmpb = (np.tensordot(pa[1], dm2bb) - np.tensordot(pa[0], dm2ab))
                for b in range(natom):
                    pb = proj[b]
                    ssz[a,b] += (np.sum(tmpa*pb[0]) + np.sum(tmpb*pb[1]))/4
        else:
            # Cumulant DM2 contribution:
            for ix, x in enumerate(self.get_fragments(active=True)):
                dm2aa, dm2ab, dm2bb = x.make_fragment_dm2cumulant()
                for a in range(natom):
                    pa = proj_x[ix][a]
                    tmpa = (np.tensordot(pa[0], dm2aa) - np.tensordot(dm2ab, pa[1]))
                    tmpb = (np.tensordot(pa[1], dm2bb) - np.tensordot(pa[0], dm2ab))
                    for b in range(natom):
                        pb = proj_x[ix][b]
                        ssz[a,b] += (np.sum(tmpa*pb[0]) + np.sum(tmpb*pb[1]))/4
        return ssz

    def _get_dm_corr_energy_old(self, global_dm1=True, global_dm2=False, t_as_lambda=None):
        """Calculate correlation energy from reduced density-matrices.

        Parameters
        ----------
        global_dm1 : bool
            Use 1DM calculated from global amplitutes if True, otherwise use in cluster approximation. Default: True.
        global_dm2 : bool
            Use 2DM calculated from global amplitutes if True, otherwise use in cluster approximation. Default: False.

        Returns
        -------
        e_corr : float
            Correlation energy.
        """
        if t_as_lambda is None:
            t_as_lambda = self.opts.t_as_lambda
        if global_dm1:
            dm1a, dm1b = self._make_rdm1_ccsd_global_wf(t_as_lambda=t_as_lambda, with_mf=False)
        else:
            dm1a, dm1b = self._make_rdm1_ccsd(t_as_lambda=t_as_lambda, with_mf=False)

        # --- Core Hamiltonian + Non-cumulant 2DM contribution
        fa, fb = self.get_fock_for_energy(with_exxdiv=False)
        e1 = (einsum('pi,pq,qj,ij->', self.mo_coeff[0], fa, self.mo_coeff[0], dm1a)
            + einsum('pi,pq,qj,ij->', self.mo_coeff[1], fb, self.mo_coeff[1], dm1b))/self.ncells

        # --- Cumulant 2-DM contribution
        # Use global 2-DM
        if global_dm2:
            dm2aa, dm2ab, dm2bb = self._make_rdm2_ccsd_global_wf(t_as_lambda=t_as_lambda, with_dm1=False)
            eriaa = self.get_eris_array(self.mo_coeff[0])
            e2 = einsum('pqrs,pqrs', eriaa, dm2aa) / 2
            eriab = self.get_eris_array(2*[self.mo_coeff[0]] + 2*[self.mo_coeff[1]])
            e2 += einsum('pqrs,pqrs', eriab, dm2ab)
            eribb = self.get_eris_array(self.mo_coeff[1])
            e2 += einsum('pqrs,pqrs', eribb, dm2bb) / 2
        # Use fragment-local 2-DM
        else:
            e2 = self.get_dm_corr_energy_e2(t_as_lambda=t_as_lambda)
        e_corr = (e1 + e2)
        return e_corr
