import os
import os.path
from datetime import datetime

import numpy as np

import pyscf

from .qemb import QEmbedding
from .ufragment import UFragment

from vayesta.core.ao2mo.postscf_ao2mo import postscf_ao2mo
from vayesta.core.util import *

# Amplitudes
from .amplitudes import get_t1_uhf
from .amplitudes import get_t2_uhf
from .amplitudes import get_t12_uhf

# Density-matrices
from .urdm import make_rdm1_demo

class UEmbedding(QEmbedding):
    """Spin unrestricted quantum embedding."""

    # Shadow this in inherited methods:
    Fragment = UFragment

    def init_vhf(self):
        if self.opts.recalc_vhf:
            self.log.debug("Recalculating HF potential from MF object.")
            return None
        self.log.debug("Determining HF potential from MO energies and coefficients.")
        cs = einsum('...ai,ab->...ib', self.mo_coeff, self.get_ovlp())
        fock = einsum('...ia,...i,...ib->ab', cs, self.mo_energy, cs)
        return (fock - self.get_hcore())

    @staticmethod
    def stack_mo(*mo_coeff):
        mo_coeff = (hstack(*[c[0] for c in mo_coeff]),
                    hstack(*[c[1] for c in mo_coeff]))
        return mo_coeff

    @property
    def nmo(self):
        """Total number of molecular orbitals (MOs)."""
        return (self.mo_coeff[0].shape[-1],
                self.mo_coeff[1].shape[-1])

    @property
    def nocc(self):
        """Number of occupied MOs."""
        return (np.count_nonzero(self.mo_occ[0] > 0),
                np.count_nonzero(self.mo_occ[1] > 0))

    @property
    def nvir(self):
        """Number of virtual MOs."""
        return (np.count_nonzero(self.mo_occ[0] == 0),
                np.count_nonzero(self.mo_occ[1] == 0))

    @property
    def mo_coeff_occ(self):
        """Occupied MO coefficients."""
        return (self.mo_coeff[0][:,:self.nocc[0]],
                self.mo_coeff[1][:,:self.nocc[1]])

    @property
    def mo_coeff_vir(self):
        """Virtual MO coefficients."""
        return (self.mo_coeff[0][:,self.nocc[0]:],
                self.mo_coeff[1][:,self.nocc[1]:])

    def check_orthonormal(self, *mo_coeff, mo_name='', **kwargs):
        mo_coeff = self.stack_mo(*mo_coeff)
        results = []
        for s, spin in enumerate(('alpha', ' beta')):
            name_s = '-'.join([spin, mo_name])
            res_s = super().check_orthonormal(mo_coeff[s], mo_name=name_s, **kwargs)
            results.append(res_s)
        return tuple(zip(*results))

    def get_exxdiv(self):
        """Get divergent exact-exchange (exxdiv) energy correction and potential.

        Returns
        -------
        e_exxdiv: float
            Divergent exact-exchange energy correction per unit cell.
        v_exxdiv: array
            Divergent exact-exchange potential correction in AO basis.
        """
        if not self.has_exxdiv: return 0, None
        ovlp = self.get_ovlp()
        sca = np.dot(ovlp, self.mo_coeff[0][:,:self.nocc[0]])
        scb = np.dot(ovlp, self.mo_coeff[1][:,:self.nocc[1]])
        madelung = pyscf.pbc.tools.madelung(self.mol, self.mf.kpt)
        e_exxdiv = -madelung * (self.nocc[0]+self.nocc[1]) / (2*self.ncells)
        v_exxdiv_a = -madelung * np.dot(sca, sca.T)
        v_exxdiv_b = -madelung * np.dot(scb, scb.T)
        self.log.debug("Divergent exact-exchange (exxdiv) correction= %+16.8f Ha", e_exxdiv)
        return e_exxdiv, (v_exxdiv_a, v_exxdiv_b)

    # TODO:

    def get_eris_array(self, mo_coeff, compact=False):
        """Get electron-repulsion integrals in MO basis as a NumPy array.

        Parameters
        ----------
        mo_coeff: (n(AO), n(MO)) array
            MO coefficients.

        Returns
        -------
        eris: (n(MO), n(MO), n(MO), n(MO)) array
            Electron-repulsion integrals in MO basis.
        """
        # TODO: check self.kdf and fold
        #t0 = timer()
        #if hasattr(self.mf, 'with_df') and self.mf.with_df is not None:
        #    eris_aa = self.mf.with_df.ao2mo(mo_coeff[0], compact=compact)
        #    eris_bb = self.mf.with_df.ao2mo(mo_coeff[1], compact=compact)
        #    eris_ab = self.mf.with_df.ao2mo((mo_coeff[0], mo_coeff[0], mo_coeff[1], mo_coeff[1]), compact=compact)
        #elif self.mf._eri is not None:
        #    eris = pyscf.ao2mo.full(self.mf._eri, mo_coeff[0], compact=compact)
        #else:
        #    eris = self.mol.ao2mo(mo_coeff, compact=compact)
        #if not compact:
        #    eris = eris.reshape(4*[mo_coeff.shape[-1]])
        #self.log.timing("Time for AO->MO of ERIs:  %s", time_string(timer()-t0))
        #return eris
        self.log.debugv("Making alpha-alpha ERIs...")
        eris_aa = super().get_eris_array(mo_coeff[0], compact=compact)
        self.log.debugv("Making beta-beta ERIs...")
        eris_bb = super().get_eris_array(mo_coeff[1], compact=compact)
        self.log.debugv("Making alpha-beta ERIs...")
        eris_ab = super().get_eris_array(2*[mo_coeff[0]] + 2*[mo_coeff[1]], compact=compact)
        return (eris_aa, eris_ab, eris_bb)

    def get_eris_object(self, posthf):
        """Get ERIs for post-HF methods.

        For folded PBC calculations, this folds the MO back into k-space
        and contracts with the k-space three-center integrals..

        Parameters
        ----------
        posthf: one of the following post-HF methods: MP2, CCSD, RCCSD, DFCCSD
            Post-HF method with attribute mo_coeff set.

        Returns
        -------
        eris: _ChemistsERIs
            ERIs which can be used for the respective post-HF method.
        """
        t0 = timer()
        #c_act = _mo_without_core(posthf, posthf.mo_coeff)
        active = posthf.get_frozen_mask()
        c_act = (posthf.mo_coeff[0][:,active[0]], posthf.mo_coeff[1][:,active[1]])
        if isinstance(posthf, pyscf.mp.mp2.MP2):
            fock = self.get_fock()
        elif isinstance(posthf, (pyscf.ci.cisd.CISD, pyscf.cc.ccsd.CCSD)):
            fock = self.get_fock(with_exxdiv=False)
        else:
            raise ValueError("Unknown post-HF method: %r", type(posthf))
        mo_energy = (einsum('ai,ab,bi->i', c_act[0], self.get_fock()[0], c_act[0]),
                     einsum('ai,ab,bi->i', c_act[1], self.get_fock()[1], c_act[1]))
        e_hf = self.mf.e_tot

        # 1) Fold MOs into k-point sampled primitive cell, to perform efficient AO->MO transformation:
        if self.kdf is not None:
            raise NotImplementedError()
            #eris = gdf_to_pyscf_eris(self.mf, self.kdf, posthf, fock=fock, mo_energy=mo_energy, e_hf=e_hf)
            #return eris
        # 2) Regular AO->MO transformation
        eris = postscf_ao2mo(posthf, fock=fock, mo_energy=mo_energy, e_hf=e_hf)
        self.log.timing("Time for AO->MO of ERIs:  %s", time_string(timer()-t0))
        return eris

    def update_mf(self, mo_coeff, mo_energy=None, veff=None):
        """Update underlying mean-field object."""
        # Chech orthonormal MOs
        if not (np.allclose(dot(mo_coeff[0].T, self.get_ovlp(), mo_coeff[0]) - np.eye(mo_coeff[0].shape[-1]), 0)
            and np.allclose(dot(mo_coeff[1].T, self.get_ovlp(), mo_coeff[1]) - np.eye(mo_coeff[1].shape[-1]), 0)):
                raise ValueError("MO coefficients not orthonormal!")
        self.mf.mo_coeff = mo_coeff
        dm = self.mf.make_rdm1(mo_coeff=mo_coeff)
        if veff is None:
            veff = self.mf.get_veff(dm=dm)
        self.set_veff(veff)
        if mo_energy is None:
            # Use diagonal of Fock matrix as MO energies
            fock = self.get_fock()
            mo_energy = (einsum('ai,ab,bi->i', mo_coeff[0], fock[0], mo_coeff[0]),
                         einsum('ai,ab,bi->i', mo_coeff[1], fock[1], mo_coeff[1]))
        self.mf.mo_energy = mo_energy
        self.mf.e_tot = self.mf.energy_tot(dm=dm, h1e=self.get_hcore(), vhf=veff)

    def check_fragment_symmetry(self, dm1, charge_tol=1e-6, spin_tol=1e-6):
        frags = self.get_symmetry_child_fragments(include_parents=True)
        for group in frags:
            parent, children = group[0], group[1:]
            for child in children:
                charge_err, spin_err = parent.get_tsymmetry_error(child, dm1=dm1)
                if (charge_err > charge_tol) or (spin_err > spin_tol):
                    raise RuntimeError("%s and %s not symmetric: charge error= %.3e spin error= %.3e !"
                            % (parent.name, child.name, charge_err, spin_err))
                self.log.debugv("Symmetry between %s and %s: charge error= %.3e spin error= %.3e", parent.name, child.name, charge_err, spin_err)

    # --- CC Amplitudes
    # -----------------

    get_t1 = get_t1_uhf
    get_t2 = get_t2_uhf
    get_t12 = get_t12_uhf

    # --- Density-matrices
    # --------------------

    make_rdm1_demo = make_rdm1_demo

    # TODO:
    def make_rdm1_ccsd(self, *args, **kwargs):
        raise NotImplementedError()

    def make_rdm2_demo(self, *args, **kwargs):
        raise NotImplementedError()

    def make_rdm2_ccsd(self, *args, **kwargs):
        raise NotImplementedError()

    # --- Other
    # ---------

    def pop_analysis(self, dm1, mo_coeff=None, write=True, **kwargs):
        pop = []
        for s, spin in enumerate(('alpha', 'beta')):
            mo = mo_coeff[s] if mo_coeff is not None else None
            pop.append(super().pop_analysis(dm1[s], mo_coeff=mo, write=False, **kwargs))
        pop = tuple(pop)
        if write:
            self.write_population(pop, **kwargs)
        return pop

    def get_atomic_charges(self, pop):
        charges = np.zeros(self.mol.natm)
        spins = np.zeros(self.mol.natm)
        for i, label in enumerate(self.mol.ao_labels(fmt=None)):
            charges[label[0]] -= (pop[0][i] + pop[1][i])
            spins[label[0]] += (pop[0][i] - pop[1][i])
        charges += self.mol.atom_charges()
        return charges, spins
