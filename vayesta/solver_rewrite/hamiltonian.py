import numpy as np
import scipy.linalg

from vayesta.core.util import *
import dataclasses
import pyscf.scf
from typing import Optional
from vayesta.core.types import Orbitals
from vayesta.core.qemb import scrcoulomb
from vayesta.rpa import ssRPA

def ClusterHamiltonian(fragment, mf, log=None, **kwargs):
    rhf = np.ndim(mf.mo_coeff[0]) == 1
    eb = hasattr(fragment, "bos_freqs")
    if rhf:
        if eb:
            return EB_RClusterHamiltonian(fragment, mf, log=log, **kwargs)
        else:
            return RClusterHamiltonian(fragment, mf, log=log, **kwargs)
    if eb:
        return EB_UClusterHamiltonian(fragment, mf, log=log, **kwargs)
    return UClusterHamiltonian(fragment, mf, log=log, **kwargs)


class RClusterHamiltonian:
    @dataclasses.dataclass
    class Options(OptionsBase):
        screening: Optional[str] = None

    @property
    def _scf_class(self):
        return pyscf.scf.RHF

    def __init__(self, fragment, mf, log=None, **kwargs):

        self.mf = mf
        # Do we want to populate all parameters at initialisation, so fragment isn't actually saved here?
        # This would be more of a computational
        self._fragment = fragment
        self.log = (log or fragment.log)
        # --- Options:
        self.opts = self.Options()
        self.opts.update(**kwargs)
        self.log.info("Parameters of %s:" % self.__class__.__name__)
        self.log.info(break_into_lines(str(self.opts), newline='\n    '))
        self.v_ext = None
        self._seris = None
        self.seris_initialised = self.opts.screening is None

    @property
    def cluster(self):
        return self._fragment.cluster

    @property
    def mo(self):
        return Orbitals(self.cluster.c_active, occ=self.cluster.nocc_active)

    @property
    def nelec(self):
        return (self.cluster.nocc_active, self.cluster.nocc_active)

    @property
    def ncas(self):
        return (self.cluster.norb_active, self.cluster.norb_active)

    def target_space_projector(self, c=None):
        """Projector to the target fragment space within our cluster."""
        if c is None:
            c = self.cluster.c_active
        return self._fragment.get_fragment_projector(c)

    def get_fock(self, with_vext=True):
        c = self.cluster.c_active
        fock = dot(c.T, self.mf.get_fock(), c)
        if with_vext and self.v_ext is not None:
            fock += self.v_ext
        return fock

    def get_clus_mf_info(self, ao_basis=False):
        if ao_basis:
            nao = self.cluster.c_active.shape[1]
        else:
            nao = self.ncas
        mo_energy = np.diag(self.get_fock())
        mo_occ = np.zeros_like(mo_energy)
        mo_occ[:self.nelec[0]] = 2.0
        # Determine whether we want our cluster orbitals expressed in the basis of active orbitals, or in the AO basis.
        if ao_basis:
            mo_coeff = self.cluster.c_active
            ovlp = self.mf.get_ovlp()
        else:
            mo_coeff = np.eye(self.ncas[0])
            ovlp = np.eye(self.ncas[0])
        return nao, mo_coeff, mo_energy, mo_occ, ovlp

    def get_integrals(self, bare_eris=None, with_vext=True):
        heff = self.get_heff(bare_eris, with_vext=with_vext)
        # Depending on calculation this may be the same as bare_eris
        seris = self.get_eris()
        return heff, seris

    def get_heff(self, eris=None, fock=None, with_vext=True):
        if eris is None:
            eris = self.get_eris_bare()
        if fock is None:
            fock = self.get_fock()
        occ = np.s_[:self.cluster.nocc_active]
        v_act = 2 * einsum('iipq->pq', eris[occ, occ]) - einsum('iqpi->pq', eris[occ, :, :, occ])
        h_eff = fock - v_act
        if with_vext and self.v_ext is not None:
            h_eff += self.v_ext
        return h_eff

    def get_eris(self):
        # This will only return the bare eris if no screening is expected
        if self.opts.screening is None:
            return self.get_eris_bare()
        if self._seris is None:
            self.log.critical("ERIs requested before expected screened interactions have been initialised.")
            raise RuntimeError("ERIs requested before expected screened interactions have been initialised.")
        return self._seris

    def get_eris_bare(self):
        with log_time(self.log.timing, "Time for AO->MO of ERIs:  %s"):
            coeff = self.cluster.c_active
            eris = self._fragment.base.get_eris_array(coeff)
        return eris

    def to_pyscf_mf(self, force_bare_eris=False, overwrite_fock=False):
        # Using this requires equal spin channels.
        self.assert_equal_spin_channels()
        nao, mo_coeff, mo_energy, mo_occ, ovlp = self.get_clus_mf_info(ao_basis=False)

        clusmol = self.mf.mol.__class__()
        clusmol.nelec = self.nelec
        # NB if the number of alpha and beta active orbitals is different then will likely need to ensure the `ao2mo`
        # of pyscf approaches is replaced in Vayesta to support this.
        # If we wanted to actually run a HF calculation would need to replace `scf.energy_elec()` to support
        # spin-dependent output to `get_hcore()`.

        clusmol.nao = self.ncas[0]
        clusmol.build()

        bare_eris = self.get_eris_bare()

        heff = self.get_heff(eris=bare_eris, with_vext=True)

        clusmf = self._scf_class(clusmol)
        clusmf.get_hcore = lambda *args, **kwargs: heff
        clusmf.get_ovlp = lambda *args, **kwargs: ovlp
        if overwrite_fock:
            clusmf.get_fock = lambda *args, **kwargs: self.get_fock(with_vext=True)
            clusmf.get_veff = lambda *args, **kwargs: np.array(clusmf.get_fock(*args, **kwargs)) - np.array(clusmf.get_hcore())
        # This could be replaced by a density fitted approach if we wanted.
        if force_bare_eris:
            clusmf._eri = bare_eris
        else:
            clusmf._eri = self.get_eris()
        clusmf.mo_coeff = mo_coeff
        clusmf.mo_occ = mo_occ
        clusmf.mo_energy = mo_energy
        return clusmf

    def calc_loc_erpa(self):

        clusmf = self.to_pyscf_mf(force_bare_eris=True)
        clusrpa = ssRPA(clusmf)
        M, AmB, ApB, eps, v = clusrpa._gen_arrays()
        erpa = clusrpa.kernel()
        m0 = clusrpa.gen_moms(0)

        def get_product_projector():
            nocc = self.nelec
            nvir = tuple([x - y for x, y in zip(self.ncas, self.nelec)])
            p_occ_frag = self.target_space_projector(self.cluster.c_active_occ)

            if (not isinstance(p_occ_frag, tuple)) and np.ndim(p_occ_frag) == 2:
                p_occ_frag = (p_occ_frag, p_occ_frag)

            def get_product_projector(p_o, p_v, no, nv):
                return einsum("ij,ab->iajb", p_o, p_v).reshape((no*nv, no*nv))
            pa = get_product_projector(p_occ_frag[0], np.eye(nvir[0]), nocc[0], nvir[0])
            pb = get_product_projector(p_occ_frag[1], np.eye(nvir[1]), nocc[1], nvir[1])
            return scipy.linalg.block_diag(pa, pb)

        proj = get_product_projector()
        eloc = 0.5 * einsum("pq,qr,rp->", proj, m0, ApB) - einsum("pq,qp->", proj, ApB + AmB)
        return eloc

    def add_screening(self, seris_intermed=None):
        """Add screened interactions into the Hamiltonian."""
        if self.opts.screening == "mrpa":
            assert(seris_intermed is not None)

            # Use bare coulomb interaction from hamiltonian; this could well be cached in future.
            bare_eris = self.get_eris_bare()

            self._seris = scrcoulomb.get_screened_eris_full(bare_eris, seris_intermed)

        elif self.opts.screening == "crpa":
            raise NotImplementedError()

        else:
            raise ValueError("Unknown cluster screening protocol: %s" % self.opts.screening)

    def assert_equal_spin_channels(self, message=""):
        na, nb = self.ncas
        if na != nb:
            raise NotImplementedError("Active spaces with different number of alpha and beta orbitals are not yet "
                                      "supported with this configuration. %s", message)


class UClusterHamiltonian(RClusterHamiltonian):
    @property
    def _scf_class(self):
        class UHF_spindep(pyscf.scf.uhf.UHF):
            def energy_elec(mf, dm=None, h1e=None, vhf=None):
                '''Electronic energy of Unrestricted Hartree-Fock

                Note this function has side effects which cause mf.scf_summary updated.

                Returns:
                    Hartree-Fock electronic energy and the 2-electron part contribution
                '''
                if dm is None: dm = mf.make_rdm1()
                if h1e is None:
                    h1e = mf.get_hcore()
                if isinstance(dm, np.ndarray) and dm.ndim == 2:
                    dm = np.array((dm * .5, dm * .5))
                if vhf is None:
                    vhf = mf.get_veff(mf.mol, dm)
                e1 = np.einsum('ij,ji->', h1e[0], dm[0])
                e1 += np.einsum('ij,ji->', h1e[1], dm[1])
                e_coul = (np.einsum('ij,ji->', vhf[0], dm[0]) +
                          np.einsum('ij,ji->', vhf[1], dm[1])) * .5
                e_elec = (e1 + e_coul).real
                mf.scf_summary['e1'] = e1.real
                mf.scf_summary['e2'] = e_coul.real
                #logger.debug(mf, 'E1 = %s  Ecoul = %s', e1, e_coul.real)
                return e_elec, e_coul

        return UHF_spindep

    @property
    def ncas(self):
        return self.cluster.norb_active

    @property
    def nelec(self):
        return self.cluster.nocc_active

    def get_fock(self, with_vext=True):
        ca, cb = self.cluster.c_active
        fa, fb = self.mf.get_fock()
        fock = (dot(ca.T, fa, ca), dot(cb.T, fb, cb))
        if with_vext and self.v_ext is not None:
            fock = ((fock[0] + self.v_ext[0]),
                     (fock[1] + self.v_ext[1]))
        return fock

    def get_clus_mf_info(self, ao_basis=False):
        if ao_basis:
            nao = self.cluster.c_active.shape[1]
        else:
            nao = self.ncas
        fock = self.get_fock()
        mo_energy = (np.diag(fock[0]), np.diag(fock[1]))
        mo_occ = tuple([np.zeros_like(x) for x in mo_energy])
        mo_occ[0][:self.nelec[0]] = 1.0
        mo_occ[1][:self.nelec[1]] = 1.0
        # Determine whether we want our cluster orbitals expressed in the basis of active orbitals, or in the AO basis.
        if ao_basis:
            mo_coeff = self.cluster.c_active
            ovlp = self.mf.get_ovlp()
        else:
            mo_coeff = (np.eye(self.ncas[0]), np.eye(self.ncas[1]))
            ovlp = (np.eye(self.ncas[0]), np.eye(self.ncas[1]))
        return nao, mo_coeff, mo_energy, mo_occ, ovlp

    def get_heff(self, eris=None, fock=None, with_vext=True):
        if eris is None:
            eris = self.get_eris_bare()
        if fock is None:
            fock = self.get_fock(with_vext=False)

        oa = np.s_[:self.cluster.nocc_active[0]]
        ob = np.s_[:self.cluster.nocc_active[1]]
        gaa, gab, gbb = eris
        va = (einsum('iipq->pq', gaa[oa, oa]) + einsum('pqii->pq', gab[:, :, ob, ob])  # Coulomb
              - einsum('ipqi->pq', gaa[oa, :, :, oa]))  # Exchange
        vb = (einsum('iipq->pq', gbb[ob, ob]) + einsum('iipq->pq', gab[oa, oa])  # Coulomb
              - einsum('ipqi->pq', gbb[ob, :, :, ob]))  # Exchange
        h_eff = (fock[0] - va, fock[1] - vb)
        if with_vext and self.v_ext is not None:
            h_eff = ((h_eff[0] + self.v_ext[0]),
                     (h_eff[1] + self.v_ext[1]))
        return h_eff

    def get_eris_bare(self, *args, **kwargs):
        with log_time(self.log.timing, "Time for AO->MO of ERIs:  %s"):
            coeff = self.cluster.c_active
            eris = (self._fragment.base.get_eris_array(coeff[0]),
                    self._fragment.base.get_eris_array((coeff[0], coeff[0], coeff[1], coeff[1])),
                    self._fragment.base.get_eris_array(coeff[1]))
        return eris

    def to_pyscf_mf(self, force_bare_eris=False):
        # For
        return super().to_pyscf_mf(overwrite_fock=True, force_bare_eris=force_bare_eris)


class EB_RClusterHamiltonian(RClusterHamiltonian):
    @dataclasses.dataclass
    class Options(RClusterHamiltonian.Options):
        polaritonic_shift: bool = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.unshifted_couplings = self._fragment.couplings
        self.bos_freqs = self._fragment.bos_freqs
        if self.opts.polaritonic_shift:
            self.set_polaritonic_shift(self.bos_freqs, self.unshifted_couplings)

    @property
    def polaritonic_shift(self):
        try:
            return self._polaritonic_shift
        except AttributeError as e:
            self.log.critical("Polaritonic shift not yet set.")
            raise e

    @property
    def couplings(self):
        if self.opts.polaritonic_shift:
            return self.get_polaritonic_shifted_couplings()
        return self.unshifted_couplings

    def set_polaritonic_shift(self, freqs, couplings):
        no = self.cluster.nocc_active
        if isinstance(no, int):
            noa = nob = no
        else:
            noa, nob = no
        self._polaritonic_shift = np.multiply(freqs ** (-1), einsum("npp->n", couplings[0][:, :noa, :noa]) +
                                              einsum("npp->n", couplings[1][:, :nob, :nob]))
        self.log.info("Applying Polaritonic shift gives energy change of %e",
                      -sum(np.multiply(self._polaritonic_shift ** 2, freqs)))

    def get_heff(self, eris=None, fock=None, with_vext=True):
        heff = super().get_heff(eris, fock, with_vext)
        if self.opts.polaritonic_shift:
            fock_shift = self.get_polaritonic_fock_shift(self.unshifted_couplings)
            if not np.allclose(fock_shift[0], fock_shift[1]):
                self.log.critical("Polaritonic shift breaks cluster spin symmetry; please either use an unrestricted"
                                  "formalism or bosons without polaritonic shift.")
            heff = heff + fock_shift[0]
        return heff

    def get_polaritonic_fock_shift(self, couplings):
        return tuple([- einsum("npq,n->pq", x + x.transpose(0, 2, 1), self.polaritonic_shift) for x in couplings])

    def get_polaritonic_shifted_couplings(self):
        temp = np.multiply(self.polaritonic_shift, self.bos_freqs) / (2 * self.cluster.nocc_active)
        couplings = tuple([x - einsum("pq,n->npq", np.eye(x.shape[1]), temp) for x in self.unshifted_couplings])
        if not np.allclose(couplings[0], couplings[1]):
            self.log.critical("Polaritonic shifted bosonic fermion-boson couplings break cluster spin symmetry; please"
                              "use an unrestricted formalism.")
            raise RuntimeError()
        return couplings[0]

    def get_eb_dm_polaritonic_shift(self, dm1):
        return (-einsum("n,pq->pqn", self.polaritonic_shift, dm1 / 2),) * 2


class EB_UClusterHamiltonian(UClusterHamiltonian, EB_RClusterHamiltonian):
    @dataclasses.dataclass
    class Options(EB_RClusterHamiltonian.Options):
        polaritonic_shift: bool = True

    def get_heff(self, eris, fock=None, with_vext=True):
        heff = super().get_heff(eris, fock, with_vext)

        if self.opts.polaritonic_shift:
            fock_shift = self.get_polaritonic_fock_shift(self.unshifted_couplings)
            heff = tuple([x + y for x, y in zip(heff, fock_shift)])
        return heff

    def get_polaritonic_shifted_couplings(self):
        temp = np.multiply(self.polaritonic_shift, self.bos_freqs) / (2 * self.cluster.nocc_active)
        return tuple([x - einsum("pq,n->npq", np.eye(x.shape[1]), temp) for x in self.unshifted_couplings])

    def get_eb_dm_polaritonic_shift(self, dm1):
        return tuple([-einsum("n,pq->pqn", self.polaritonic_shift, x) for x in dm1])
