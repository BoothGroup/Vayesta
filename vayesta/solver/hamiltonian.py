import dataclasses
from typing import Optional

import numpy as np
import pyscf.lib
import pyscf.scf
import scipy.linalg

from vayesta.core.util import dot, einsum, OptionsBase, break_into_lines, log_time
from typing import Optional
from vayesta.core.types import Orbitals
from vayesta.core.screening import screening_moment, screening_crpa
from vayesta.rpa import ssRPA


def is_ham(ham):
    return issubclass(type(ham), RClusterHamiltonian)


def is_uhf_ham(ham):
    return issubclass(type(ham), UClusterHamiltonian)


def is_eb_ham(ham):
    return issubclass(type(ham), EB_RClusterHamiltonian)


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


class DummyERIs:
    def __init__(self, getter, valid_blocks, **kwargs):
        self.getter = getter
        self.valid_blocks = valid_blocks
        for k, v in kwargs.items():
            if k in self.valid_blocks:
                raise ValueError("DummyERIs object passed same attribute twice!")
            else:
                self.__setattr__(k, v)

    def __getattr__(self, key: str):
        """Just-in-time attribute getter."""
        if key in self.valid_blocks:
            return self.getter(block=key)
        else:
            raise AttributeError


class RClusterHamiltonian:
    @dataclasses.dataclass
    class Options(OptionsBase):
        screening: Optional[str] = None
        cache_eris: bool = True

    @property
    def _scf_class(self):
        return pyscf.scf.RHF

    def __init__(self, fragment, mf, log=None, cluster=None, **kwargs):
        self.orig_mf = mf
        # Do we want to populate all parameters at initialisation, so fragment isn't actually saved here?
        self._fragment = fragment
        self._cluster = cluster
        self.log = (log or fragment.log)
        # --- Options:
        self.opts = self.Options()
        self.opts.update(**kwargs)
        self.log.info("Parameters of %s:" % self.__class__.__name__)
        self.log.info(break_into_lines(str(self.opts), newline='\n    '))
        self.v_ext = None
        self._seris = None
        self._eris = None

    @property
    def cluster(self):
        return self._cluster or self._fragment.cluster

    @property
    def mo(self):
        return Orbitals(self.cluster.c_active, occ=self.cluster.nocc_active)

    @property
    def nelec(self):
        return (self.cluster.nocc_active, self.cluster.nocc_active)

    @property
    def ncas(self):
        return (self.cluster.norb_active, self.cluster.norb_active)

    @property
    def has_screening(self):
        return self._seris is not None

    def target_space_projector(self, c=None):
        """Projector to the target fragment space within our cluster."""
        return self._fragment.get_fragment_projector(self.cluster.c_active, c)

    # Integrals for the cluster calculations.

    def get_integrals(self, bare_eris=None, with_vext=True):
        heff = self.get_heff(bare_eris, with_vext=with_vext)
        # Depending on calculation this may be the same as bare_eris
        seris = self.get_eris_screened()
        return heff, seris

    def get_fock(self, with_vext=True, use_seris=True, with_exxdiv=False):
        c = self.cluster.c_active
        fock = dot(c.T, self._fragment.base.get_fock(with_exxdiv=with_exxdiv), c)
        if with_vext and self.v_ext is not None:
            fock += self.v_ext

        if self._seris is not None and use_seris:
            # Generates the fock matrix if screened ERIs are used in place of bare eris.
            occ = np.s_[:self.cluster.nocc_active]

            eris = self.get_eris_bare()
            v_act_bare = 2 * einsum('iipq->pq', eris[occ, occ]) - einsum('iqpi->pq', eris[occ, :, :, occ])
            v_act_seris = 2 * einsum('iipq->pq', self._seris[occ, occ]) - \
                          einsum('iqpi->pq', self._seris[occ, :, :, occ])

            fock += v_act_seris - v_act_bare

        return fock

    def get_heff(self, eris=None, fock=None, with_vext=True, with_exxdiv=False):
        if eris is None:
            eris = self.get_eris_bare()
        if fock is None:
            fock = self.get_fock(with_vext=False, use_seris=False, with_exxdiv=with_exxdiv)
        occ = np.s_[:self.cluster.nocc_active]
        v_act = 2 * einsum('iipq->pq', eris[occ, occ]) - einsum('iqpi->pq', eris[occ, :, :, occ])
        h_eff = fock - v_act
        if with_vext and self.v_ext is not None:
            h_eff += self.v_ext
        return h_eff

    def get_eris_screened(self, block=None):
        # This will only return the bare eris if no screening is expected
        if self.opts.screening is None:
            return self.get_eris_bare(block)
        if self._seris is None:
            self.log.critical("ERIs requested before expected screened interactions have been initialised.")
            raise RuntimeError("ERIs requested before expected screened interactions have been initialised.")
        if block is None:
            return self._seris
        else:
            return self._get_eris_block(self._seris, block)

    def get_eris_bare(self, block=None):
        if block is None:
            if self._eris is None:
                with log_time(self.log.timing, "Time for AO->MO of ERIs:  %s"):
                    coeff = self.cluster.c_active
                    eris = self._fragment.base.get_eris_array(coeff)
                if self.opts.cache_eris:
                    self._eris = eris
                return eris
            else:
                return self._eris
        else:
            assert len(block) == 4 and set(block.lower()).issubset(set("ov"))
            if self._eris is None:
                # Actually generate the relevant block.
                coeffs = [self.cluster.c_active_occ if i == "o" else self.cluster.c_active_vir for i in block.lower()]
                return self._fragment.base.get_eris_array(coeffs)
            else:
                return self._get_eris_block(self._eris, block)

    def _get_eris_block(self, eris, block):
        assert len(block) == 4 and set(block.lower()).issubset({"o", "v"})
        # Just get slices of cached eri.
        occ = slice(self.cluster.nocc_active)
        vir = slice(self.cluster.nocc_active, self.cluster.norb_active)
        sl = tuple([occ if i == "o" else vir for i in block.lower()])
        return eris[sl]

    def get_cderi_bare(self, only_ov=False, compress=False, svd_threshold=1e-12):

        if only_ov:
            # We only need the (L|ov) block for MP2:
            mo_coeff = (self.cluster.c_active_occ, self.cluster.c_active_vir)
        else:
            mo_coeff = self.cluster.c_active

        with log_time(self.log.timing, "Time for 2e-integral transformation: %s"):
            cderi, cderi_neg = self._fragment.base.get_cderi(mo_coeff)

        if compress:
            # SVD and compress the cderi tensor. This scales as O(N_{aux} N_{clus}^4), so this will be worthwhile
            # provided the following approach has a lower scaling than this.
            def compress_cderi(cd):
                naux, n1, n2 = cd.shape
                u, s, v = np.linalg.svd(cd.reshape(naux, n1 * n2), full_matrices=False)
                want = s > svd_threshold
                self.log.debugv("CDERIs auxbas compressed from %d to %d in size", naux, sum(want))
                return einsum("n,nq->nq", s[want], v[want]).reshape(-1, n1, n2)

            cderi = compress_cderi(cderi)
            if cderi_neg is not None:
                cderi_neg = compress_cderi(cderi_neg)

        return cderi, cderi_neg

    # Generate mean-field object representing the cluster.

    def to_pyscf_mf(self, allow_dummy_orbs=False, force_bare_eris=False, overwrite_fock=False, allow_df=False):
        """
        Generate pyscf.scf object representing this active space Hamiltonian.
        This should be able to be passed into a standard post-hf `pyscf` solver without modification.

        Parameters
        ----------
        allow_dummy_orbs : bool, optional
            Whether the introduction of dummy orbitals into the mean-field, which are then frozen, is permitted.
            Default is False
        force_bare_eris : bool, optional
            Forces resultant mean-field object to use unscreened eris.
            Default is False
        overwrite_fock : bool, optional
            Whether `mf.get_fock` should be set to `self.get_fock`. Mainly for use in UHF.
            Default is False
        allow_df : bool, optional
            Whether the resultant mean-field object should include a `.with_df` object containing the projection of the
            CDERIs into the cluster space.
            Default is False

        Returns
        -------
        clusmf : pyscf.scf.SCF
            Representation of cluster as pyscf mean-field.
        orbs_to_freeze : list of lists
            Which orbitals to freeze, split by spin channel if UHF.
        """
        # Using this approach requires dummy orbitals or equal spin channels.
        if not allow_dummy_orbs:
            self.assert_equal_spin_channels()
        nmo = max(self.ncas)
        nsm = min(self.ncas)
        # Get dummy information on mf state. Note our `AOs` are in fact the rotated MOs.
        nao, mo_coeff, mo_energy, mo_occ, ovlp = self.get_clus_mf_info(ao_basis=False, with_vext=True)
        # Now, define function to equalise spin channels.
        if nsm == nmo:
            # No need to pad, these functions don't need to do anything.
            def pad_to_match(array, diag_val=0.0):
                return array

            orbs_to_freeze = None
            dummy_energy = 0.0
        else:
            self.log.info("Using %d dummy orbital(s) to pad local Hamiltonian.", nmo - nsm)
            # Note that this branch is actually UHF-exclusive.
            padchannel = self.ncas.index(nsm)
            orbs_to_freeze = [[], []]
            orbs_to_freeze[padchannel] = list(range(nsm, nmo))

            # Pad all indices which are smaller than nmo to this size with zeros.
            # Optionally introduce value on diagonal if all indices are padded.
            def pad_to_match(array_tup, diag_val=0.0):
                def pad(a, diag_val):
                    topad = np.array(a.shape) < nmo
                    if any(topad):
                        sl = tuple([slice(nsm) if x else slice(None) for x in topad])
                        new = np.zeros((nmo,) * a.ndim)
                        new[sl] = a
                        a = new
                    if all(topad):
                        for i in range(nsm, nmo):
                            a[(i,) * a.ndim] = diag_val
                    return a

                return [pad(x, diag_val) for x in array_tup]

            # For appropriate one-body quantities (hcore, fock) set diagonal dummy index value to effective virtual
            # orbital energy higher than highest actual orbital energy.
            dummy_energy = 10.0 + max([x.max() for x in mo_energy])

        # Set up dummy mol object representing cluster.
        clusmol = pyscf.gto.mole.Mole()
        # Copy over all output controls from original mol object.
        clusmol.verbose = self.orig_mf.mol.verbose
        if self.orig_mf.mol.output is not None:
            clusmol.output = f"f{self._fragment.id}_{self.orig_mf.mol.output}"
            self.log.debugv("Setting solver output file to %s", clusmol.output)
        # Set information as required for our cluster.
        clusmol.nelec = self.nelec
        clusmol.nao = nmo
        clusmol.build()
        # Then scf object representing mean-field electronic state.
        clusmf = self._scf_class(clusmol)
        # First set mean-field parameters.
        clusmf.mo_coeff = pad_to_match(mo_coeff, 1.0)
        clusmf.mo_occ = np.array(pad_to_match(mo_occ, 0.0))
        clusmf.mo_energy = pad_to_match(mo_energy, dummy_energy)
        clusmf.get_ovlp = lambda *args, **kwargs: pad_to_match(ovlp, 1.0)

        # Determine if we want to use DF within the cluster to reduce memory costs.
        # Only want this if
        #   -using RHF (UHF would be more complicated).
        #   -using bare ERIs in cluster.
        #   -ERIs are PSD.
        #   -our mean-field has DF.
        use_df = allow_df and np.ndim(clusmf.mo_coeff[1]) == 1 and self.opts.screening is None and \
                 not (self._fragment.base.pbc_dimension in (1, 2)) and hasattr(self.orig_mf, 'with_df') \
                 and self.orig_mf.with_df is not None
        clusmol.incore_anyway = not use_df

        if use_df:
            # Set up with DF
            clusmf = clusmf.density_fit()
            # Populate a dummy density fitting object.
            cderis = pyscf.lib.pack_tril(self.get_cderi_bare(only_ov=False, compress=True)[0])
            clusmf.with_df._cderi = cderis
            # This gives us enough information to generate the local effective interaction.
            # This works since it must also be RHF.
            heff = self.get_fock(with_vext=True, use_seris=False) - clusmf.get_veff()
        else:
            # Just set up heff using standard bare eris.
            bare_eris = self.get_eris_bare()
            heff = pad_to_match(self.get_heff(eris=bare_eris, with_vext=True), dummy_energy)
            if force_bare_eris:
                clusmf._eri = pad_to_match(bare_eris, 0.0)
            else:
                clusmf._eri = pad_to_match(self.get_eris_screened())

        clusmf.get_hcore = lambda *args, **kwargs: heff
        if overwrite_fock:
            clusmf.get_fock = lambda *args, **kwargs: pad_to_match(
                self.get_fock(with_vext=True, use_seris=not force_bare_eris), dummy_energy)
            clusmf.get_veff = lambda *args, **kwargs: np.array(clusmf.get_fock(*args, **kwargs)) - \
                                                      np.array(clusmf.get_hcore())

        return clusmf, orbs_to_freeze

    def get_clus_mf_info(self, ao_basis=False, with_vext=True, with_exxdiv=False):
        if ao_basis:
            nao = self.cluster.c_active.shape[1]
        else:
            nao = self.ncas
        mo_energy = np.diag(self.get_fock(with_vext=with_vext, with_exxdiv=with_exxdiv))
        mo_occ = np.zeros_like(mo_energy)
        mo_occ[:self.nelec[0]] = 2.0
        # Determine whether we want our cluster orbitals expressed in the basis of active orbitals, or in the AO basis.
        if ao_basis:
            mo_coeff = self.cluster.c_active
            ovlp = self.orig_mf.get_ovlp()
        else:
            mo_coeff = np.eye(self.ncas[0])
            ovlp = np.eye(self.ncas[0])
        return nao, mo_coeff, mo_energy, mo_occ, ovlp

    # Functionality for use with screened interactions and external corrections.

    def calc_loc_erpa(self, m0, amb):

        no, nv = self.cluster.nocc_active, self.cluster.nvir_active
        nov = no * nv
        # Bare coulomb interaction in cluster ov-ov space.
        v = self.get_eris_bare()[:no, no:, :no, no:].reshape((nov, nov))
        ro = self._fragment.get_overlap("fragment|cluster-occ")
        po = dot(ro.T, ro)

        def gen_spin_components(mat):
            return mat[:nov, :nov], mat[:nov, nov:], mat[nov:, nov:]

        m0_aa, m0_ab, m0_bb = gen_spin_components(m0)
        d_aa, d_ab, d_bb = gen_spin_components(amb)
        # This should be zero.
        assert(abs(d_ab).max() < 1e-10)

        def compute_e_rrpa(proj):
            def pr(m):
                m = m.reshape((no, nv, no*nv))
                m = np.tensordot(proj, m, axes=[0, 0])
                return m.reshape((no*nv, no*nv))
            erpa = 0.5 * (einsum("pq,qp->", pr(m0_aa), d_aa) + einsum("pq,qp->", pr(m0_bb), d_bb))
            erpa += einsum("pq,qp->", pr(m0_aa + m0_ab + m0_ab.T + m0_bb), v)
            erpa -= 0.5 * (pr(d_aa + v + d_bb + v).trace())
            self.log.info("Computed fragment RPA energy contribution for cluster %s as %s", self._fragment.id,
                          energy_string(erpa))
            return erpa

        compute_e_rrpa(np.eye(no))

        return compute_e_rrpa(po)

    def add_screening(self, seris_intermed=None):
        """Add screened interactions into the Hamiltonian."""
        self._seris = self._add_screening(seris_intermed, spin_integrate=True)

    def _add_screening(self, seris_intermed=None, spin_integrate=True):

        def spin_integrate_and_report(m, warn_threshold=1e-6):
            spat = (m[0] + m[1] + m[2] + m[1].transpose((2, 3, 0, 1))) / 4.0

            dev = [abs(x - spat).max() for x in m] + [abs(m[2].transpose(2, 3, 0, 1) - spat).max()]
            self.log.info("Largest change in screened interactions due to spin integration: %e", max(dev))
            if max(dev) > warn_threshold:
                self.log.warning("Significant change in screened interactions due to spin integration: %e", max(dev))
            else:
                self.log.info("Largest change in screened interactions due to spin integration: %e", max(dev))

            return spat
        seris = None
        if self.opts.screening is None:
            raise ValueError("Attempted to add screening to fragment with no screening protocol specified.")
        if self.opts.screening == "mrpa":
            assert(seris_intermed is not None)
            # Use bare coulomb interaction from hamiltonian.
            bare_eris = self.get_eris_bare()
            seris = screening_moment.get_screened_eris_full(bare_eris, seris_intermed[0], log=self.log)
            if spin_integrate:
                seris = spin_integrate_and_report(seris)
        elif self.opts.screening[:4] == "crpa":
            bare_eris = self.get_eris_bare()
            delta, crpa = screening_crpa.get_frag_deltaW(self.mf, self._fragment,
                                                         pcoupling=("pcoupled" in self.opts.screening),
                                                         only_ov_screened= ("ov" in self.opts.screening),
                                                         log=self.log)
            if "store" in self.opts.screening:
                self.log.warning("Storing cRPA object in Hamiltonian- O(N^4) memory cost!")
                self.crpa = crpa
            if "full" in self.opts.screening:
                # Add a check just in case.
                self.log.critical("Static screening of frequency-dependent interactions not supported")
                self.log.critical("This statement should be impossible to reach!")
                raise ValueError("Static screening of frequency-dependent interactions not supported")
            else:
                if spin_integrate:
                    delta = spin_integrate_and_report(delta)
                    seris = bare_eris + delta
                else:
                    seris = tuple([x + y for x, y in zip(bare_eris, delta)])
        else:
            raise ValueError("Unknown cluster screening protocol: %s" % self.opts.screening)

        def report_screening(screened, bare, spins):
            maxidx = np.unravel_index(np.argmax(abs(screened-bare)), bare.shape)
            if spins is None:
                wstring = "W"
            else:
                wstring = "W(%2s|%2s)" % (2*spins[0], 2*spins[1])
            self.log.info(
                "Maximally screened element of %s: V= %.3e -> W= %.3e (delta= %.3e)",
                     wstring, bare[maxidx], screened[maxidx], screened[maxidx]-bare[maxidx])
            #self.log.info(
            #    "           Corresponding norms%s: ||V||= %.3e, ||W||= %.3e, ||delta||= %.3e",
            #              " " * len(wstring), np.linalg.norm(bare), np.linalg.norm(screened),
            #              np.linalg.norm(screened-bare))

        if spin_integrate:
            report_screening(seris, bare_eris, None)
        else:
            report_screening(seris[0], bare_eris[0], "aa")
            report_screening(seris[1], bare_eris[1], "ab")
            report_screening(seris[2], bare_eris[2], "bb")

            def get_sym_breaking(norm_aa, norm_ab, norm_bb):
                spinsym = abs(norm_aa - norm_bb)/ ((norm_aa + norm_bb)/2)
                spindep = abs((norm_aa+norm_bb)/2-norm_ab) / ((norm_aa+norm_bb+norm_ab)/3)
                return spinsym, spindep

            bss, bsd = get_sym_breaking(*[np.linalg.norm(x) for x in bare_eris])
            sss, ssd = get_sym_breaking(*[np.linalg.norm(x) for x in seris])
            dss, dsd = get_sym_breaking(*[np.linalg.norm(x-y) for x,y in zip(bare_eris, seris)])

            self.log.info("Proportional spin symmetry breaking in norms: V= %.3e, W= %.3e, (W-V= %.3e)", bss, sss, dss)
            self.log.info("Proportional spin dependence in norms: V= %.3e, W= %.3e, (W-V= %.3e)", bsd, ssd, dsd)
        return seris

    def assert_equal_spin_channels(self, message=""):
        na, nb = self.ncas
        if na != nb:
            raise NotImplementedError("Active spaces with different number of alpha and beta orbitals are not yet "
                                      "supported with this configuration. %s" % message)

    def with_new_cluster(self, cluster):
        return self.__class__(self._fragment, self.orig_mf, self.log, cluster, **self.opts.asdict())

    def get_dummy_eri_object(self, force_bare=False, with_vext=True, with_exxdiv=False):
        # Avoid enumerating all possible keys.
        class ValidRHFKeys:
            def __contains__(self, item):
                return type(item) == str and len(item) == 4 and set(item).issubset(set("ov"))

        getter = self.get_eris_bare if force_bare else self.get_eris_screened
        fock = self.get_fock(with_vext=with_vext, use_seris=not force_bare, with_exxdiv=with_exxdiv)
        return DummyERIs(getter, valid_blocks=ValidRHFKeys(), fock=fock, nocc=self.cluster.nocc_active)


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
                # logger.debug(mf, 'E1 = %s  Ecoul = %s', e1, e_coul.real)
                return e_elec, e_coul

        return UHF_spindep

    @property
    def ncas(self):
        return self.cluster.norb_active

    @property
    def nelec(self):
        return self.cluster.nocc_active

    # Integrals for the cluster calculations.

    def get_fock(self, with_vext=True, use_seris=True, with_exxdiv=False):
        ca, cb = self.cluster.c_active
        fa, fb = self._fragment.base.get_fock(with_exxdiv=with_exxdiv)
        fock = (dot(ca.T, fa, ca), dot(cb.T, fb, cb))
        if with_vext and self.v_ext is not None:
            fock = ((fock[0] + self.v_ext[0]),
                    (fock[1] + self.v_ext[1]))
        if self._seris is not None and use_seris:
            # Generates the fock matrix if screened ERIs are used in place of bare eris.
            noa, nob = self.cluster.nocc_active
            oa = np.s_[:noa]
            ob = np.s_[:nob]
            saa, sab, sbb = self._seris
            dfa = (einsum('iipq->pq', saa[oa, oa]) + einsum('pqii->pq', sab[:, :, ob, ob])  # Coulomb
                   - einsum('ipqi->pq', saa[oa, :, :, oa]))  # Exchange
            dfb = (einsum('iipq->pq', sbb[ob, ob]) + einsum('iipq->pq', sab[oa, oa])  # Coulomb
                   - einsum('ipqi->pq', sbb[ob, :, :, ob]))  # Exchange
            gaa, gab, gbb = self.get_eris_bare()
            dfa -= (einsum('iipq->pq', gaa[oa, oa]) + einsum('pqii->pq', gab[:, :, ob, ob])  # Coulomb
                    - einsum('ipqi->pq', gaa[oa, :, :, oa]))  # Exchange
            dfb -= (einsum('iipq->pq', gbb[ob, ob]) + einsum('iipq->pq', gab[oa, oa])  # Coulomb
                    - einsum('ipqi->pq', gbb[ob, :, :, ob]))  # Exchange
            fock = ((fock[0] + dfa),
                    (fock[1] + dfb))

        return fock

    def get_heff(self, eris=None, fock=None, with_vext=True, with_exxdiv=False):
        if eris is None:
            eris = self.get_eris_bare()
        if fock is None:
            fock = self.get_fock(with_vext=False, use_seris=False, with_exxdiv=with_exxdiv)

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

    def get_eris_bare(self, block=None):
        if block is None:
            if self._eris is None:
                with log_time(self.log.timing, "Time for AO->MO of ERIs:  %s"):
                    eris = self._fragment.base.get_eris_array_uhf(self.cluster.c_active)
                if self.opts.cache_eris:
                    self._eris = eris
                return eris
            else:
                return self._eris
        else:
            if self._eris is None:
                coeffs = [self.cluster.c_active_occ[int(i.upper() == i)] if i.lower() == "o" else
                          self.cluster.c_active_vir[int(i.upper() == i)] for i in block]
                return self._fragment.base.get_eris_array(coeffs)
            else:
                return self._get_eris_block(self._eris, block)

    def _get_eris_block(self, eris, block):
        assert len(block) == 4 and set(block.lower()).issubset({"o", "v"})
        sp = [int(i.upper() == i) for i in block]
        flip = sum(sp) == 2 and sp[0] == 1
        if flip:  # Store ab not ba contribution.
            block = block[::-1]
        d = {"o": slice(self.cluster.nocc_active[0]), "O": slice(self.cluster.nocc_active[1]),
             "v": slice(self.cluster.nocc_active[0], self.cluster.norb_active[0]),
             "V": slice(self.cluster.nocc_active[1], self.cluster.norb_active[1])
             }
        sl = tuple([d[i] for i in block])
        res = eris[sum(sp) // 2][sl]
        if flip:
            res = res.transpose(3, 2, 1, 0).conjugate()
        return res

    def get_cderi_bare(self, only_ov=False, compress=False, svd_threshold=1e-12):

        if only_ov:
            # We only need the (L|ov) and (L|OV) blocks:
            c_aa = [self.cluster.c_active_occ[0], self.cluster.c_active_vir[0]]
            c_bb = [self.cluster.c_active_occ[1], self.cluster.c_active_vir[1]]
        else:
            c_aa, c_bb = self.cluster.c_active

        with log_time(self.log.timing, "Time for 2e-integral transformation: %s"):
            cderi_a, cderi_neg_a = self._fragment.base.get_cderi(c_aa)
            cderi_b, cderi_neg_b = self._fragment.base.get_cderi(c_bb)
        cderi = (cderi_a, cderi_b)
        cderi_neg = (cderi_neg_a, cderi_neg_b)

        if compress:
            # SVD and compress the cderi tensor. This scales as O(N_{aux} N_{clus}^4), so this will be worthwhile
            # provided the following approach has a lower scaling than this.
            def compress_cderi(cd):
                naux, n1, n2 = cd.shape
                u, s, v = np.linalg.svd(cd.reshape(naux, n1 * n2), full_matrices=False)
                want = s > svd_threshold
                self.log.debugv("CDERIs auxbas compressed from %d to %d in size", naux, sum(want))
                return einsum("n,nq->nq", s[want], v[want]).reshape(-1, n1, n2)

            cderi = (compress_cderi(cderi[0]), compress_cderi(cderi[0]))
            if cderi_neg[0] is not None:
                cderi_neg = (compress_cderi(cderi_neg[0]), cderi_neg[1])
            if cderi_neg[1] is not None:
                cderi_neg = (cderi_neg[0], compress_cderi(cderi_neg[1]))

        return cderi, cderi_neg

    # Generate mean-field object representing the cluster.

    def to_pyscf_mf(self, allow_dummy_orbs=True, force_bare_eris=False, overwrite_fock=True, allow_df=False):
        # Need to overwrite fock integrals to avoid errors.
        return super().to_pyscf_mf(allow_dummy_orbs=allow_dummy_orbs, force_bare_eris=force_bare_eris,
                                   overwrite_fock=True, allow_df=allow_df)

    def get_clus_mf_info(self, ao_basis=False, with_vext=True, with_exxdiv=False):
        if ao_basis:
            nao = self.cluster.c_active.shape[1]
        else:
            nao = self.ncas
        fock = self.get_fock(with_vext=with_vext, with_exxdiv=with_exxdiv)
        mo_energy = (np.diag(fock[0]), np.diag(fock[1]))
        mo_occ = [np.zeros_like(x) for x in mo_energy]
        mo_occ[0][:self.nelec[0]] = 1.0
        mo_occ[1][:self.nelec[1]] = 1.0
        if mo_occ[0].shape == mo_occ[1].shape:
            mo_occ = np.array(mo_occ)
        # Determine whether we want our cluster orbitals expressed in the basis of active orbitals, or in the AO basis.
        if ao_basis:
            mo_coeff = self.cluster.c_active
            ovlp = self.orig_mf.get_ovlp()
        else:
            mo_coeff = (np.eye(self.ncas[0]), np.eye(self.ncas[1]))
            ovlp = (np.eye(self.ncas[0]), np.eye(self.ncas[1]))
        return nao, mo_coeff, mo_energy, mo_occ, ovlp

    def get_dummy_eri_object(self, force_bare=False, with_vext=True, with_exxdiv=False):
        # Avoid enumerating all possible keys.
        class ValidUHFKeys:
            def __contains__(self, item):
                return type(item) == str and len(item) == 4 and set(item).issubset(set("ovOV"))

        getter = self.get_eris_bare if force_bare else self.get_eris_screened
        fock = self.get_fock(with_vext=with_vext, use_seris=not force_bare, with_exxdiv=with_exxdiv)
        return DummyERIs(getter, valid_blocks=ValidUHFKeys(), fock=fock, nocc=self.cluster.nocc_active)

    def add_screening(self, seris_intermed=None):
        """Add screened interactions into the Hamiltonian."""
        self._seris = self._add_screening(seris_intermed, spin_integrate=False)


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
        return self.unshifted_couplings[0]

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
                              " use an unrestricted formalism.")
            raise RuntimeError("Polaritonic shifted bosonic fermion-boson couplings break cluster spin symmetry; please"
                              " use an unrestricted formalism.")
        return couplings[0]

    def get_eb_dm_polaritonic_shift(self, dm1):
        return (-einsum("n,pq->pqn", self.polaritonic_shift, dm1 / 2),) * 2

    def _add_screening(self, seris_intermed=None, spin_integrate=True):
        return self.get_eris_bare()

class EB_UClusterHamiltonian(UClusterHamiltonian, EB_RClusterHamiltonian):
    @dataclasses.dataclass
    class Options(EB_RClusterHamiltonian.Options):
        polaritonic_shift: bool = True

    @property
    def couplings(self):
        if self.opts.polaritonic_shift:
            return self.get_polaritonic_shifted_couplings()
        return self.unshifted_couplings

    def get_heff(self, eris=None, fock=None, with_vext=True):
        heff = super().get_heff(eris, fock, with_vext)

        if self.opts.polaritonic_shift:
            fock_shift = self.get_polaritonic_fock_shift(self.unshifted_couplings)
            heff = tuple([x + y for x, y in zip(heff, fock_shift)])
        return heff

    def get_polaritonic_shifted_couplings(self):
        temp = np.multiply(self.polaritonic_shift, self.bos_freqs) / sum(self.cluster.nocc_active)
        return tuple([x - einsum("pq,n->npq", np.eye(x.shape[1]), temp) for x in self.unshifted_couplings])

    def get_eb_dm_polaritonic_shift(self, dm1):
        return tuple([-einsum("n,pq->pqn", self.polaritonic_shift, x) for x in dm1])

    def calc_loc_erpa(self, m0, amb):
        raise NotImplementedError()
