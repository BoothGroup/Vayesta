import dataclasses
import numpy as np

from vayesta.core.util import *
# FCI_Solver has infrastructure we require to obtain effective cluster Hamiltonian.
from .solver import EBClusterSolver


class EBCCSD_Solver(EBClusterSolver):
    @dataclasses.dataclass
    class Options(EBClusterSolver.Options):
        rank: tuple = (2, 1, 1)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        try:
            from ebcc import ebccsd
        except ImportError as e:
            raise ImportError("Cannot find ebcc; required to use EBCCSD solver.")

    @property
    def ncas(self):
        return self.cluster.norb_active

    @property
    def nelec(self):
        return 2 * self.cluster.nocc_active

    @property
    def nbos(self):
        return self.fragment.nbos

    def get_eris(self):
        with log_time(self.log.timing, "Time for AO->MO of ERIs:  %s"):
            eris = self.base.get_eris_array(self.cluster.c_active)
        return eris

    def get_input(self, eris=None):
        if eris is None:
            eris = self.get_eris()
        f_act = dot(self.cluster.c_active.T, self.base.get_fock(), self.cluster.c_active)
        couplings = self.fragment.couplings
        if self.opts.polaritonic_shift:
            fock_shift, coupling_shift = self.get_polaritonic_shift(self.fragment.bos_freqs, self.fragment.couplings)
            if not np.allclose(fock_shift[0], fock_shift[1]):
                self.log.critical("Polaritonic shift breaks cluster spin symmetry; please either use an unrestricted"
                                  "formalism or bosons without polaritonic shift.")
                raise RuntimeError
            f_act = f_act + fock_shift[0]
            couplings = tuple([x + y for x, y in zip(couplings, coupling_shift)])

        return ((f_act, f_act), (eris, eris, eris), (self.cluster.nocc_active, self.cluster.nocc_active),
               (self.cluster.nvir_active, self.cluster.nvir_active)), tuple([x.transpose(0, 2, 1) for x in couplings])

    def kernel(self, eris=None):
        """Run FCI kernel."""
        from ebcc import ebccsd

        t0 = timer()
        # This interface handles all conversion into GHF quantities for us.
        # [TODO] Double check if we need this transpose in the couplings.
        #  EBCC expects the annihilation term but may also swap the indexing of fermionic operators...
        inp, couplings = self.get_input(eris)
        self.solver = ebccsd.EBCCSD.fromUHFarrays(*inp,
                                                  gmat=couplings,
                                                  omega=self.fragment.bos_freqs,
                                                  rank=self.opts.rank, autogen_code=True)

        self.e_corr = self.solver.kernel()

        self.solver.solve_lambda()

        self.log.timing("Time for EBCCSD: %s", time_string(timer() - t0))
        self.log.debugv("E_corr(CAS)= %s", energy_string(self.e_corr))
        self.converged = self.solver.converged_t and self.solver.converged_l

    def get_ghf_to_uhf_indices(self):
        no = 2 * self.cluster.nocc_active
        nv = 2 * self.cluster.nvir_active
        nso = no + nv
        temp = [i for i in range(nso)]
        aindx = temp[:self.cluster.nocc_active] + temp[no:no + self.cluster.nvir_active]
        bindx = temp[self.cluster.nocc_active:no] + temp[no + self.cluster.nvir_active:]
        return aindx, bindx

    def make_rdm1(self):
        # This is in GHF orbital ordering.
        ghf_dm1 = self.solver.make_1rdm_f()
        aindx, bindx = self.get_ghf_to_uhf_indices()
        # Want RHF spatial dm1.
        self.dm1 = ghf_dm1[np.ix_(aindx, aindx)] + ghf_dm1[np.ix_(bindx, bindx)]
        return self.dm1

    def make_rdm2(self):
        # This is in GHF orbital ordering.
        ghf_dm2 = self.solver.make_2rdm_f()
        aindx, bindx = self.get_ghf_to_uhf_indices()
        self.dm2 = ghf_dm2[np.ix_(aindx, aindx, aindx, aindx)] + ghf_dm2[np.ix_(bindx, bindx, bindx, bindx)] + \
                   ghf_dm2[np.ix_(aindx, aindx, bindx, bindx)] + ghf_dm2[np.ix_(bindx, bindx, aindx, aindx)]
        return self.dm2

    def make_rdm12(self):
        return self.make_rdm1(), self.make_rdm2()

    def make_dd_moms(self, max_mom, coeffs=None):
        dd_moms = self.solver.make_dd_EOM_moms(max_mom, include_ref_proj=True)
        aindx, bindx = self.get_ghf_to_uhf_indices()
        if isinstance(coeffs, tuple):
            ca, cb = coeffs
        else:
            ca = cb = coeffs
        dd_moms = {i: (dd_moms[np.ix_(aindx, aindx, aindx, aindx, [i])][:, :, :, :, 0],
                       dd_moms[np.ix_(aindx, aindx, bindx, bindx, [i])][:, :, :, :, 0],
                       dd_moms[np.ix_(bindx, bindx, bindx, bindx, [i])][:, :, :, :, 0])
                   for i in range(dd_moms.shape[4])}
        return {i: (einsum("ijkl,ip,jq,kr,ls->pqrs", v[0], ca, ca, ca, ca),
                    einsum("ijkl,ip,jq,kr,ls->pqrs", v[1], ca, ca, cb, cb),
                    einsum("ijkl,ip,jq,kr,ls->pqrs", v[2], cb, cb, cb, cb))
                for i, v in dd_moms.items()
                }

    def make_rdm_eb(self):
        """P[p,q,b] corresponds to <0|b^+ p^+ q|0>.
        """
        # This is in GHF orbital ordering, and returns both bosonic creation and annihilation.
        dm_eb = self.solver.make_eb_coup_rdm(unshifted_bos=True)
        aindx, bindx = self.get_ghf_to_uhf_indices()
        # Check that the density matrices from CC obey this relation; if not we probably need to average, but want to
        # know.
        assert (np.allclose(dm_eb[0], dm_eb[1].transpose(0, 2, 1)))
        dm_eb_crea = dm_eb[0].transpose(1, 2, 0)
        self.dm_eb = (dm_eb_crea[np.ix_(aindx, aindx, list(range(self.nbos)))],
                      dm_eb_crea[np.ix_(bindx, bindx, list(range(self.nbos)))])
        if self.opts.polaritonic_shift:
            self.dm_eb = [x + y for x, y in zip(self.dm_eb, self.get_eb_dm_polaritonic_shift())]
        return self.dm_eb


class UEBCCSD_Solver(EBCCSD_Solver):

    def get_eris(self):
        c_act = self.cluster.c_active
        with log_time(self.log.timing, "Time for AO->MO of ERIs:  %s"):
            eris_aa = self.base.get_eris_array(c_act[0])
            eris_ab = self.base.get_eris_array((c_act[0], c_act[0], c_act[1], c_act[1]))
            eris_bb = self.base.get_eris_array(c_act[1])
        return (eris_aa, eris_ab, eris_bb)

    def get_input(self, eris=None):
        if eris is None:
            eris = self.get_eris()
        fock = self.base.get_fock()
        f_act = tuple([dot(self.cluster.c_active[i].T, fock[i], self.cluster.c_active[i]) for i in [0, 1]])

        couplings = self.fragment.couplings
        if self.opts.polaritonic_shift:
            fock_shift, coupling_shift = self.get_polaritonic_shift(self.fragment.bos_freqs, self.fragment.couplings)
            f_act = tuple([x + y for x, y in zip(f_act, fock_shift)])
            couplings = tuple([x + y for x, y in zip(couplings, coupling_shift)])

        return (f_act, eris, self.cluster.nocc_active, self.cluster.nvir_active), \
               tuple([x.transpose(0, 2, 1) for x in couplings])

    def get_ghf_to_uhf_indices(self):
        no = sum(self.cluster.nocc_active)
        nv = sum(self.cluster.nvir_active)
        nso = no + nv
        temp = [i for i in range(nso)]
        aindx = temp[:self.cluster.nocc_active[0]] + temp[no:no + self.cluster.nvir_active[0]]
        bindx = temp[self.cluster.nocc_active[0]:no] + temp[no + self.cluster.nvir_active[0]:]
        return aindx, bindx

    def make_rdm1(self):
        # This is in GHF orbital ordering.
        ghf_dm1 = self.solver.make_1rdm_f()
        aindx, bindx = self.get_ghf_to_uhf_indices()
        # Want UHF spin dm1.
        self.dm1 = (ghf_dm1[np.ix_(aindx, aindx)], ghf_dm1[np.ix_(bindx, bindx)])
        return self.dm1

    def make_rdm2(self):
        # This is in GHF orbital ordering.
        ghf_dm2 = self.solver.make_2rdm_f()
        aindx, bindx = self.get_ghf_to_uhf_indices()
        self.dm2 = (ghf_dm2[np.ix_(aindx, aindx, aindx, aindx)], ghf_dm2[np.ix_(aindx, aindx, bindx, bindx)],
                    ghf_dm2[np.ix_(bindx, bindx, bindx, bindx)])
        return self.dm2
