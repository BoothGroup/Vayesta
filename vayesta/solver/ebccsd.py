import dataclasses
import numpy as np

from vayesta.core.util import *
# FCI_Solver has infrastructure we require to obtain effective cluster Hamiltonian.
from .solver2 import ClusterSolver


class EBCCSD_Solver(ClusterSolver):
    @dataclasses.dataclass
    class Options(ClusterSolver.Options):
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

    def kernel(self, eris=None):
        """Run FCI kernel."""
        from ebcc import ebccsd

        if eris is None:
            eris = self.get_eris()
        f_act = dot(self.cluster.c_active.T, self.base.get_fock(), self.cluster.c_active)
        if self.opts.v_ext is not None:
            f_act += self.opts.v_ext
        t0 = timer()
        # This interface handles all conversion into GHF quantities for us.
        self.solver = ebccsd.EBCCSD.fromUHFarrays((f_act, f_act), (eris, eris, eris),
                                                  (self.cluster.nocc_active, self.cluster.nocc_active),
                                                  (self.cluster.nvir_active, self.cluster.nvir_active),
                                                  gmat=self.fragment.couplings, omega=self.fragment.bos_freqs,
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
        # This is in GHF orbital ordering, and returns both bosonic creation and annihilation.
        dm_eb = self.solver.make_eb_coup_rdm(unshifted_bos=True)
        aindx, bindx = self.get_ghf_to_uhf_indices()
        # Check that the density matrices from CC obey this relation; if not we probably need to average, but want to
        # know.
        assert (np.allclose(dm_eb[0], dm_eb[1].transpose(0, 2, 1)))
        dm_eb_crea = dm_eb[0].transpose(1, 2, 0)
        self.dm_eb = (dm_eb_crea[np.ix_(aindx, aindx, list(range(self.nbos)))],
                      dm_eb_crea[np.ix_(bindx, bindx, list(range(self.nbos)))])
        return self.dm_eb
