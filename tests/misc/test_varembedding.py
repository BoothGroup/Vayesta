import pytest
from pyscf import lib
import vayesta
import vayesta.ewf
from tests.common import TestCase
from tests import testsystems


class Test_UHF_var_emb(TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import pygnme
        except ImportError:
            pytest.skip("Variational Embedding requires pygnme")

        cls.mf = testsystems.heli_631g.uhf()

    @classmethod
    def tearDownClass(cls):
        del cls.mf

    def test_unrestr_regression_var_emb(self):
        emb = vayesta.ewf.EWF(self.mf)
        with emb.iao_fragmentation(minao="sto-6g") as f:
            fci_frags = f.add_all_atomic_fragments(solver="FCI", bath_options=dict(bathtype="dmet"), auxiliary=True)
        emb.kernel()
        from vayesta.misc.variational_embedding import variational_params

        h, s, dm = variational_params.get_wf_couplings(emb, inc_mf=False)
        w_bare, _, _ = lib.linalg_helper.safe_eigh(h, s, lindep=1e-12)
        # Return lowest eigenvalue.
        e_opt0 = w_bare[0]
        self.assertAlmostEqual(e_opt0, -10.275860643502382)
        e_opt = variational_params.optimise_full_varwav(emb, replace_wf=True)
        self.assertAlmostEqual(e_opt, -10.275860643502385)
        self.assertAlmostEqual(emb.e_tot, -10.274941481117565)


class Test_RHF_var_emb(TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import pygnme
        except ImportError:
            pytest.skip("Variational Embedding requires pygnme")

        cls.mf = testsystems.h6_sto6g.rhf()

    @classmethod
    def tearDownClass(cls):
        del cls.mf

    def test_restr_regression_var_emb(self):
        emb = vayesta.ewf.EWF(self.mf)
        with emb.iao_fragmentation(minao="sto-6g") as f:
            with f.rotational_symmetry(3, axis="z"):
                fci_frags = f.add_atomic_fragment(
                    [0, 1], solver="FCI", bath_options=dict(bathtype="dmet"), auxiliary=True
                )

        emb.kernel()
        from vayesta.misc.variational_embedding import variational_params

        h, s, dm = variational_params.get_wf_couplings(emb, inc_mf=False)
        w_bare, _, _ = lib.linalg_helper.safe_eigh(h, s, lindep=1e-12)
        # Return lowest eigenvalue.
        e_opt0 = w_bare[0]
        e_opt = variational_params.optimise_full_varwav(emb, replace_wf=True)
        self.assertAlmostEqual(e_opt0, -3.238135016548521)
        self.assertAlmostEqual(e_opt, -3.2560646440022927)
        self.assertAlmostEqual(emb.e_tot, -3.1775491323759244)
