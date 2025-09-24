import unittest
import pytest

import numpy as np

import pyscf.ao2mo
import pyscf.pbc
import vayesta
import vayesta.ewf
from vayesta.core.util import cache
from vayesta.core.qemb.self_energy import make_self_energy_1proj, make_self_energy_2proj
from vayesta.tests import testsystems
from vayesta.tests.common import TestCase

@pytest.mark.skip(reason="Requires refactor of self-energy code")
class Test_SelfEnergy(TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import dyson
        except ImportError:
            pytest.skip("Requires dyson")

    def test_fci_hubbard1d_full_frag(self):
        # RHF
        mf = testsystems.hubb_10_u2.rhf()

        from dyson import MBLGF, Spectral
        from dyson.expressions import FCI

        spectral_moment_order = (4,5)

        fci = FCI.hole.from_mf(mf)
        th = fci.build_gf_moments(spectral_moment_order[0])

        fci = FCI.particle.from_mf(mf)
        tp = fci.build_gf_moments(spectral_moment_order[1])

        solverh = MBLGF(th)
        solverp = MBLGF(tp)
        solverh.kernel()
        solverp.kernel()
        se_fci = Spectral.combine(solverh.result, solverp.result).get_self_energy()

        # Full bath EWF
        ewf = vayesta.ewf.EWF(
            mf, bath_options=dict(bathtype="full"), solver_options=dict( n_moments=spectral_moment_order), solver="FCI"
        )
        with ewf.site_fragmentation() as f:
            f.add_atomic_fragment(list(range(10)))
        ewf.kernel()

        se_mom_order = 3

        se1_ewf, se1_static_ewf, _ = make_self_energy_1proj(ewf, use_sym=False)
        se2_ewf, se2_static_ewf, _ = make_self_energy_2proj(ewf, use_sym=False)

        se_mom_fci = [se_fci.moment(i) for i in range(se_mom_order)]
        se_mom_ewf_1proj = [se1_ewf.moment(i) for i in range(se_mom_order)]
        se_mom_ewf_2proj = [se2_ewf.moment(i) for i in range(se_mom_order)]

        self.assertTrue(np.allclose(se_mom_fci, se_mom_ewf_1proj, atol=1e-5))
        self.assertTrue(np.allclose(se_mom_fci, se_mom_ewf_2proj, atol=1e-5))  

    def test_fci_H6_full_bath(self):
        # RHF
        mf = testsystems.h6_sto6g.rhf()
        try:
            from dyson import MBLGF, Spectral
            from dyson.expressions import FCI
        except ImportError:
            pytest.skip("Requires dyson")

        spectral_moment_order = (4,5)

        fci = FCI.hole.from_mf(mf)
        th = fci.build_gf_moments(spectral_moment_order[0])

        fci = FCI.particle.from_mf(mf)
        tp = fci.build_gf_moments(spectral_moment_order[1])

        solverh = MBLGF(th)
        solverp = MBLGF(tp)
        solverh.kernel()
        solverp.kernel()
        se_fci = Spectral.combine(solverh.result, solverp.result).get_self_energy()
        se_static_fci = th[1] + tp[1] - mf.mo_coeff.T @ mf.get_fock() @ mf.mo_coeff

        # Full bath EWF
        ewf = vayesta.ewf.EWF(
            mf, bath_options=dict(bathtype="full"), solver_options=dict( n_moments=spectral_moment_order), solver="FCI"
        )
        with ewf.site_fragmentation() as f:
            f.add_all_atomic_fragments()
        ewf.kernel()

        se_mom_order = 3

        se1_ewf, se1_static_ewf, _ = make_self_energy_1proj(ewf, use_sym=False)
        se1_ewf_sym, se1_static_ewf_sym, _ = make_self_energy_1proj(ewf, use_sym=True)

        se_mom_fci = [se_fci.moment(i) for i in range(se_mom_order)]
        se_mom_ewf_1proj = [se1_ewf.moment(i) for i in range(se_mom_order)]
        se_mom_ewf_1proj_sym = [se1_ewf_sym.moment(i) for i in range(se_mom_order)]

        self.assertTrue(np.allclose(se_mom_fci, se_mom_ewf_1proj, atol=1e-5))
        self.assertTrue(np.allclose(se_mom_fci, se_mom_ewf_1proj_sym, atol=1e-5))
        
        
        self.assertTrue(np.allclose(se1_static_ewf, se_static_fci, atol=1e-6))
        self.assertTrue(np.allclose(se1_static_ewf_sym, se_static_fci, atol=1e-6))

    def test_fci_hubbard_full_bath(self):
        # RHF
        mf = testsystems.hubb_10_u2.rhf()
        try:
            from dyson import MBLGF, Spectral
            from dyson.expressions import FCI
        except ImportError:
            pytest.skip("Requires dyson")

        spectral_moment_order = (4,5)

        fci = FCI.hole.from_mf(mf)
        th = fci.build_gf_moments(spectral_moment_order[0])

        fci = FCI.particle.from_mf(mf)
        tp = fci.build_gf_moments(spectral_moment_order[1])

        solverh = MBLGF(th)
        solverp = MBLGF(tp)
        solverh.kernel()
        solverp.kernel()
        se_fci = Spectral.combine(solverh.result, solverp.result).get_self_energy()
        se_static_fci = th[1] + tp[1] - mf.mo_coeff.T @ mf.get_fock() @ mf.mo_coeff

        # Full bath EWF
        ewf = vayesta.ewf.EWF(
            mf, bath_options=dict(bathtype="full"), solver_options=dict( n_moments=spectral_moment_order), solver="FCI"
        )
        nfrag = 2
        ewf.symmetry.set_translations([mf.mol.nsite//nfrag, 1, 1])
        with ewf.site_fragmentation() as f:
            f.add_atomic_fragment(list(range(nfrag)))
        ewf.kernel()

        se_mom_order = 3

        se1_ewf, se1_static_ewf, _ = make_self_energy_1proj(ewf, use_sym=False)
        se1_ewf_sym, se1_static_ewf_sym, _ = make_self_energy_1proj(ewf, use_sym=True)

        se_mom_fci = [se_fci.moment(i) for i in range(se_mom_order)]
        se_mom_ewf_1proj = [se1_ewf.moment(i) for i in range(se_mom_order)]
        se_mom_ewf_1proj_sym = [se1_ewf_sym.moment(i) for i in range(se_mom_order)]

        self.assertTrue(np.allclose(se_mom_fci, se_mom_ewf_1proj, atol=1e-5))
        self.assertTrue(np.allclose(se_mom_fci, se_mom_ewf_1proj_sym, atol=1e-5))
        
        
        self.assertTrue(np.allclose(se1_static_ewf, se_static_fci, atol=1e-6))
        self.assertTrue(np.allclose(se1_static_ewf_sym, se_static_fci, atol=1e-6))

if __name__ == "__main__":
    print("Running %s" % __file__)
    unittest.main()
