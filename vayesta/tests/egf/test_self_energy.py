import unittest
import pytest

import numpy as np

import pyscf.pbc
import vayesta
import vayesta.egf
from vayesta.core.util import cache
from vayesta.egf.self_energy import make_self_energy_1proj, make_self_energy_2proj, make_static_self_energy, make_self_energy_moments
from vayesta.tests import testsystems
from vayesta.tests.common import TestCase
try:
    from dyson import CCSD, FCI, MBLGF, AuxiliaryShift, AufbauPrinciple, NullLogger
except ImportError:
    pytest.skip("Requires dyson")

class Test_SelfEnergy(TestCase):
    NMOM_MAX_GF = 8
    NMOM_MAX_SE = 4
    system = testsystems.hubb_10_u2
    solver = 'FCI'
    EXPR = FCI

    @classmethod
    def setUpClass(cls):
        cls.mf = cls.system.rhf()

    @classmethod
    @cache
    def exact(cls, mf, hermitian=True, shift=None):
        try:
            from dyson import CCSD, FCI, MBLGF, MixedMBLGF, AuxiliaryShift, NullLogger
        except ImportError:
            pytest.skip("Requires dyson")

        expr = cls.EXPR["1h"](mf)
        th = expr.build_gf_moments(cls.NMOM_MAX_GF) 
        expr = cls.EXPR["1p"](mf)
        tp = expr.build_gf_moments(cls.NMOM_MAX_GF)

        solverh = MBLGF(th, hermitian=hermitian, log=NullLogger())
        solverp = MBLGF(tp, hermitian=hermitian, log=NullLogger())
        solver = MixedMBLGF(solverh, solverp)
        solver.kernel()
        se = solver.get_self_energy()
        if shift is not None:
            Shift = AuxiliaryShift if shift == 'aux' else AufbauPrinciple
            solver = Shift(th[1]+tp[1], se, mf.mol.nelectron, log=NullLogger())
            solver.kernel()
        gf = solver.get_greens_function()
        se = solver.get_self_energy()
        se_static = th[1] + tp[1]
        return se_static, se, gf
    
    @classmethod
    @cache
    def emb(cls, mf, nfrag):
        opts = dict(solver=cls.solver, proj=1, aux_shift=True, use_sym=True)
        bath_opts = dict(bathtype='full', dmet_threshold=1e-12)
        solver_opts =dict(conv_tol=1e-15, n_moments=(cls.NMOM_MAX_GF, cls.NMOM_MAX_GF))
        emb = vayesta.egf.EGF(mf, **opts, bath_options=bath_opts, solver_options=solver_opts)
        nimage = [emb.mf.mol.natm//nfrag, 1, 1]
        emb.symmetry.set_translations(nimage)
        with emb.site_fragmentation() as f:
            f.add_atomic_fragment(list(range(nfrag)))
        emb.kernel()
        return emb
    
    def test_static_self_energy_full_frag(self):
        emb = self.emb(self.mf, self.mf.mol.natm)
        se_static, se, gf = self.exact(self.mf)
        se_static_sym = 0.5 * (se_static + se_static.T.conj())
        for use_sym in [True, False]:  
            se_static_emb = make_static_self_energy(emb, proj=2, sym_moms=False, with_mf=True, use_sym=use_sym)
            self.assertAllclose(se_static, se_static_emb, atol=1e-6)

            se_static_emb_sym = make_static_self_energy(emb, proj=2, sym_moms=True, with_mf=True, use_sym=use_sym)
            self.assertAllclose(se_static_sym, se_static_emb_sym, atol=1e-6)

    def test_static_self_energy_full_bath(self):
        emb = self.emb(self.mf, 2)
        se_static, se, gf = self.exact(self.mf)
        se_static_sym = 0.5 * (se_static + se_static.T.conj())
        for use_sym in [True, False]:  
            se_static_emb = make_static_self_energy(emb, proj=1, sym_moms=False, with_mf=True, use_sym=use_sym)
            self.assertAllclose(se_static, se_static_emb, atol=1e-6)

            se_static_emb_sym = make_static_self_energy(emb, proj=1, sym_moms=True, with_mf=True, use_sym=use_sym)
            self.assertAllclose(se_static_sym, se_static_emb_sym, atol=1e-6)

    def test_self_energy_lehmann_full_frag(self):
        emb = self.emb(self.mf, self.mf.mol.natm)
        for hermitian in [True, False]:
            #shifts = [None, 'auf', 'aux'] if hermitian else [None, 'auf']
            shifts = [None]
            for chempot_clus in shifts:
                se_static, se, gf = self.exact(self.mf, hermitian=hermitian, shift=chempot_clus)
                se_moms_exact = np.array([se.moment(i) for i in range(self.NMOM_MAX_SE)])
                seh_moms_exact = np.array([se.occupied().moment(i) for i in range(self.NMOM_MAX_SE)])
                sep_moms_exact = np.array([se.virtual().moment(i) for i in range(self.NMOM_MAX_SE)])
                for use_sym in [True, False]:
                    for remove_degeneracy in [True, False]:
                        se_emb = make_self_energy_2proj(emb, hermitian=hermitian, remove_degeneracy=remove_degeneracy, use_sym=use_sym, chempot_clus=chempot_clus)
                        se_moms_emb = np.array([se_emb.moment(i) for i in range(self.NMOM_MAX_SE)])
                        seh_moms_emb = np.array([se_emb.occupied().moment(i) for i in range(self.NMOM_MAX_SE)])
                        sep_moms_emb = np.array([se_emb.virtual().moment(i) for i in range(self.NMOM_MAX_SE)])

                        self.assertAllclose(seh_moms_exact, seh_moms_emb, atol=1e-4)
                        self.assertAllclose(sep_moms_exact, sep_moms_emb, atol=1e-4)
                        self.assertAllclose(se_moms_exact, se_moms_emb, atol=1e-4)


    def test_self_energy_lehmann_full_bath(self):
        emb = self.emb(self.mf, 2)
        for hermitian in [True, False]:
            #shifts = [None, 'auf', 'aux'] if hermitian else [None, 'auf']
            shifts = [None]
            for chempot_clus in shifts:
                se_static, se, gf = self.exact(self.mf, hermitian=hermitian, shift=chempot_clus)
                se_moms_exact = np.array([se.moment(i) for i in range(self.NMOM_MAX_SE)])
                seh_moms_exact = np.array([se.occupied().moment(i) for i in range(self.NMOM_MAX_SE)])
                sep_moms_exact = np.array([se.virtual().moment(i) for i in range(self.NMOM_MAX_SE)])
                for use_sym in [True, False]:
                    for remove_degeneracy in [True, False]:
                        for use_svd in [True]:
                            for img_space in [True, False]:
                                se_emb = make_self_energy_1proj(emb, hermitian=hermitian, use_sym=use_sym, use_svd=use_svd, chempot_clus=chempot_clus, remove_degeneracy=remove_degeneracy, img_space=img_space)
                                se_moms_emb = np.array([se_emb.moment(i) for i in range(self.NMOM_MAX_SE)])
                                seh_moms_emb = np.array([se_emb.occupied().moment(i) for i in range(self.NMOM_MAX_SE)])
                                sep_moms_emb = np.array([se_emb.virtual().moment(i) for i in range(self.NMOM_MAX_SE)])

                                self.assertAllclose(se_moms_exact, se_moms_emb, atol=1e-4)
                                self.assertAllclose(seh_moms_exact, seh_moms_emb, atol=1e-4)
                                self.assertAllclose(sep_moms_exact, sep_moms_emb, atol=1e-4)

    def test_self_energy_moments_full_bath(self):
        emb = self.emb(self.mf, 2)
        for hermitian in [True, False]:
            #shifts = [None, 'auf', 'aux'] if hermitian else [None, 'auf']
            shifts = [None]
            for chempot_clus in shifts:
                se_static, se, gf = self.exact(self.mf, hermitian=hermitian, shift=chempot_clus)
                se_moms_exact = np.array([se.moment(i) for i in range(self.NMOM_MAX_SE)])
                seh_moms_exact = np.array([se.occupied().moment(i) for i in range(self.NMOM_MAX_SE)])
                sep_moms_exact = np.array([se.virtual().moment(i) for i in range(self.NMOM_MAX_SE)])
                for use_sym in [True, False]:
                    seh_moms_emb, sep_moms_emb = make_self_energy_moments(emb, proj=1, ph_separation=True, nmom_se=self.NMOM_MAX_SE, use_sym=use_sym, hermitian=hermitian)
                    se_moms_emb = make_self_energy_moments(emb, proj=1, ph_separation=False, nmom_se=self.NMOM_MAX_SE, use_sym=use_sym, hermitian=hermitian)

                    self.assertAllclose(se_moms_exact, se_moms_emb, atol=1e-4)
                    self.assertAllclose(seh_moms_exact, seh_moms_emb, atol=1e-4)
                    self.assertAllclose(sep_moms_exact, sep_moms_emb, atol=1e-4)

    def test_self_energy_moments_full_frag(self):
        emb = self.emb(self.mf, self.mf.mol.natm)
        for hermitian in [True]:
            #shifts = [None, 'auf', 'aux'] if hermitian else [None, 'auf']
            shifts = [None]
            for chempot_clus in shifts:
                se_static, se, gf = self.exact(self.mf, hermitian=hermitian, shift=chempot_clus)
                se_moms_exact = np.array([se.moment(i) for i in range(self.NMOM_MAX_SE)])
                seh_moms_exact = np.array([se.occupied().moment(i) for i in range(self.NMOM_MAX_SE)])
                sep_moms_exact = np.array([se.virtual().moment(i) for i in range(self.NMOM_MAX_SE)])
                for use_sym in [True, False]:
                    seh_moms_emb, sep_moms_emb = make_self_energy_moments(emb, proj=2, ph_separation=True, nmom_se=self.NMOM_MAX_SE, use_sym=use_sym, hermitian=hermitian)
                    se_moms_emb = make_self_energy_moments(emb, proj=2, ph_separation=False, nmom_se=self.NMOM_MAX_SE, use_sym=use_sym, hermitian=hermitian)

                    self.assertAllclose(se_moms_exact, se_moms_emb, atol=1e-4)
                    self.assertAllclose(seh_moms_exact, seh_moms_emb, atol=1e-4)
                    self.assertAllclose(sep_moms_exact, sep_moms_emb, atol=1e-4)


class Test_SelfEnergy_H6_CCSD(Test_SelfEnergy):
    solver = 'CCSD'
    EXPR = CCSD
    system = testsystems.h6_sto6g

    @classmethod
    @cache
    def emb(cls, mf, nfrag):
        opts = dict(solver=cls.solver, proj=1, aux_shift=True, use_sym=True)
        bath_opts = dict(bathtype='full', dmet_threshold=1e-12)
        solver_opts =dict(conv_tol=1e-15, n_moments=(cls.NMOM_MAX_GF, cls.NMOM_MAX_GF))
        emb = vayesta.egf.EGF(mf, **opts, bath_options=bath_opts, solver_options=solver_opts)
        with emb.iaopao_fragmentation() as f:
            with f.rotational_symmetry(order=int(mf.mol.natm//nfrag), axis='z') as rot: 
                f.add_atomic_fragment(range(nfrag))
        emb.kernel()
        return emb



# class Test_SelfEnergy(TestCase):
#     @classmethod
#     def setUpClass(cls):
#         try:
#             import dyson
#         except ImportError:
#             pytest.skip("Requires dyson")

#     def test_fci_hubbard1d_full_frag(self):
#         # RHF
#         mf = testsystems.hubb_10_u2.rhf()

#         from dyson import MBLGF, MixedMBLGF, NullLogger
#         from dyson.expressions import FCI

#         spectral_moment_order = (4,5)

#         fci = FCI["1h"](mf)
#         th = fci.build_gf_moments(spectral_moment_order[0])

#         fci = FCI["1p"](mf)
#         tp = fci.build_gf_moments(spectral_moment_order[1])

#         solverh = MBLGF(th, log=NullLogger())
#         solverp = MBLGF(tp, log=NullLogger())
#         solver = MixedMBLGF(solverh, solverp)
#         solver.kernel()
#         se_fci = solver.get_self_energy()

#         # Full bath egf
#         egf = vayesta.egf.EGF(
#             mf, bath_options=dict(bathtype="full"), solver_options=dict( n_moments=spectral_moment_order), solver="FCI"
#         )
#         with egf.site_fragmentation() as f:
#             f.add_atomic_fragment(list(range(10)))
#         egf.kernel()

#         se_mom_order = 4

#         se1_egf, se1_static_egf, _ = make_self_energy_1proj(egf, use_sym=False)
#         se2_egf, se2_static_egf, _ = make_self_energy_2proj(egf, use_sym=False)

#         se_mom_fci = [se_fci.moment(i) for i in range(se_mom_order)]
#         se_mom_egf_1proj = [se1_egf.moment(i) for i in range(se_mom_order)]
#         se_mom_egf_2proj = [se2_egf.moment(i) for i in range(se_mom_order)]

#         self.assertTrue(np.allclose(se_mom_fci, se_mom_egf_1proj, atol=1e-5))
#         self.assertTrue(np.allclose(se_mom_fci, se_mom_egf_2proj, atol=1e-5))  

#     def test_fci_H6_full_bath(self):
#         # RHF
#         mf = testsystems.h6_sto6g.rhf()
#         try:
#             from dyson import MBLGF, MixedMBLGF, NullLogger
#             from dyson.expressions import FCI
#         except ImportError:
#             pytest.skip("Requires dyson")

#         spectral_moment_order = (4,5)

#         fci = FCI["1h"](mf)
#         th = fci.build_gf_moments(spectral_moment_order[0])

#         fci = FCI["1p"](mf)
#         tp = fci.build_gf_moments(spectral_moment_order[1])

#         solverh = MBLGF(th, log=NullLogger())
#         solverp = MBLGF(tp, log=NullLogger())
#         solver = MixedMBLGF(solverh, solverp)
#         solver.kernel()
#         se_fci = solver.get_self_energy()
#         se_static_fci = th[1] + tp[1] - mf.mo_coeff.T @ mf.get_fock() @ mf.mo_coeff

#         # Full bath egf
#         egf = vayesta.egf.EGF(
#             mf, bath_options=dict(bathtype="full"), solver_options=dict( n_moments=spectral_moment_order), solver="FCI"
#         )
#         with egf.site_fragmentation() as f:
#             f.add_all_atomic_fragments()
#         egf.kernel()

#         se_mom_order = 4

#         se1_egf, se1_static_egf, _ = make_self_energy_1proj(egf, use_sym=False)
#         se1_egf_sym, se1_static_egf_sym, _ = make_self_energy_1proj(egf, use_sym=True)

#         se_mom_fci = [se_fci.moment(i) for i in range(se_mom_order)]
#         se_mom_egf_1proj = [se1_egf.moment(i) for i in range(se_mom_order)]
#         se_mom_egf_1proj_sym = [se1_egf_sym.moment(i) for i in range(se_mom_order)]

#         self.assertTrue(np.allclose(se_mom_fci, se_mom_egf_1proj, atol=1e-5))
#         self.assertTrue(np.allclose(se_mom_fci, se_mom_egf_1proj_sym, atol=1e-5))
        
        
#         self.assertTrue(np.allclose(se1_static_egf, se_static_fci, atol=1e-6))
#         self.assertTrue(np.allclose(se1_static_egf_sym, se_static_fci, atol=1e-6))

#     def test_fci_hubbard_full_bath(self):
#         # RHF
#         mf = testsystems.hubb_10_u2.rhf()
#         try:
#             from dyson import MBLGF, MixedMBLGF, NullLogger
#             from dyson.expressions import FCI
#         except ImportError:
#             pytest.skip("Requires dyson")

#         spectral_moment_order = (4,5)

#         fci = FCI["1h"](mf)
#         th = fci.build_gf_moments(spectral_moment_order[0])

#         fci = FCI["1p"](mf)
#         tp = fci.build_gf_moments(spectral_moment_order[1])

#         solverh = MBLGF(th, log=NullLogger())
#         solverp = MBLGF(tp, log=NullLogger())
#         solver = MixedMBLGF(solverh, solverp)
#         solver.kernel()
#         se_fci = solver.get_self_energy()
#         se_static_fci = th[1] + tp[1] - mf.mo_coeff.T @ mf.get_fock() @ mf.mo_coeff

#         # Full bath egf
#         egf = vayesta.egf.EGF(
#             mf, bath_options=dict(bathtype="full"), solver_options=dict( n_moments=spectral_moment_order), solver="FCI"
#         )
#         nfrag = 2
#         egf.symmetry.set_translations([mf.mol.nsite//nfrag, 1, 1])
#         with egf.site_fragmentation() as f:
#             f.add_atomic_fragment(list(range(nfrag)))
#         egf.kernel()

#         se_mom_order = 4

#         se1_egf, se1_static_egf, _ = make_self_energy_1proj(egf, use_sym=False)
#         se1_egf_sym, se1_static_egf_sym, _ = make_self_energy_1proj(egf, use_sym=True)

#         se_mom_fci = [se_fci.moment(i) for i in range(se_mom_order)]
#         se_mom_egf_1proj = [se1_egf.moment(i) for i in range(se_mom_order)]
#         se_mom_egf_1proj_sym = [se1_egf_sym.moment(i) for i in range(se_mom_order)]

#         self.assertTrue(np.allclose(se_mom_fci, se_mom_egf_1proj, atol=1e-5))
#         self.assertTrue(np.allclose(se_mom_fci, se_mom_egf_1proj_sym, atol=1e-5))
        
        
#         self.assertTrue(np.allclose(se1_static_egf, se_static_fci, atol=1e-6))
#         self.assertTrue(np.allclose(se1_static_egf_sym, se_static_fci, atol=1e-6))

if __name__ == "__main__":
    print("Running %s" % __file__)
    unittest.main()
