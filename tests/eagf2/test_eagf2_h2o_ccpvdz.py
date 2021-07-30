# Standard library
import unittest
import logging
import warnings
warnings.simplefilter('ignore', DeprecationWarning)

# NumPy
import numpy as np

# PySCF
from pyscf import gto, scf, lib

# Vayesta
import vayesta
from vayesta.eagf2 import ragf2, eagf2


class KnownValues(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        mol = gto.Mole()
        mol.atom = 'O 0 0 0.11779; H 0 0.755453 -0.471161; H 0 -0.755453 -0.471161'
        mol.basis = 'cc-pvdz'
        mol.max_memory = 1e9
        mol.verbose = 0
        mol.build()

        mf = scf.RHF(mol)
        mf.conv_tol = 1e-14
        mf.conv_tol_grad = 1e-10
        mf.run()

        mf_df = scf.RHF(mol)
        mf_df = mf_df.density_fit()
        mf_df.conv_tol = 1e-14
        mf_df.conv_tol_grad = 1e-10
        mf_df.run()

        silent_log = logging.Logger('silent')
        silent_log.setLevel(logging.CRITICAL)

        gf2_params = {
            'conv_tol': 1e-8,
            'conv_tol_t0': 1e-8,
            'conv_tol_rdm1': 1e-9,
            'conv_tol_nelec': 1e-8,
            'max_cycle_inner': 200,
            'weight_tol': 0.0,
        }

        gf2 = ragf2.RAGF2(mf, log=silent_log, **gf2_params)
        gf2.run()

        gf2_df = ragf2.RAGF2(mf_df, log=silent_log, **gf2_params)
        gf2_df.run()

        self.mol = mol
        self.mf = mf
        self.mf_df = mf_df
        self.log = vayesta.log
        self.silent_log = silent_log
        self.gf2_params = gf2_params
        self.gf2 = gf2
        self.gf2_df = gf2_df

    @classmethod
    def tearDownClass(self):
        del self.mol, self.mf, self.mf_df, self.silent_log, self.gf2, self.gf2_df

    def _test_exact(self, gf2):
        # --- Perform exact test (within convergence) for i.e. a complete bath
        for x, frag in enumerate(gf2.fragments):
            self.assertTrue(frag.results.converged)

        # Test ground-state energies
        self.assertAlmostEqual(gf2.mf.e_tot, self.gf2.mf.e_tot, 12)
        self.assertAlmostEqual(gf2.results.e_1b, self.gf2.e_1b, 8)

        # Test moments of the GF
        self.assertAlmostEqual(np.max(np.abs(gf2.results.gf.moment(0)-self.gf2.gf.moment(0))), 0, 8)
        self.assertAlmostEqual(np.max(np.abs(gf2.results.gf.moment(1)-self.gf2.gf.moment(1))), 0, 8)
        self.assertAlmostEqual(np.max(np.abs(gf2.results.gf.moment(2)-self.gf2.gf.moment(2))), 0, 8)

    def _test_exact_df(self, gf2):
        # --- Perform exact test for DF
        for x, frag in enumerate(gf2.fragments):
            self.assertTrue(frag.results.converged)

        # Test ground-state energies
        self.assertAlmostEqual(gf2.mf.e_tot, self.gf2_df.mf.e_tot, 12)
        self.assertAlmostEqual(gf2.results.e_1b, self.gf2_df.e_1b, 8)

        # Test moments of the GF
        self.assertAlmostEqual(np.max(np.abs(gf2.results.gf.moment(0)-self.gf2_df.gf.moment(0))), 0, 8)
        self.assertAlmostEqual(np.max(np.abs(gf2.results.gf.moment(1)-self.gf2_df.gf.moment(1))), 0, 8)
        self.assertAlmostEqual(np.max(np.abs(gf2.results.gf.moment(2)-self.gf2_df.gf.moment(2))), 0, 8)

    def test__lowdin__dmet_mp2__complete(self):
        # --- Complete bath via DMET+MP2

        gf2 = eagf2.EAGF2(
                self.mf,
                log=self.silent_log,
                fragment_type='Lowdin-AO',
                bath_type='MP2-BNO',
                bno_threshold=0.0,
                dmet_threshold=1e-10,
                solver_options=self.gf2_params,
        )

        for i in range(self.mol.natm):
            frag = gf2.make_atom_fragment(i)

        gf2.kernel()

        self._test_exact(gf2)

    def test__lowdin__all(self):
        # --- Explicitly complete bath (should be same as complete DMET+MP2)

        gf2 = eagf2.EAGF2(
                self.mf,
                log=self.silent_log,
                fragment_type='Lowdin-AO',
                bath_type='ALL',
                dmet_threshold=1e-10,
                solver_options=self.gf2_params,
        )

        for i in range(self.mol.natm):
            frag = gf2.make_atom_fragment(i)

        gf2.kernel()

        self._test_exact(gf2)

    def test__lowdin__power__complete(self):
        # --- Complete bath via power orbitals

        gf2 = eagf2.EAGF2(
                self.mf,
                log=self.silent_log,
                fragment_type='Lowdin-AO',
                bath_type='POWER',
                nmom_bath=4,
                dmet_threshold=1e-10,
                solver_options=self.gf2_params,
        )

        for i in range(self.mol.natm):
            frag = gf2.make_atom_fragment(i)

        gf2.kernel()

        self._test_exact(gf2)

    def test__df__lowdin__dmet_mp2__complete(self):
        # --- Complete bath via DMET+MP2

        gf2 = eagf2.EAGF2(
                self.mf_df,
                log=self.silent_log,
                fragment_type='Lowdin-AO',
                bath_type='MP2-BNO',
                bno_threshold=0.0,
                dmet_threshold=1e-10,
                solver_options=self.gf2_params,
        )

        for i in range(self.mol.natm):
            frag = gf2.make_atom_fragment(i)

        gf2.kernel()

        self._test_exact_df(gf2)

    def test__lowdin__dmet_mp2(self):
        # --- Incomplete DMET+MP2 bath

        gf2 = eagf2.EAGF2(
                self.mf,
                log=self.silent_log,
                fragment_type='Lowdin-AO',
                bath_type='MP2-BNO',
                bno_threshold=1e-4,
                dmet_threshold=1e-10,
                solver_options=self.gf2_params,
        )

        for i in range(self.mol.natm):
            frag = gf2.make_atom_fragment(i)

        gf2.kernel()

        for x, frag in enumerate(gf2.fragments):
            self.assertTrue(frag.results.converged)

        self.assertAlmostEqual(gf2.results.gf.make_rdm1().trace(), self.mol.nelectron, 8)

        self.assertAlmostEqual(gf2.results.e_1b,                 -75.8994510476693, 8)
        self.assertAlmostEqual(gf2.results.e_2b,                  -0.3359556660481, 8)
        self.assertAlmostEqual(lib.fp(gf2.results.gf.moment(0)),   0.6671541571085, 8)
        self.assertAlmostEqual(lib.fp(gf2.results.gf.moment(1)), -44.1077558841127, 8)
        self.assertAlmostEqual(lib.fp(gf2.results.gf.moment(2)), 355.5543812465610, 8)

    #TODO: test these using He, 0th SE moment non pos def here
    #def test__lowdin__dmet(self):
    #    # --- Incomplete DMET bath

    #    gf2 = eagf2.EAGF2(
    #            self.mf,
    #            log=self.silent_log,
    #            fragment_type='Lowdin-AO',
    #            bath_type='NONE',
    #            dmet_threshold=1e-10,
    #            solver_options=self.gf2_params,
    #    )

    #    for i in range(self.mol.natm):
    #        frag = gf2.make_atom_fragment(i)

    #    gf2.kernel()

    #    for x, frag in enumerate(gf2.fragments):
    #        self.assertTrue(frag.results.converged)

    #    self.assertAlmostEqual(gf2.results.gf.make_rdm1().trace(), self.mol.nelectron, 8)

    #    self.assertAlmostEqual(gf2.results.e_1b,                 -75.9790999958446, 8)
    #    self.assertAlmostEqual(gf2.results.e_2b,                    0.000000000000, 8)
    #    self.assertAlmostEqual(lib.fp(gf2.results.gf.moment(0)),    0.667154157109, 8)
    #    self.assertAlmostEqual(lib.fp(gf2.results.gf.moment(1)), -43.9438823784379, 8)
    #    self.assertAlmostEqual(lib.fp(gf2.results.gf.moment(2)), 357.7367820105954, 8)

    #def test__lowdin__dmet_via_ewdmet(self):
    #    # --- Incomplete DMET bath via EwDMET with nmom=0

    #    gf2 = eagf2.EAGF2(
    #            self.mf,
    #            log=self.silent_log,
    #            fragment_type='Lowdin-AO',
    #            bath_type='POWER',
    #            nmom_bath=0,
    #            dmet_threshold=1e-10,
    #            solver_options=self.gf2_params,
    #    )

    #    for i in range(self.mol.natm):
    #        frag = gf2.make_atom_fragment(i)

    #    gf2.kernel()

    #    for x, frag in enumerate(gf2.fragments):
    #        self.assertTrue(frag.results.converged)

    #    self.assertAlmostEqual(gf2.results.gf.make_rdm1().trace(), self.mol.nelectron, 8)

    #    self.assertAlmostEqual(gf2.results.e_1b,                 -75.9790999958446, 8)
    #    self.assertAlmostEqual(gf2.results.e_2b,                    0.000000000000, 8)
    #    self.assertAlmostEqual(lib.fp(gf2.results.gf.moment(0)),    0.667154157109, 8)
    #    self.assertAlmostEqual(lib.fp(gf2.results.gf.moment(1)), -43.9438823784379, 8)
    #    self.assertAlmostEqual(lib.fp(gf2.results.gf.moment(2)), 357.7367820105954, 8)

    def test__lowdin__power(self):
        # --- Incomplete bath via power orbitals

        gf2 = eagf2.EAGF2(
                self.mf,
                log=self.silent_log,
                fragment_type='Lowdin-AO',
                bath_type='POWER',
                nmom_bath=3,
                dmet_threshold=1e-10,
                solver_options=self.gf2_params,
        )

        for i in range(self.mol.natm):
            frag = gf2.make_atom_fragment(i)

        gf2.kernel()

        self.assertAlmostEqual(gf2.results.gf.make_rdm1().trace(), self.mol.nelectron, 8)

        self.assertAlmostEqual(gf2.results.e_1b,                 -75.898135786957, 8)
        self.assertAlmostEqual(gf2.results.e_2b,                  -0.336850822798, 8)
        self.assertAlmostEqual(lib.fp(gf2.results.gf.moment(0)),   0.667154157109, 8)
        self.assertAlmostEqual(lib.fp(gf2.results.gf.moment(1)), -44.106041308364, 8)
        self.assertAlmostEqual(lib.fp(gf2.results.gf.moment(2)), 355.524388875832, 8)



if __name__ == '__main__':
    unittest.main()
