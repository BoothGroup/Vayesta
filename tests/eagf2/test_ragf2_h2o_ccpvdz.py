# Standard library
import unittest
import logging
import warnings
warnings.simplefilter('ignore', DeprecationWarning)

# NumPy
import numpy as np

# PySCF
from pyscf import gto, scf, lib, agf2

# Vayesta
import vayesta
from vayesta.eagf2 import ragf2

#TODO: tests for non-dyson, higher moments, diagonal, SCS, no fock loop


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

        self.mol = mol
        self.mf = mf
        self.mf_df = mf_df
        self.log = vayesta.log
        self.silent_log = silent_log

    @classmethod
    def tearDownClass(self):
        del self.mol, self.mf, self.mf_df, self.silent_log

    def _test_ragf2(self, gf2):
        self.assertTrue(gf2.converged)

        self.assertAlmostEqual(gf2.mf.e_tot, -76.02676799737664, 12)
        self.assertAlmostEqual(gf2.e_1b,     -75.90249374888124, 8)
        self.assertAlmostEqual(gf2.e_2b,      -0.32648601519860, 8)
        self.assertAlmostEqual(gf2.e_init,    -0.16996293837941, 8)

        self.assertAlmostEqual(lib.fp(gf2.gf.moment(0)),    0.6671541571085)
        self.assertAlmostEqual(lib.fp(gf2.gf.moment(1)),  -44.1093999122217)
        self.assertAlmostEqual(lib.fp(gf2.gf.moment(2)),  355.4201908917253)
        self.assertAlmostEqual(lib.fp(gf2.se.moment(0)),    3.2824688295369)
        self.assertAlmostEqual(lib.fp(gf2.se.moment(1)),  -66.0489454955111)
        self.assertAlmostEqual(lib.fp(gf2.se.moment(2)), 1451.2191097894556)

    def test_ragf2(self):
        gf2 = ragf2.RAGF2(
                self.mf,
                log=self.silent_log,
                conv_tol=1e-10,
                conv_tol_rdm1=1e-14,
                max_cycle_inner=200,
        )
        gf2.kernel()

        self._test_ragf2(gf2)

    def test_aofock_ragf2(self):
        gf2 = ragf2.RAGF2(
                self.mf,
                log=self.silent_log,
                conv_tol=1e-10,
                conv_tol_rdm1=1e-14,
                max_cycle_inner=200,
                fock_basis='AO',
        )
        gf2.kernel()

        self._test_ragf2(gf2)

    def test_damping_ragf2(self):
        gf2 = ragf2.RAGF2(
                self.mf,
                log=self.silent_log,
                conv_tol=1e-10,
                conv_tol_t0=1e-10,
                conv_tol_t1=1e-8,
                conv_tol_rdm1=1e-14,
                max_cycle_inner=200,
                damping=0.5,
        )
        gf2.kernel()

        self._test_ragf2(gf2)

    def test_df_ragf2(self):
        gf2 = ragf2.RAGF2(
                self.mf_df,
                log=self.silent_log,
                conv_tol=1e-10,
                conv_tol_rdm1=1e-14,
                max_cycle_inner=200,
        )
        gf2.kernel()

        self.assertTrue(gf2.converged)

        self.assertAlmostEqual(gf2.mf.e_tot, -76.0267469569879, 12)
        self.assertAlmostEqual(gf2.e_1b,     -75.9024993136641, 8)
        self.assertAlmostEqual(gf2.e_2b,      -0.3264172668262, 8)
        self.assertAlmostEqual(gf2.e_init,    -0.1699205368288, 8)

        self.assertAlmostEqual(lib.fp(gf2.gf.moment(0)),    0.6671541571085)
        self.assertAlmostEqual(lib.fp(gf2.gf.moment(1)),  -44.1012550886380)
        self.assertAlmostEqual(lib.fp(gf2.gf.moment(2)),  354.0191682603555)
        self.assertAlmostEqual(lib.fp(gf2.se.moment(0)),    1.8560050584434)
        self.assertAlmostEqual(lib.fp(gf2.se.moment(1)),  -66.5313581826644)
        self.assertAlmostEqual(lib.fp(gf2.se.moment(2)), 1400.2940768244748)

    def test_frozen_ragf2(self):
        gf2 = ragf2.RAGF2(
                self.mf,
                log=self.silent_log,
                conv_tol=1e-10,
                conv_tol_rdm1=1e-14,
                max_cycle_inner=200,
                frozen=(2, 2),
        )
        gf2.kernel()

        self.assertTrue(gf2.converged)

        self.assertAlmostEqual(gf2.mf.e_tot, -76.02676799737664, 12)
        self.assertAlmostEqual(gf2.e_1b,     -75.92577305796752, 8)
        self.assertAlmostEqual(gf2.e_2b,      -0.22450384689459, 8)
        self.assertAlmostEqual(gf2.e_init,    -0.11801015953703, 8)

        self.assertAlmostEqual(lib.fp(gf2.gf.moment(0)), -0.004714096298873)
        self.assertAlmostEqual(lib.fp(gf2.gf.moment(1)), -1.784497175855591)
        self.assertAlmostEqual(lib.fp(gf2.gf.moment(2)), -4.619064619469411)
        self.assertAlmostEqual(lib.fp(gf2.se.moment(0)),  0.303938136102794)
        self.assertAlmostEqual(lib.fp(gf2.se.moment(1)),  1.230414763288702)
        self.assertAlmostEqual(lib.fp(gf2.se.moment(2)),  7.538970289262878)



if __name__ == '__main__':
    unittest.main()
