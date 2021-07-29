# Standard library
import unittest
import logging

# NumPy
import numpy as np

# PySCF
from pyscf import gto, scf

# Vayesta
import vayesta
from vayesta.agf2 import ragf2, eagf2


class KnownValues(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        mol = gto.Mole()
        mol.atom = 'He 0 0 0; He 0 0 1'
        mol.basis = '6-31g'
        mol.max_memory = 1e9
        mol.verbose = 0
        mol.build()

        mf = scf.RHF(mol)
        mf.conv_tol = 1e-14
        mf.conv_tol_grad = 1e-10
        mf.run()

        silent_log = logging.Logger('silent')
        silent_log.setLevel(logging.CRITICAL)

        gf2_params = {
            'conv_tol': 1e-8,
            'conv_tol_rdm1': 1e-12,
            'conv_tol_nelec': 1e-10,
        }

        gf2 = ragf2.RAGF2(mf, log=silent_log, **gf2_params)
        gf2.run()

        self.mol = mol
        self.mf = mf
        self.log = vayesta.log
        self.silent_log = silent_log
        self.gf2_params = gf2_params
        self.gf2 = gf2

    @classmethod
    def tearDownClass(self):
        del self.mol, self.mf, self.silent_log, self.gf2

    def _test_exact(self, gf2):
        # --- Perform exact test (within convergence) for a complete bath

        # Test ground-state energies
        self.assertAlmostEqual(gf2.mf.e_tot, self.gf2.mf.e_tot, 12)
        self.assertAlmostEqual(gf2.results.e_1b, self.gf2.e_1b, 6)

        # Test moments of the GF
        self.assertAlmostEqual(np.max(np.abs(gf2.results.gf.moment(0)-self.gf2.gf.moment(0))), 0, 6)
        self.assertAlmostEqual(np.max(np.abs(gf2.results.gf.moment(1)-self.gf2.gf.moment(1))), 0, 6)

    def test__lowdin__dmet_mp2(self):
        # --- Complete bath via DMET+MP2

        gf2 = eagf2.EAGF2(
                self.mf,
                log=self.silent_log,
                fragment_type='Lowdin-AO',
                bath_type='MP2-BNO',
                bno_threshold=0.0,
                solver_options=self.gf2_params,
        )

        for i in range(self.mol.natm):
            frag = gf2.make_atom_fragment(i)
            frag.make_bath()

        gf2.kernel()

        self._test_exact(gf2)

    def test__lowdin__dmet(self):
        # --- Complete bath via DMET

        gf2 = eagf2.EAGF2(
                self.mf,
                log=self.silent_log,
                fragment_type='Lowdin-AO',
                bath_type='NONE',
                dmet_threshold=1e-10,
                solver_options=self.gf2_params,
        )

        for i in range(self.mol.natm):
            frag = gf2.make_atom_fragment(i)
            frag.make_bath()

        gf2.kernel()

        for x, frag in enumerate(gf2.fragments):
            self.assertTrue(frag.results.converged)

        self._test_exact(gf2)

    def test__lowdin__ewdmet(self):
        # --- Complete DMET bath via EwDMET with nmom=0

        gf2 = eagf2.EAGF2(
                self.mf,
                log=self.silent_log,
                fragment_type='Lowdin-AO',
                bath_type='POWER',
                nmom_bath=0,
                dmet_threshold=1e-10,
                solver_options=self.gf2_params,
        )

        for i in range(self.mol.natm):
            frag = gf2.make_atom_fragment(i)
            frag.make_bath()

        gf2.kernel()

        for x, frag in enumerate(gf2.fragments):
            self.assertTrue(frag.results.converged)

        self._test_exact(gf2)



if __name__ == '__main__':
    unittest.main()
