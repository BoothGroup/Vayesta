import pytest
import unittest
import vayesta
import vayesta.ewf
from vayesta.core.util import cache
from vayesta.tests import testsystems
from vayesta.tests.common import TestCase


@pytest.mark.veryslow
class DiamondTest(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mf = testsystems.diamond_sto3g_k333.rhf()

    @classmethod
    def tearDownClass(cls):
        del cls.mf
        cls.emb.cache_clear()

    @classmethod
    @cache
    def emb(cls, bno_threshold, symmetry=None):
        emb = vayesta.ewf.EWF(
            cls.mf, bath_options=dict(threshold=bno_threshold), solver_options=dict(solve_lambda=True)
        )
        center = (1 / 8, 1 / 8, 1 / 8)  # point between C-atoms
        with emb.fragmentation() as frag:
            if symmetry is None:
                frag.add_atomic_fragment(0)
                frag.add_atomic_fragment(1)
            elif symmetry == "rotation":
                with frag.rotational_symmetry(2, axis=(1, 0, -1), center=center, unit="latvec"):
                    frag.add_atomic_fragment(0)
            elif symmetry == "inversion":
                with frag.inversion_symmetry(center=center, unit="latvec"):
                    frag.add_atomic_fragment(0)
            elif symmetry == "mirror":
                with frag.mirror_symmetry(center=center, axis=(1, 1, 1), unit="latvec"):
                    frag.add_atomic_fragment(0)
            else:
                raise ValueError
        emb.kernel()
        return emb

    # --- Tests for T-symmetry

    def test_dm1_tsymmetry(self):
        emb = self.emb(1e-4)
        dm1_nosym = emb._make_rdm1_ccsd_global_wf(use_sym=False)
        dm1_sym = emb._make_rdm1_ccsd_global_wf(use_sym=True)
        self.assertAllclose(dm1_sym, dm1_nosym)

    def test_corrfunc_dndn_tsymmetry(self):
        emb = self.emb(1e-4)
        corr_nosym = emb.get_corrfunc("dN,dN", use_symmetry=False)
        corr_sym = emb.get_corrfunc("dN,dN", use_symmetry=True)
        self.assertAllclose(corr_sym, corr_nosym)

    def test_corrfunc_szsz_tsymmetry(self):
        emb = self.emb(1e-4)
        corr_nosym = emb.get_corrfunc("Sz,Sz", use_symmetry=False)
        corr_sym = emb.get_corrfunc("Sz,Sz", use_symmetry=True)
        self.assertAllclose(corr_sym, corr_nosym)

    # --- Tests for combined symmetries

    def _test_dm1_symmetry(self, symmetry):
        emb_sym = self.emb(1e-4, symmetry=symmetry)
        emb = self.emb(1e-4)
        dm1_nosym = emb._make_rdm1_ccsd_global_wf(use_sym=False)
        dm1_tsym = emb._make_rdm1_ccsd_global_wf()
        dm1_sym = emb_sym._make_rdm1_ccsd_global_wf()
        self.assertAllclose(dm1_sym, dm1_nosym)
        self.assertAllclose(dm1_tsym, dm1_nosym)

    def test_dm1_rotation_symmetry(self):
        return self._test_dm1_symmetry("rotation")

    def test_dm1_inversion_symmetry(self):
        return self._test_dm1_symmetry("inversion")

    # TODO: Fix failing
    # def test_dm1_mirror_symmetry(self):
    #    return self._test_dm1_symmetry('mirror')


    def _test_ccsd_t_symmetry(self, symmetry):
        emb_sym = self.emb(1e-4, symmetry=symmetry)
        emb = self.emb(1e-4)
        e_ccsd_t = emb.get_ccsd_t_corr_energy(global_t1=False)
        e_ccsd_t_sym = emb_sym.get_ccsd_t_corr_energy(global_t1=False)
        self.assertAlmostEqual(e_ccsd_t, e_ccsd_t_sym)
        e_ccsd_tg = emb.get_ccsd_t_corr_energy(global_t1=True)
        e_ccsd_tg_sym = emb_sym.get_ccsd_t_corr_energy(global_t1=True)
        self.assertAlmostEqual(e_ccsd_tg, e_ccsd_tg_sym)

    def test_ccsd_t_symmetry_rotation(self):
        return self._test_ccsd_t_symmetry("rotation")

    def test_ccsd_t_symmetry_inversion(self):
        return self._test_ccsd_t_symmetry("inversion")

    # def test_ccsd_t_symmetry_mirror(self):
    #     return self._test_ccsd_t_symmetry("mirror")    

if __name__ == "__main__":
    print("Running %s" % __file__)
    unittest.main()
