import unittest

import numpy as np

import pyscf
import pyscf.ao2mo
import pyscf.pbc
import vayesta
import vayesta.ewf
from vayesta.core.util import cache
from tests import systems
from tests.common import TestCase


class Test_Restricted(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mf = systems.water_631g.rhf()
        cls.cc = systems.water_631g.rccsd()

    @classmethod
    def tearDownClass(cls):
        del cls.mf, cls.cc
        if hasattr(cls, "kmf"):
            del cls.kmf
        cls.emb.cache_clear()

    @classmethod
    @cache
    def emb(cls, bno_threshold):
        mf = getattr(cls, "kmf", cls.mf)
        solver_opts = dict(conv_tol=1e-10, conv_tol_normt=1e-8)
        emb = vayesta.ewf.EWF(mf, solver_options=solver_opts, bath_options=dict(threshold=bno_threshold))
        with emb.sao_fragmentation() as f:
            f.add_all_atomic_fragments()
        emb.kernel()
        return emb

    @property
    def nkpts(self):
        if hasattr(self, "kmf"):
            return len(self.kmf.kpts)
        return 1

    def test_make_rdm1_demo_full_bath(self):
        """Test full bath"""
        emb = self.emb(-1)
        dm1_ref = self.cc.make_rdm1(ao_repr=True)
        self.assertAllclose(emb.make_rdm1_demo(ao_basis=True), dm1_ref)

    def test_make_rdm2_demo_full_bath(self):
        """Test full bath"""
        emb = self.emb(-1)
        # Full 2-DM:
        dm2_ref = self.cc.make_rdm2(ao_repr=True)
        self.assertAllclose(emb.make_rdm2_demo(ao_basis=True), dm2_ref)
        self.assertAllclose(emb.make_rdm2_demo(ao_basis=True, part_cumulant=False), dm2_ref)
        self.assertAllclose(emb.make_rdm2_demo(ao_basis=True, approx_cumulant=False), dm2_ref)
        # Approximate cumulant:
        dm2_ref = self.cc.make_rdm2(ao_repr=True, with_dm1=False)
        self.assertAllclose(emb.make_rdm2_demo(ao_basis=True, with_dm1=False), dm2_ref)
        self.assertAllclose(emb.make_rdm2_demo(ao_basis=True, with_dm1=False, part_cumulant=False), dm2_ref)

    def _energy_from_dms(self, dm1, dm2):
        e_nuc = self.mf.energy_nuc()
        h1e = self.mf.get_hcore()
        eri = pyscf.ao2mo.restore(1, self.mf._eri, h1e.shape[-1])
        if getattr(self.mf, "exxdiv", None) is not None:
            madelung = pyscf.pbc.tools.madelung(self.mf.mol, self.mf.kpt)
            e_exxdiv = -madelung * self.mf.mol.nelectron / 2  # / len(self.mf.kpts)
        else:
            e_exxdiv = 0
        return (e_nuc + e_exxdiv + np.sum(h1e * dm1) + np.sum(dm2 * eri) / 2) / self.nkpts

    def test_dmet_energy_part_2dm_full_bath(self):
        """Literature DMET energy."""
        emb = self.emb(-1)
        # DMET energy:
        e_dmet = emb.get_dmet_energy(part_cumulant=False)
        self.assertAllclose(e_dmet, self.cc.e_tot / self.nkpts)
        # DMET energy from DMs:
        dm1 = emb.make_rdm1_demo(ao_basis=True)
        dm2 = emb.make_rdm2_demo(ao_basis=True, part_cumulant=False)
        e_dmet = self._energy_from_dms(dm1, dm2)
        self.assertAllclose(e_dmet, self.cc.e_tot / self.nkpts)

    def test_dmet_energy_part_cumulant_full_bath(self):
        """Improved DMET energy."""
        emb = self.emb(-1)
        # DMET energy:
        e_dmet = emb.get_dmet_energy(part_cumulant=True)
        self.assertAllclose(e_dmet, self.cc.e_tot / self.nkpts)
        # DMET energy from DMs:
        dm1 = emb.make_rdm1_demo(ao_basis=True)
        dm2 = emb.make_rdm2_demo(ao_basis=True, part_cumulant=True)
        e_dmet = self._energy_from_dms(dm1, dm2)
        self.assertAllclose(e_dmet, self.cc.e_tot / self.nkpts)

    e_ref_dmet_part_2dm = -76.16067576457968

    def test_dmet_energy_part_2dm_dmet_bath(self):
        """Literature DMET energy."""
        emb = self.emb(np.inf)
        # DMET energy:
        e_dmet = emb.get_dmet_energy(part_cumulant=False)
        self.assertAllclose(e_dmet, self.e_ref_dmet_part_2dm)
        # DMET energy from DMs:
        dm1 = emb.make_rdm1_demo(ao_basis=True)
        dm2 = emb.make_rdm2_demo(ao_basis=True, part_cumulant=False)
        e_dmet = self._energy_from_dms(dm1, dm2)
        self.assertAllclose(e_dmet, self.e_ref_dmet_part_2dm)

    e_ref_dmet_part_cumulant = -76.12825899582506

    def test_dmet_energy_part_cumulant_dmet_bath(self):
        """Literature DMET energy."""
        emb = self.emb(np.inf)
        # DMET energy:
        e_dmet = emb.get_dmet_energy(part_cumulant=True)
        self.assertAllclose(e_dmet, self.e_ref_dmet_part_cumulant)
        # DMET energy from DMs:
        dm1 = emb.make_rdm1_demo(ao_basis=True)
        dm2 = emb.make_rdm2_demo(ao_basis=True, part_cumulant=True)
        e_dmet = self._energy_from_dms(dm1, dm2)
        self.assertAllclose(e_dmet, self.e_ref_dmet_part_cumulant)


class Test_Unrestricted(Test_Restricted):
    @classmethod
    def setUpClass(cls):
        cls.mf = systems.water_cation_631g.uhf()
        cls.cc = systems.water_cation_631g.uccsd()

    def _energy_from_dms(self, dm1, dm2):
        dm1 = dm1[0] + dm1[1]
        dm2 = dm2[0] + 2 * dm2[1] + dm2[2]
        return super()._energy_from_dms(dm1, dm2)

    e_ref_dmet_part_2dm = -75.6952601321852
    e_ref_dmet_part_cumulant = -75.69175122675334


class Test_PBC_Restricted(Test_Restricted):
    @classmethod
    def setUpClass(cls):
        cls.mf = systems.h2_sto3g_s311.rhf()
        cls.kmf = systems.h2_sto3g_k311.rhf()
        cls.cc = systems.h2_sto3g_s311.rccsd()

    e_ref_dmet_part_2dm = -3.848969147919312 / 3
    e_ref_dmet_part_cumulant = -3.8505073633380364 / 3


class Test_PBC_Unrestricted(Test_Unrestricted):
    @classmethod
    def setUpClass(cls):
        cls.mf = systems.h3_sto3g_s311.uhf()
        cls.kmf = systems.h3_sto3g_k311.uhf()
        cls.cc = systems.h3_sto3g_s311.uccsd()

    e_ref_dmet_part_2dm = -1.7456278335868185
    e_ref_dmet_part_cumulant = -1.7461038863675442


if __name__ == "__main__":
    print("Running %s" % __file__)
    unittest.main()
