import unittest

import numpy as np
from vayesta.core.bath import DMET_Bath
from vayesta.core.bath import EwDMET_Bath
from vayesta.core.bath import MP2_Bath, RPA_Bath
from vayesta.core.qemb import Embedding
from vayesta.core.qemb import UEmbedding
from vayesta.tests.common import TestCase
from vayesta.tests import testsystems


class EwDMET_Bath_Test(TestCase):
    def test_ewdmet_bath(self):
        """The EwDMET bath of order k reproduces the fragment moments of the Fock matrix
        (projected into the occupied or virtual environment) up to order 2k+1."""
        mf = testsystems.ethanol_ccpvdz.rhf()

        emb = Embedding(mf)
        with emb.iao_fragmentation() as f:
            frag = f.add_atomic_fragment(0)
        dmet_bath = DMET_Bath(frag, dmet_threshold=1e-8)
        dmet_bath.kernel()

        fock = mf.get_fock()
        c_frag = frag.c_frag
        nfrag = c_frag.shape[-1]

        def get_moments(c, nmom):
            f = np.linalg.multi_dot((c.T, fock, c))
            moms = [np.eye(f.shape[0])]
            for n in range(1, nmom):
                moms.append(np.dot(moms[-1], f))
            return np.asarray([m[:nfrag, :nfrag] for m in moms])

        for occtype in ("occupied", "virtual"):
            c_env = dmet_bath.c_env_occ if occtype == "occupied" else dmet_bath.c_env_vir
            mom_full = get_moments(np.hstack((c_frag, c_env)), 10)
            ewdmet_bath = EwDMET_Bath(frag, dmet_bath, occtype, max_order=4)
            for kmax in range(0, 4):
                c_bath = ewdmet_bath.get_bath(kmax)[0]
                c_cluster = np.hstack((c_frag, c_bath))
                n_cluster = c_cluster.shape[-1]
                self.assertAllclose(np.linalg.multi_dot((c_cluster.T, mf.get_ovlp(), c_cluster)), np.eye(n_cluster))
                mom_cluster = get_moments(c_cluster, 2 * kmax + 2)
                self.assertAllclose(mom_cluster, mom_full[: 2 * kmax + 2], atol=1e-7, rtol=1e-7)


class MP2_BNO_Test(TestCase):
    def test_bno_Bath(self):
        rhf = testsystems.ethanol_ccpvdz.rhf()

        remb = Embedding(rhf)
        with remb.iao_fragmentation() as f:
            rfrag = f.add_atomic_fragment("O")
        rdmet_bath = DMET_Bath(rfrag)
        rdmet_bath.kernel()
        rbno_bath_occ = MP2_Bath(rfrag, rdmet_bath, occtype="occupied")
        rbno_bath_vir = MP2_Bath(rfrag, rdmet_bath, occtype="virtual")

        uhf = testsystems.ethanol_ccpvdz.uhf()
        uemb = UEmbedding(uhf)
        with uemb.iao_fragmentation() as f:
            ufrag = f.add_atomic_fragment("O")
        udmet_bath = DMET_Bath(ufrag)
        udmet_bath.kernel()
        ubno_bath_occ = MP2_Bath(ufrag, udmet_bath, occtype="occupied")
        ubno_bath_vir = MP2_Bath(ufrag, udmet_bath, occtype="virtual")

        # Check maximum, minimum, and mean occupations
        n_occ_max = 0.005243099445814127
        n_occ_min = 2.9822620128851076e-06
        n_occ_mean = 0.0018101294711391177
        n_vir_max = 0.00828117541051843
        n_vir_min = 2.0353121374248057e-09
        n_vir_mean = 0.0005582689971478813
        # RHF
        self.assertAlmostEqual(np.amax(rbno_bath_occ.occup), n_occ_max)
        self.assertAlmostEqual(np.amin(rbno_bath_occ.occup), n_occ_min)
        self.assertAlmostEqual(np.mean(rbno_bath_occ.occup), n_occ_mean)
        self.assertAlmostEqual(np.amax(rbno_bath_vir.occup), n_vir_max)
        self.assertAlmostEqual(np.amin(rbno_bath_vir.occup), n_vir_min)
        self.assertAlmostEqual(np.mean(rbno_bath_vir.occup), n_vir_mean)
        # UHF
        self.assertAlmostEqual(np.amax(ubno_bath_occ.occup[0]), n_occ_max)
        self.assertAlmostEqual(np.amax(ubno_bath_occ.occup[1]), n_occ_max)
        self.assertAlmostEqual(np.amin(ubno_bath_occ.occup[0]), n_occ_min)
        self.assertAlmostEqual(np.amin(ubno_bath_occ.occup[1]), n_occ_min)
        self.assertAlmostEqual(np.mean(ubno_bath_occ.occup[0]), n_occ_mean)
        self.assertAlmostEqual(np.mean(ubno_bath_occ.occup[1]), n_occ_mean)
        self.assertAlmostEqual(np.amax(ubno_bath_vir.occup[0]), n_vir_max)
        self.assertAlmostEqual(np.amax(ubno_bath_vir.occup[1]), n_vir_max)
        self.assertAlmostEqual(np.amin(ubno_bath_vir.occup[0]), n_vir_min)
        self.assertAlmostEqual(np.amin(ubno_bath_vir.occup[1]), n_vir_min)
        self.assertAlmostEqual(np.mean(ubno_bath_vir.occup[0]), n_vir_mean)
        self.assertAlmostEqual(np.mean(ubno_bath_vir.occup[1]), n_vir_mean)

        # Compare RHF and UHF
        self.assertAllclose(rbno_bath_occ.occup, ubno_bath_occ.occup[0])
        self.assertAllclose(rbno_bath_occ.occup, ubno_bath_occ.occup[1])
        self.assertAllclose(rbno_bath_vir.occup, ubno_bath_vir.occup[0])
        self.assertAllclose(rbno_bath_vir.occup, ubno_bath_vir.occup[1])

    def test_project_dmet(self):
        rhf = testsystems.ethanol_ccpvdz.rhf()
        uhf = testsystems.ethanol_ccpvdz.uhf()

        remb = Embedding(rhf)
        with remb.iao_fragmentation() as f:
            rfrag = f.add_atomic_fragment("O")
        rdmet_bath = DMET_Bath(rfrag)
        rdmet_bath.kernel()
        rbno_bath_occ = MP2_Bath(rfrag, rdmet_bath, project_dmet="linear", occtype="occupied")
        rbno_bath_vir = MP2_Bath(rfrag, rdmet_bath, project_dmet="linear", occtype="virtual")

        uemb = UEmbedding(uhf)
        with uemb.iao_fragmentation() as f:
            ufrag = f.add_atomic_fragment("O")
        udmet_bath = DMET_Bath(ufrag)
        udmet_bath.kernel()
        ubno_bath_occ = MP2_Bath(ufrag, udmet_bath, project_dmet="linear", occtype="occupied")
        ubno_bath_vir = MP2_Bath(ufrag, udmet_bath, project_dmet="linear", occtype="virtual")

        # Check maximum, minimum, and mean occupations
        n_occ_mean = 7.377564076299791e-05
        n_vir_mean = 0.0005062785110509356
        # RHF
        self.assertAlmostEqual(np.mean(rbno_bath_occ.occup), n_occ_mean)
        self.assertAlmostEqual(np.mean(rbno_bath_vir.occup), n_vir_mean)
        # UHF
        self.assertAlmostEqual(np.mean(ubno_bath_occ.occup[0]), n_occ_mean)
        self.assertAlmostEqual(np.mean(ubno_bath_occ.occup[1]), n_occ_mean)
        self.assertAlmostEqual(np.mean(ubno_bath_vir.occup[0]), n_vir_mean)
        self.assertAlmostEqual(np.mean(ubno_bath_vir.occup[1]), n_vir_mean)
        # Compare RHF and UHF
        self.assertAllclose(rbno_bath_occ.occup, ubno_bath_occ.occup[0])
        self.assertAllclose(rbno_bath_occ.occup, ubno_bath_occ.occup[1])
        self.assertAllclose(rbno_bath_vir.occup, ubno_bath_vir.occup[0])
        self.assertAllclose(rbno_bath_vir.occup, ubno_bath_vir.occup[1])


class RPA_Test(TestCase):
    def test_bno_Bath(self):
        rhf = testsystems.ethanol_631g_df.rhf()

        remb = Embedding(rhf)
        with remb.iao_fragmentation() as f:
            rfrag = f.add_atomic_fragment("O")
        rdmet_bath = DMET_Bath(rfrag)
        rdmet_bath.kernel()
        rbno_bath_occ = RPA_Bath(rfrag, rdmet_bath, occtype="occupied")
        rbno_bath_vir = RPA_Bath(rfrag, rdmet_bath, occtype="virtual")

        # Check maximum, minimum, and mean occupations
        n_occ_max = 0.008502908095347851
        n_occ_min = 0.000007766938370992
        n_occ_mean = 0.002300040902721369
        n_vir_max = 0.056091218249012025
        n_vir_min = 0.000122187516348096
        n_vir_mean = 0.009570044321105821

        # RHF
        self.assertAlmostEqual(np.amax(rbno_bath_occ.occup), n_occ_max)
        self.assertAlmostEqual(np.amin(rbno_bath_occ.occup), n_occ_min)
        self.assertAlmostEqual(np.mean(rbno_bath_occ.occup), n_occ_mean)
        self.assertAlmostEqual(np.amax(rbno_bath_vir.occup), n_vir_max)
        self.assertAlmostEqual(np.amin(rbno_bath_vir.occup), n_vir_min)
        self.assertAlmostEqual(np.mean(rbno_bath_vir.occup), n_vir_mean)


if __name__ == "__main__":
    print("Running %s" % __file__)
    unittest.main()
