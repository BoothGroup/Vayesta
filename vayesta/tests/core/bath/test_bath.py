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
        return True
        mf = testsystems.ethanol_ccpvdz.rhf()

        emb = Embedding(mf)
        with emb.iao_fragmentation() as f:
            frag = f.add_atomic_fragment(0)
        dmet_bath = DMET_Bath(frag, dmet_threshold=1e-8)
        dmet_bath.kernel()

        # Exact moments
        ovlp = mf.get_ovlp()
        moments = np.arange(12)
        c_frag = frag.c_frag
        csc = np.linalg.multi_dot((c_frag.T, ovlp, mf.mo_coeff))
        mom_full = np.einsum('xi,ik,yi->kxy', csc, np.power.outer(mf.mo_energy, moments), csc)

        # Test bath orbitals up to kmax = 4
        for kmax in range(0, 4):

            ewdmet_bath = EwDMET_Bath(frag, dmet_bath, max_order=kmax)
            ewdmet_bath.kernel()

            c_cluster = np.hstack((ewdmet_bath.c_cluster_occ, ewdmet_bath.c_cluster_vir))
            n_cluster = c_cluster.shape[-1]
            assert np.allclose(np.linalg.multi_dot((c_cluster.T, ovlp, c_cluster)) - np.eye(n_cluster), 0)

            f_cluster = np.linalg.multi_dot((c_cluster.T, mf.get_fock(), c_cluster))
            e, r = np.linalg.eigh(f_cluster)
            c_cluster_mo = np.dot(c_cluster, r)

            csc = np.linalg.multi_dot((c_frag.T, ovlp, c_cluster_mo))
            mom_cluster = np.einsum('xi,ik,yi->kxy', csc, np.power.outer(e, moments), csc)

            # Check "2n + 1"
            for order in range(2*kmax + 2):
                print("Testing EwDMET bath: kmax= %d moment= %d" % (kmax, order))
                self.assertIsNone(np.testing.assert_allclose(mom_cluster[order], mom_full[order], atol=1e-7, rtol=1e-7))

class MP2_BNO_Test(TestCase):

    def test_bno_Bath(self):
        rhf = testsystems.ethanol_ccpvdz.rhf()

        remb = Embedding(rhf)
        with remb.iao_fragmentation() as f:
            rfrag = f.add_atomic_fragment('O')
        rdmet_bath = DMET_Bath(rfrag)
        rdmet_bath.kernel()
        rbno_bath_occ = MP2_Bath(rfrag, rdmet_bath, occtype='occupied')
        rbno_bath_vir = MP2_Bath(rfrag, rdmet_bath, occtype='virtual')

        uhf = testsystems.ethanol_ccpvdz.uhf()
        uemb = UEmbedding(uhf)
        with uemb.iao_fragmentation() as f:
            ufrag = f.add_atomic_fragment('O')
        udmet_bath = DMET_Bath(ufrag)
        udmet_bath.kernel()
        ubno_bath_occ = MP2_Bath(ufrag, udmet_bath, occtype='occupied')
        ubno_bath_vir = MP2_Bath(ufrag, udmet_bath, occtype='virtual')

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
            rfrag = f.add_atomic_fragment('O')
        rdmet_bath = DMET_Bath(rfrag)
        rdmet_bath.kernel()
        rbno_bath_occ = MP2_Bath(rfrag, rdmet_bath, project_dmet='linear', occtype='occupied')
        rbno_bath_vir = MP2_Bath(rfrag, rdmet_bath, project_dmet='linear', occtype='virtual')

        uemb = UEmbedding(uhf)
        with uemb.iao_fragmentation() as f:
            ufrag = f.add_atomic_fragment('O')
        udmet_bath = DMET_Bath(ufrag)
        udmet_bath.kernel()
        ubno_bath_occ = MP2_Bath(ufrag, udmet_bath, project_dmet='linear', occtype='occupied')
        ubno_bath_vir = MP2_Bath(ufrag, udmet_bath, project_dmet='linear', occtype='virtual')

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
        rhf = testsystems.water_ccpvdz_df.rhf()

        remb = Embedding(rhf)
        with remb.iao_fragmentation() as f:
            rfrag = f.add_atomic_fragment([1])
        rdmet_bath = DMET_Bath(rfrag)
        rdmet_bath.kernel()
        rbno_bath_occ = RPA_Bath(rfrag, rdmet_bath, occtype='occupied')
        rbno_bath_vir = RPA_Bath(rfrag, rdmet_bath, occtype='virtual')

        uhf = testsystems.water_ccpvdz_df.uhf()
        uemb = UEmbedding(uhf)
        with uemb.iao_fragmentation() as f:
            ufrag = f.add_atomic_fragment([1])
        udmet_bath = DMET_Bath(ufrag)
        udmet_bath.kernel()
        ubno_bath_occ = RPA_Bath(ufrag, udmet_bath, occtype='occupied')
        ubno_bath_vir = RPA_Bath(ufrag, udmet_bath, occtype='virtual')

        # Check maximum, minimum, and mean occupations
        n_occ_max = 0.016568121109225252
        n_occ_min = 0.0010439760475971453
        n_occ_mean = 0.009467167315765655
        n_vir_max = 0.013330524652430347
        n_vir_min = 0.0001553930018495784
        n_vir_mean = 0.003853626680636801

        print(np.amax(rbno_bath_occ.occup), np.amin(rbno_bath_occ.occup), np.mean(rbno_bath_occ.occup))
        print(np.amax(rbno_bath_vir.occup), np.amin(rbno_bath_vir.occup), np.mean(rbno_bath_vir.occup))

        print(np.amax(ubno_bath_occ.occup[0]), np.amin(rbno_bath_occ.occup), np.mean(rbno_bath_occ.occup))
        print(np.amax(rbno_bath_vir.occup), np.amin(rbno_bath_vir.occup), np.mean(rbno_bath_vir.occup))

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



if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
