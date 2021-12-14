import unittest

import numpy as np

import vayesta
import vayesta.core
from vayesta.core.bath import DMET_Bath
from vayesta.core.bath import EwDMET_Bath
from vayesta.core.qemb import Embedding
from vayesta.tests.cache import moles


class EwDMET_Bath_Test(unittest.TestCase):

    def test_ewdmet_bath(self):
        mf = moles['ethanol_ccpvdz']['rhf']

        emb = Embedding(mf)
        emb.iao_fragmentation()
        frag = emb.add_atomic_fragment(0)
        dmet_bath = DMET_Bath(frag, dmet_threshold=1e-8)
        dmet_bath.kernel()

        ewdmet_bath = EwDMET_Bath(frag, dmet_bath)
        ewdmet_bath.kernel()

        # Exact moments
        ovlp = mf.get_ovlp()
        moments = np.arange(12)
        c_frag = frag.c_frag
        csc = np.linalg.multi_dot((c_frag.T, ovlp, mf.mo_coeff))
        mom_full = np.einsum('xi,ik,yi->kxy', csc, np.power.outer(mf.mo_energy, moments), csc)

        # Test bath orbitals up to kmax = 4
        for kmax in range(0, 4):
            c_bath_occ, c_rest_occ = ewdmet_bath.get_occupied_bath(kmax=kmax)
            c_bath_vir, c_rest_vir = ewdmet_bath.get_virtual_bath(kmax=kmax)

            c_cluster = np.hstack((
                ewdmet_bath.dmet_bath.c_cluster_occ, c_bath_occ,
                c_bath_vir, ewdmet_bath.dmet_bath.c_cluster_vir))
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


if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
