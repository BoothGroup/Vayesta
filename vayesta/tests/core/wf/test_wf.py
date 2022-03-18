import unittest
import numpy as np

import pyscf
import pyscf.mp
import pyscf.ci
import pyscf.cc
import pyscf.fci

import vayesta
import vayesta.ewf
from vayesta.core.util import *

from vayesta.tests import cache

class TestDMs(unittest.TestCase):

    # --- CCSD

    def _test_rccsd(self, key, atol=1e-7):
        mf = cache.moles[key]['rhf']

        emb = vayesta.ewf.EWF(mf, solver='CCSD', bath_type='full', solve_lambda=True)
        emb.iao_fragmentation()
        frag = emb.add_atomic_fragment(list(range(mf.mol.natm)))
        emb.kernel()

        # Reference
        cc = pyscf.cc.CCSD(mf)
        cc.kernel()
        dm1 = cc.make_rdm1()
        dm2 = cc.make_rdm2()

        dm1f = frag.results.wf.make_rdm1()
        dm2f = frag.results.wf.make_rdm2()
        r = np.linalg.multi_dot((frag.results.wf.mo.coeff.T, emb.get_ovlp(), mf.mo_coeff))

        dm1f = einsum('ij,iI,jJ->IJ', dm1f, r, r)
        self.assertIsNone(np.testing.assert_allclose(dm1, dm1f, atol=atol))

        dm2f = einsum('ijkl,iI,jJ,kK,lL->IJKL', dm2f, r, r, r, r)
        self.assertIsNone(np.testing.assert_allclose(dm2, dm2f, atol=atol))

    def _test_uccsd(self, key, atol=1e-7):
        mf = cache.moles[key]['uhf']

        emb = vayesta.ewf.EWF(mf, solver='CCSD', bath_type='full', solve_lambda=True)
        emb.iao_fragmentation()
        frag = emb.add_atomic_fragment(list(range(mf.mol.natm)))
        emb.kernel()

        # Reference
        cc = pyscf.cc.CCSD(mf)
        cc.kernel()
        dm1 = cc.make_rdm1()
        dm2 = cc.make_rdm2()

        dm1f = frag.results.wf.make_rdm1()
        dm2f = frag.results.wf.make_rdm2()
        ra = np.linalg.multi_dot((frag.results.wf.mo.coeff[0].T, emb.get_ovlp(), mf.mo_coeff[0]))
        rb = np.linalg.multi_dot((frag.results.wf.mo.coeff[1].T, emb.get_ovlp(), mf.mo_coeff[1]))

        dm1fa = einsum('ij,iI,jJ->IJ', dm1f[0], ra, ra)
        dm1fb = einsum('ij,iI,jJ->IJ', dm1f[1], rb, rb)
        self.assertIsNone(np.testing.assert_allclose(dm1[0], dm1fa, atol=atol))
        self.assertIsNone(np.testing.assert_allclose(dm1[1], dm1fb, atol=atol))

        dm2faa = einsum('ijkl,iI,jJ,kK,lL->IJKL', dm2f[0], ra, ra, ra, ra)
        dm2fab = einsum('ijkl,iI,jJ,kK,lL->IJKL', dm2f[1], ra, ra, rb, rb)
        dm2fbb = einsum('ijkl,iI,jJ,kK,lL->IJKL', dm2f[2], rb, rb, rb, rb)
        self.assertIsNone(np.testing.assert_allclose(dm2[0], dm2faa, atol=atol))
        self.assertIsNone(np.testing.assert_allclose(dm2[1], dm2fab, atol=atol))
        self.assertIsNone(np.testing.assert_allclose(dm2[2], dm2fbb, atol=atol))

    #def test_rccsd(self):
    #    return self._test_rccsd('h4_6-31g')

    #def test_rccsd_df(self):
    #    return self._test_rccsd('h4_6-31g_df')

    #def test_uccsd(self):
    #    return self._test_uccsd('h3_6-31g')

    #def test_uccsd_df(self):
    #    return self._test_uccsd('h3_6-31g_df')

    # --- FCI

    def _test_rfci(self, key, atol=1e-7):
        mf = cache.moles[key]['rhf']

        emb = vayesta.ewf.EWF(mf, solver='FCI', bath_type='full', solve_lambda=True)
        emb.iao_fragmentation()
        frag = emb.add_atomic_fragment(list(range(mf.mol.natm)))
        emb.kernel()

        # Reference
        fci = pyscf.fci.FCI(mf)
        fci.threads = 1
        fci.conv_tol = 1e-14
        fci = pyscf.fci.addons.fix_spin_(fci, ss=0.0)
        fci.kernel()
        args = (fci.ci, mf.mol.nao, mf.mol.nelectron)
        dm1 = fci.make_rdm1(*args)
        dm2 = fci.make_rdm12(*args)[1]

        dm1f = frag.results.wf.make_rdm1()
        dm2f = frag.results.wf.make_rdm2()
        r = np.linalg.multi_dot((frag.results.wf.mo.coeff.T, emb.get_ovlp(), mf.mo_coeff))

        dm1f = einsum('ij,iI,jJ->IJ', dm1f, r, r)
        self.assertIsNone(np.testing.assert_allclose(dm1, dm1f, atol=atol))

        dm2f = einsum('ijkl,iI,jJ,kK,lL->IJKL', dm2f, r, r, r, r)
        self.assertIsNone(np.testing.assert_allclose(dm2, dm2f, atol=atol))

    def _test_ufci(self, key, atol=1e-7):
        mf = cache.moles[key]['uhf']

        emb = vayesta.ewf.EWF(mf, solver='FCI', bath_type='full', solve_lambda=True)
        emb.iao_fragmentation()
        frag = emb.add_atomic_fragment(list(range(mf.mol.natm)))
        emb.kernel()

        # Reference
        fci = pyscf.fci.FCI(mf)
        fci.threads = 1
        fci.conv_tol = 1e-14
        fci.kernel()
        norb = mf.mol.nao
        nelec = (np.count_nonzero(mf.mo_occ[0]>0), np.count_nonzero(mf.mo_occ[1]>0))
        args = (fci.ci, norb, nelec)
        dm1 = fci.make_rdm1s(*args)
        dm2 = fci.make_rdm12s(*args)[1]

        dm1f = frag.results.wf.make_rdm1()
        dm2f = frag.results.wf.make_rdm2()
        ra = np.linalg.multi_dot((frag.results.wf.mo.coeff[0].T, emb.get_ovlp(), mf.mo_coeff[0]))
        rb = np.linalg.multi_dot((frag.results.wf.mo.coeff[1].T, emb.get_ovlp(), mf.mo_coeff[1]))

        dm1fa = einsum('ij,iI,jJ->IJ', dm1f[0], ra, ra)
        dm1fb = einsum('ij,iI,jJ->IJ', dm1f[1], rb, rb)
        self.assertIsNone(np.testing.assert_allclose(dm1[0], dm1fa, atol=atol))
        self.assertIsNone(np.testing.assert_allclose(dm1[1], dm1fb, atol=atol))

        dm2faa = einsum('ijkl,iI,jJ,kK,lL->IJKL', dm2f[0], ra, ra, ra, ra)
        dm2fab = einsum('ijkl,iI,jJ,kK,lL->IJKL', dm2f[1], ra, ra, rb, rb)
        dm2fbb = einsum('ijkl,iI,jJ,kK,lL->IJKL', dm2f[2], rb, rb, rb, rb)
        self.assertIsNone(np.testing.assert_allclose(dm2[0], dm2faa, atol=atol))
        self.assertIsNone(np.testing.assert_allclose(dm2[1], dm2fab, atol=atol))
        self.assertIsNone(np.testing.assert_allclose(dm2[2], dm2fbb, atol=atol))

    #def test_rfci(self):
    #    return self._test_rfci('h4_ccpvdz')

    #def test_rfci_df(self):
    #    return self._test_rfci('h4_ccpvdz_df')

    def test_ufci(self):
        return self._test_ufci('h3_ccpvdz')

    def test_ufci_df(self):
        return self._test_ufci('h3_ccpvdz_df')


if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
