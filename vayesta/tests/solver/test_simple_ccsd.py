import pytest
import unittest
import numpy as np

import pyscf
import pyscf.ao2mo
import pyscf.cc

import vayesta

from vayesta.tests.common import TestCase
from vayesta.tests import testsystems

import vayesta.solver.simple


@pytest.mark.fast
class TestRSpin(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.mf = testsystems.water_631g.rhf()
        cls.cc = testsystems.water_631g.rccsd()

    @classmethod
    def tearDownClass(cls):
        del cls.mf, cls.cc

    def test(self):
        mf = self.mf
        fock = np.diag(mf.mo_energy)
        nmo = fock.shape[-1]
        eris = pyscf.ao2mo.kernel(mf.mol, mf.mo_coeff, compact=False).reshape(4*[nmo])
        nocc = np.count_nonzero(mf.mo_occ > 0)
        ccsd = vayesta.solver.simple.CCSD(fock, eris, nocc, conv_tol=1e-10, conv_tol_normt=1e-8)
        ccsd.kernel()
        self.assertAllclose(ccsd.t1, self.cc.t1)
        self.assertAllclose(ccsd.t2, self.cc.t2)


@pytest.mark.fast
class TestUSpin(TestRSpin):

    @classmethod
    def setUpClass(cls):
        cls.mf = testsystems.water_cation_631g.uhf()
        cls.cc = testsystems.water_cation_631g.uccsd()

    def test(self):
        mf = self.mf
        fock = (np.diag(mf.mo_energy[0]), np.diag(mf.mo_energy[1]))
        nmo = (fock[0].shape[-1], fock[1].shape[-1])
        eris_aa = pyscf.ao2mo.kernel(mf.mol, mf.mo_coeff[0], compact=False).reshape(4*[nmo[0]])
        eris_bb = pyscf.ao2mo.kernel(mf.mol, mf.mo_coeff[1], compact=False).reshape(4*[nmo[1]])
        eris_ab = pyscf.ao2mo.general(mf.mol, 2*[mf.mo_coeff[0]] + 2*[mf.mo_coeff[1]], compact=False).reshape(2*[nmo[0]] + 2*[nmo[1]])
        eris = (eris_aa, eris_ab, eris_bb)
        nocc = (np.count_nonzero(mf.mo_occ[0] > 0), np.count_nonzero(mf.mo_occ[1] > 0))
        ccsd = vayesta.solver.simple.CCSD(fock, eris, nocc, conv_tol=1e-10, conv_tol_normt=1e-8)
        ccsd.kernel()
        self.assertAllclose(ccsd.t1, self.cc.t1)
        self.assertAllclose(ccsd.t2, self.cc.t2)


if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
