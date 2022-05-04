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
from vayesta.tests.common import TestCase
from vayesta.tests import testsystems

class Test_DM(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.mf = testsystems.water_631g.rhf()
        cls.cc = testsystems.water_631g.rccsd()

    @classmethod
    def tearDownClass(cls):
        del cls.mf
        del cls.cc

    @classmethod
    @cache
    def frag(cls):
        emb = vayesta.ewf.EWF(cls.mf, bath_type='full', solve_lambda=True)
        emb.iao_fragmentation()
        frag = emb.add_atomic_fragment(list(range(emb.mol.natm)))
        emb.kernel()
        return frag

    def get_dm1_ref(self):
        return self.cc.make_rdm1(ao_repr=True)

    def get_dm2_ref(self):
        return self.cc.make_rdm2(ao_repr=True)

    def test_dm1(self):
        dm1_ref = self.get_dm1_ref()
        frag = self.frag()
        dm1 = frag.results.wf.make_rdm1(ao_basis=True)
        self.assertAllclose(dm1, dm1_ref)

    def test_dm2(self):
        dm2_ref = self.get_dm2_ref()
        frag = self.frag()
        dm2 = frag.results.wf.make_rdm2(ao_basis=True)
        self.assertAllclose(dm2, dm2_ref)

class Test_DM_UHF(Test_DM):

    @classmethod
    def setUpClass(cls):
        cls.mf = testsystems.water_cation_631g.uhf()
        cls.cc = testsystems.water_cation_631g.uccsd()

class Test_DM_FCI(Test_DM):

    @classmethod
    def setUpClass(cls):
        cls.mf = testsystems.water_sto3g.rhf()
        cls.cc = testsystems.water_sto3g.rfci()

    @classmethod
    @cache
    def frag(cls):
        emb = vayesta.ewf.EWF(cls.mf, bath_type='full', solver='FCI')
        emb.iao_fragmentation()
        frag = emb.add_atomic_fragment(list(range(emb.mol.natm)))
        emb.kernel()
        return frag

    def get_dm1_ref(self):
        args = (self.cc.ci, self.mf.mol.nao, self.mf.mol.nelectron)
        dm1_ref = self.cc.make_rdm1(*args)
        dm1_ref = np.linalg.multi_dot((self.mf.mo_coeff, dm1_ref, self.mf.mo_coeff.T))
        return dm1_ref

    def get_dm2_ref(self):
        args = (self.cc.ci, self.mf.mol.nao, self.mf.mol.nelectron)
        dm2_ref = self.cc.make_rdm12(*args)[1]
        dm2_ref = einsum('ijkl,ai,bj,ck,dl->abcd', dm2_ref, *(4*[self.mf.mo_coeff]))
        return dm2_ref

class Test_DM_FCI_UHF(Test_DM_FCI):

    @classmethod
    def setUpClass(cls):
        cls.mf = testsystems.water_cation_sto3g.uhf()
        cls.cc = testsystems.water_cation_sto3g.ufci()

    def get_dm1_ref(self):
        nelec = (np.count_nonzero(self.mf.mo_occ[0]>0), np.count_nonzero(self.mf.mo_occ[1]>0))
        args = (self.cc.ci, self.mf.mol.nao, nelec)
        dm1_ref = self.cc.make_rdm1s(*args)
        dm1_ref = (np.linalg.multi_dot((self.mf.mo_coeff[0], dm1_ref[0], self.mf.mo_coeff[0].T)),
                   np.linalg.multi_dot((self.mf.mo_coeff[1], dm1_ref[1], self.mf.mo_coeff[1].T)))
        return dm1_ref

    def get_dm2_ref(self):
        nelec = (np.count_nonzero(self.mf.mo_occ[0]>0), np.count_nonzero(self.mf.mo_occ[1]>0))
        args = (self.cc.ci, self.mf.mol.nao, nelec)
        dm2_ref = self.cc.make_rdm12s(*args)[1]
        dm2_ref = (einsum('ijkl,ai,bj,ck,dl->abcd', dm2_ref[0], *(4*[self.mf.mo_coeff[0]])),
                   einsum('ijkl,ai,bj,ck,dl->abcd', dm2_ref[1], *(2*[self.mf.mo_coeff[0]] + 2*[self.mf.mo_coeff[1]])),
                   einsum('ijkl,ai,bj,ck,dl->abcd', dm2_ref[2], *(4*[self.mf.mo_coeff[1]])))
        return dm2_ref


if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
