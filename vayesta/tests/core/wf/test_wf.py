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




class Test_UFCI_wf_w_dummy(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mf = testsystems.heli_631g.uhf()
        cls.ufci = testsystems.heli_631g.ufci()

    @classmethod
    def tearDownClass(cls):
        del cls.mf
        del cls.ufci

    def get_ufci_ref(self):
        return self.ufci.e_tot

    def test_ufci_w_dummy_atomic_fragmentation(self):
        emb = vayesta.ewf.EWF(self.mf)

        with emb.iao_fragmentation() as f:
            fci_frags = f.add_all_atomic_fragments(solver='FCI',
         bath_options=dict(bathtype='full'), store_wf_type='CCSDTQ', auxiliary=True)
            ccsd_frag = f.add_full_system(solver='CCSD', bath_options=dict(bathtype='full'))
        ccsd_frag.add_external_corrections(fci_frags, correction_type='external', projectors=1, low_level_coul=True)
        emb.kernel()

        self.assertAlmostEqual(emb.e_tot, self.get_ufci_ref())

    def test_ufci_w_dummy_full_system_fragment(self):
        emb = vayesta.ewf.EWF(self.mf)
        fci_frags=[]
        with emb.iao_fragmentation() as f:
            fci_frags.append(f.add_full_system(solver='FCI',
         bath_options=dict(bathtype='full'), store_wf_type='CCSDTQ', auxiliary=True))
            ccsd_frag = f.add_full_system(solver='CCSD', bath_options=dict(bathtype='full'))
        ccsd_frag.add_external_corrections(fci_frags, correction_type='external', projectors=1, low_level_coul=True)
        emb.kernel()

        self.assertAlmostEqual(emb.e_tot, self.get_ufci_ref())

    def test_ufci_w_dummy_regression_full_system_fragment(self):
        emb = vayesta.ewf.EWF(self.mf)
        fci_frags=[]
        with emb.iao_fragmentation() as f:
            fci_frags.append(f.add_full_system(solver='FCI',
                bath_options=dict(bathtype='dmet'), store_wf_type='CCSDTQ', auxiliary=True))
            ccsd_frag = f.add_full_system(solver='CCSD', bath_options=dict(bathtype='full'))
        ccsd_frag.add_external_corrections(fci_frags, correction_type='external', projectors=1, low_level_coul=True)
        emb.kernel()

        self.assertAlmostEqual(emb.e_tot, -10.290621999634174)

class Test_DM(TestCase):

    solver = 'CCSD'
    solver_opts = dict(conv_tol=1e-10, conv_tol_normt=1e-8)

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
        emb = vayesta.ewf.EWF(cls.mf, bath_type='full', solve_lambda=True, solver=cls.solver,
                solver_options=cls.solver_opts)
        with emb.iao_fragmentation() as f:
            frag = f.add_atomic_fragment(list(range(emb.mol.natm)))
        emb.kernel()
        return frag

    def get_dm1_ref(self, **kwargs):
        return self.cc.make_rdm1(**kwargs)

    def get_dm2_ref(self, **kwargs):
        return self.cc.make_rdm2(**kwargs)

    def test_dm1(self):
        frag = self.frag()
        dm1_ref = self.get_dm1_ref(ao_repr=True)
        dm1 = frag.results.wf.make_rdm1(ao_basis=True)
        self.assertAllclose(dm1, dm1_ref)
        #dm1_ref = self.get_dm1_ref(ao_repr=True, with_mf=False)
        #dm1 = frag.results.wf.make_rdm1(ao_basis=True, with_mf=False)
        #self.assertAllclose(dm1, dm1_ref)

    def test_dm2(self):
        frag = self.frag()
        dm2_ref = self.get_dm2_ref(ao_repr=True)
        dm2 = frag.results.wf.make_rdm2(ao_basis=True)
        self.assertAllclose(dm2, dm2_ref)
        #dm2_ref = self.get_dm2_ref(ao_repr=True, with_dm1=False)
        #dm2 = frag.results.wf.make_rdm2(ao_basis=True, with_dm1=False)
        #self.assertAllclose(dm2, dm2_ref)

class Test_DM_MP2(Test_DM):

    solver = 'MP2'
    solver_opts = {}

    @classmethod
    def setUpClass(cls):
        cls.mf = testsystems.water_631g.rhf()
        cls.cc = testsystems.water_631g.rmp2()

    #def test_dm2(self):
    #    pass

class Test_DM_UHF(Test_DM):

    @classmethod
    def setUpClass(cls):
        cls.mf = testsystems.water_cation_631g.uhf()
        cls.cc = testsystems.water_cation_631g.uccsd()

class Test_DM_FCI(Test_DM):

    solver = 'FCI'
    solver_opts = dict(conv_tol=1e-14)

    @classmethod
    def setUpClass(cls):
        cls.mf = testsystems.water_sto3g.rhf()
        cls.cc = testsystems.water_sto3g.rfci()

    def get_dm1_ref(self, ao_repr=False):
        args = (self.cc.ci, self.mf.mol.nao, self.mf.mol.nelectron)
        dm1_ref = self.cc.make_rdm1(*args)
        if ao_repr:
            dm1_ref = np.linalg.multi_dot((self.mf.mo_coeff, dm1_ref, self.mf.mo_coeff.T))
        return dm1_ref

    def get_dm2_ref(self, ao_repr=False):
        args = (self.cc.ci, self.mf.mol.nao, self.mf.mol.nelectron)
        dm2_ref = self.cc.make_rdm12(*args)[1]
        if ao_repr:
            dm2_ref = einsum('ijkl,ai,bj,ck,dl->abcd', dm2_ref, *(4*[self.mf.mo_coeff]))
        return dm2_ref

class Test_DM_FCI_UHF(Test_DM_FCI):

    @classmethod
    def setUpClass(cls):
        cls.mf = testsystems.water_cation_sto3g.uhf()
        cls.cc = testsystems.water_cation_sto3g.ufci()

    def get_dm1_ref(self, ao_repr=False):
        nelec = (np.count_nonzero(self.mf.mo_occ[0]>0), np.count_nonzero(self.mf.mo_occ[1]>0))
        args = (self.cc.ci, self.mf.mol.nao, nelec)
        dm1_ref = self.cc.make_rdm1s(*args)
        if ao_repr:
            dm1_ref = (np.linalg.multi_dot((self.mf.mo_coeff[0], dm1_ref[0], self.mf.mo_coeff[0].T)),
                       np.linalg.multi_dot((self.mf.mo_coeff[1], dm1_ref[1], self.mf.mo_coeff[1].T)))
        return dm1_ref

    def get_dm2_ref(self, ao_repr=False):
        nelec = (np.count_nonzero(self.mf.mo_occ[0]>0), np.count_nonzero(self.mf.mo_occ[1]>0))
        args = (self.cc.ci, self.mf.mol.nao, nelec)
        dm2_ref = self.cc.make_rdm12s(*args)[1]
        if ao_repr:
            dm2_ref = (einsum('ijkl,ai,bj,ck,dl->abcd', dm2_ref[0], *(4*[self.mf.mo_coeff[0]])),
                       einsum('ijkl,ai,bj,ck,dl->abcd', dm2_ref[1], *(2*[self.mf.mo_coeff[0]] + 2*[self.mf.mo_coeff[1]])),
                       einsum('ijkl,ai,bj,ck,dl->abcd', dm2_ref[2], *(4*[self.mf.mo_coeff[1]])))
        return dm2_ref


if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
