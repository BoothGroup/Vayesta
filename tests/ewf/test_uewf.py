import unittest

import numpy as np

import pyscf
import pyscf.gto
import pyscf.scf
import pyscf.cc

import vayesta
import vayesta.ewf

class Test(unittest.TestCase):
    pass

class H2O_Test(Test):

    @classmethod
    def setUpClass(cls):
        cls.mol = pyscf.gto.Mole()
        cls.mol.atom = """
        O  0.0000   0.0000   0.1173
        H  0.0000   0.7572  -0.4692
        H  0.0000  -0.7572  -0.4692
        """
        cls.mol.basis = 'cc-pVDZ'
        cls.mol.build()

        # Hartree-Fock
        cls.rhf = pyscf.scf.RHF(cls.mol)
        cls.rhf.kernel()
        assert cls.rhf.converged
        cls.uhf = cls.rhf.to_uhf()

    @classmethod
    def tearDownClass(cls):
        del cls.mol, cls.rhf, cls.uhf

    def test_rewf_vs_uewf_no_bath(self):
        rcc = vayesta.ewf.EWF(self.rhf, bath_type=None)
        rcc.iao_fragmentation()
        rcc.add_all_atomic_fragments()
        rcc.kernel()
        ucc = vayesta.ewf.EWF(self.uhf, bath_type=None)
        ucc.iao_fragmentation()
        ucc.add_all_atomic_fragments()
        ucc.kernel()
        self.assertAlmostEqual(rcc.e_tot, ucc.e_tot, 7)

    def test_rewf_vs_uewf_full_bath(self):
        rcc = vayesta.ewf.EWF(self.rhf, bath_type='full')
        rcc.iao_fragmentation()
        rcc.add_all_atomic_fragments()
        rcc.kernel()
        ucc = vayesta.ewf.EWF(self.uhf, bath_type='full')
        ucc.iao_fragmentation()
        ucc.add_all_atomic_fragments()
        ucc.kernel()
        cc = pyscf.cc.CCSD(self.rhf)
        cc.kernel()
        self.assertAlmostEqual(rcc.e_tot, cc.e_tot, 7)
        self.assertAlmostEqual(ucc.e_tot, cc.e_tot, 7)
        self.assertAlmostEqual(rcc.e_tot, ucc.e_tot, 7)

    def test_rewf_vs_uewf_mp2_bath(self):
        rcc = vayesta.ewf.EWF(self.rhf, bno_threshold=1e-4)
        rcc.iao_fragmentation()
        rcc.add_all_atomic_fragments()
        rcc.kernel()
        ucc = vayesta.ewf.EWF(self.uhf, bno_threshold=1e-4)
        ucc.iao_fragmentation()
        ucc.add_all_atomic_fragments()
        ucc.kernel()
        self.assertAlmostEqual(rcc.e_tot, ucc.e_tot, 7)

class HF_Anion_Test(Test):

    @classmethod
    def setUpClass(cls):
        cls.mol = pyscf.gto.Mole()
        cls.mol.atom = """
        H  0.0000   0.0000   0.0000
        F  0.0000   0.0000   2.0000
        """
        cls.mol.basis = 'aug-cc-pVDZ'
        cls.mol.charge = -1
        cls.mol.spin = 1
        cls.mol.build()

        # Hartree-Fock
        cls.uhf = pyscf.scf.UHF(cls.mol)
        cls.uhf.kernel()
        assert cls.uhf.converged

        # Full CCSD
        cls.cc_full = pyscf.cc.UCCSD(cls.uhf)
        cls.cc_full.kernel()

    @classmethod
    def tearDownClass(cls):
        del cls.mol, cls.uhf, cls.cc_full

    def test_rewf_vs_uewf_full_bath(self):
        ucc = vayesta.ewf.EWF(self.uhf, bath_type='full')
        ucc.iao_fragmentation()
        ucc.add_all_atomic_fragments()
        ucc.kernel()
        self.assertAlmostEqual(ucc.e_tot, self.cc_full.e_tot, 7)

    def test_rewf_vs_uewf_full_mp2_bath(self):
        ucc = vayesta.ewf.EWF(self.uhf, bno_threshold=-1)
        ucc.iao_fragmentation()
        ucc.add_all_atomic_fragments()
        ucc.kernel()
        self.assertAlmostEqual(ucc.e_tot, self.cc_full.e_tot, 7)

    def test_rewf_vs_uewf_mp2_bath(self):
        ucc = vayesta.ewf.EWF(self.uhf, bno_threshold=-1)
        ucc.iao_fragmentation()
        ucc.add_all_atomic_fragments()
        ucc.kernel()
        self.assertAlmostEqual(ucc.e_tot, -100.17357855683768, 7)


class H2_Test(Test):

    @classmethod
    def setUpClass(cls):
        cls.mol = pyscf.gto.Mole()
        cls.mol.atom = 'H 0 0 0 ; H 0 0 0.74'
        cls.mol.basis = 'cc-pVDZ'
        cls.mol.build()
        # HF
        cls.rhf = pyscf.scf.RHF(cls.mol)
        cls.rhf.kernel()
        assert cls.rhf.converged
        cls.uhf = cls.rhf.to_uhf()

    @classmethod
    def tearDownClass(cls):
        del cls.mol, cls.rhf, cls.uhf

    def test_cisd_ccsd_fci_no_bath(self):

        def make_fragments(emb):
            emb.iao_fragmentation()
            emb.add_atomic_fragment(0)
            emb.add_atomic_fragment(1)

        # CISD
        # Restricted
        rcisd = vayesta.ewf.EWF(self.rhf, solver='CISD', bath_type='full')
        make_fragments(rcisd)
        rcisd.kernel()
        # Unrestricted
        ucisd = vayesta.ewf.UEWF(self.uhf, solver='CISD', bath_type='full')
        make_fragments(ucisd)
        ucisd.kernel()

        # FCI
        # Restricted
        rfci = vayesta.ewf.EWF(self.rhf, solver='FCI', bath_type='full')
        make_fragments(rfci)
        rfci.kernel()
        # Unrestricted
        ufci = vayesta.ewf.UEWF(self.uhf, solver='FCI', bath_type='full')
        make_fragments(ufci)
        ufci.kernel()

        # CCSD
        # Restricted
        rccsd = vayesta.ewf.EWF(self.rhf, bath_type='full')
        make_fragments(rccsd)
        rccsd.kernel()
        # Unrestricted
        uccsd = vayesta.ewf.UEWF(self.uhf, bath_type='full')
        make_fragments(uccsd)
        uccsd.kernel()

        e_tot = -1.163374496404817
        self.assertAlmostEqual(rcisd.e_tot, e_tot, 7)
        self.assertAlmostEqual(ucisd.e_tot, e_tot, 7)
        self.assertAlmostEqual(rfci.e_tot,  e_tot, 7)
        self.assertAlmostEqual(ufci.e_tot,  e_tot, 7)
        self.assertAlmostEqual(rccsd.e_tot, e_tot, 7)
        self.assertAlmostEqual(uccsd.e_tot, e_tot, 7)


if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
