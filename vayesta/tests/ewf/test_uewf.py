
import unittest

import numpy as np

import pyscf
import pyscf.gto
import pyscf.scf
import pyscf.cc

import vayesta
import vayesta.ewf

class Test(unittest.TestCase):
    mf_conv_tol = 1e-12
    cc_conv_tol = 1e-9
    places = 7


class H2_DZ_Sym(Test):

    def setUp(self):
        self.mol = pyscf.gto.Mole()
        self.mol.atom = 'Li 0 0 0 ; Li 0 0 1.0'
        #self.mol.basis = 'cc-pVDZ'
        self.mol.basis = 'STO-6G'
        #self.mol.verbose = 10
        #self.mol.verbose = 4
        self.mol.build()
        # RHF
        self.rhf = pyscf.scf.RHF(self.mol)
        self.rhf.conv_tol = self.mf_conv_tol
        self.rhf.kernel()
        assert self.rhf.converged
        # UHF
        self.uhf = self.rhf.to_uhf()

    def tearDown(self):
        del self.mol, self.rhf, self.uhf

    def test_rccsd_vs_uccsd(self):

        def make_fragments(emb):
            emb.add_atomic_fragment(0)
            emb.add_atomic_fragment(1)

        # CISD
        # Restricted
        rcisd = vayesta.ewf.EWF(self.rhf, solver='CISD', bath_type=None, solver_options={'conv_tol': self.cc_conv_tol})
        rcisd.sao_fragmentation()
        make_fragments(rcisd)
        rcisd.kernel()
        # Unrestricted
        ucisd = vayesta.ewf.UEWF(self.uhf, solver='CISD', bath_type=None, solver_options={'conv_tol': self.cc_conv_tol})
        ucisd.sao_fragmentation()
        make_fragments(ucisd)
        ucisd.kernel()

        # CCSD
        # Restricted
        rccsd = vayesta.ewf.EWF(self.rhf, bath_type=None, solver_options={'conv_tol': self.cc_conv_tol})
        rccsd.sao_fragmentation()
        make_fragments(rccsd)
        rccsd.kernel()
        # Unrestricted
        uccsd = vayesta.ewf.UEWF(self.uhf, bath_type=None, solver_options={'conv_tol': self.cc_conv_tol})
        uccsd.sao_fragmentation()
        make_fragments(uccsd)
        uccsd.kernel()

        # FCI
        # Restricted
        rfci = vayesta.ewf.EWF(self.rhf, solver='FCI', bath_type=None, solver_options={'conv_tol': self.cc_conv_tol})
        rfci.sao_fragmentation()
        make_fragments(rfci)
        rfci.kernel()
        # Unrestricted
        ufci = vayesta.ewf.UEWF(self.uhf, solver='FCI', bath_type=None, solver_options={'conv_tol': self.cc_conv_tol})
        ufci.sao_fragmentation()
        make_fragments(ufci)
        ufci.kernel()

        #gaa, gab, gbb = ufci.debug_eris
        #ha, hb = ufci.debug_h_eff
        #import pyscf.mcscf
        ##ucasci = pyscf.mcscf.ucasci.UCASCI(self.uhf, 10, (1, 1))
        #ucasci = pyscf.mcscf.ucasci.UCASCI(self.uhf, self.mol.nao_nr(), (2, 2))
        ##ucasci.verbose = 4
        #mo = ufci.fragments[0].cluster.coeff
        #mo2 = ufci.fragments[0].cluster.c_active
        #assert (mo[0] == mo2[0]).all()
        #assert (mo[1] == mo2[1]).all()
        #g2aa, g2ab, g2bb = ucasci.ao2mo(mo_coeff=mo)
        #print(np.linalg.norm(gaa - g2aa))
        #print(np.linalg.norm(gbb - g2bb))
        #print(np.linalg.norm(gab - g2ab))

        #h2a, h2b = ucasci.get_h1eff(mo_coeff=mo)[0]
        #print(ha.shape)
        #print(hb.shape)
        #print(h2a.shape)
        #print(h2b.shape)

        #print(np.linalg.norm(ha - h2a))
        #print(np.linalg.norm(hb - h2b))

        #e_tot = ucasci.kernel()[0]
        #print(e_tot)

        print("Ecorr(RCISD)= %+16.8f Ha" % rcisd.e_tot)
        print("Ecorr(UCISD)= %+16.8f Ha" % ucisd.e_tot)
        print("Ecorr(RCCSD)= %+16.8f Ha" % rccsd.e_tot)
        print("Ecorr(UCCSD)= %+16.8f Ha" % uccsd.e_tot)
        print("Ecorr(RFCI)=  %+16.8f Ha" % rfci.e_tot)
        print("Ecorr(UFCI)=  %+16.8f Ha" % ufci.e_tot)

        #self.assertAlmostEqual(rewf.e_tot, uewf.e_tot, 7)



class LiH_STO6G_Sym(Test):

    def setUp(self):
        self.mol = pyscf.gto.Mole()
        #self.mol.atom = 'Li 0 0 0 ; H 0 0 1.4'
        self.mol.atom = 'H 0 0 0 ; H 0 0 1.4'
        #self.mol.basis = 'STO6G'
        self.mol.basis = 'cc-pVDZ'
        self.mol.verbose = 10
        self.mol.build()
        # RHF
        self.rhf = pyscf.scf.RHF(self.mol)
        self.rhf.conv_tol = self.mf_conv_tol
        self.rhf.kernel()
        assert self.rhf.converged
        # UHF
        self.uhf = self.rhf.to_uhf()

    def tearDown(self):
        del self.mol, self.rhf, self.uhf

    #def test_rhf_vs_uhf(self):
    #    # Restricted
    #    rewf = vayesta.ewf.EWF(self.rhf, bath_type=None, solver_options={'conv_tol': self.cc_conv_tol})
    #    rewf.sao_fragmentation()
    #    rewf.add_all_atomic_fragments()
    #    rewf.kernel()
    #    # Unrestricted
    #    uewf = vayesta.ewf.UEWF(self.uhf, bath_type=None, solver_options={'conv_tol': self.cc_conv_tol})
    #    uewf.sao_fragmentation()
    #    uewf.add_all_atomic_fragments()
    #    uewf.kernel()
    #    self.assertAlmostEqual(rewf.e_tot, uewf.e_tot, 7)

    def test_fci(self):
        solver_opts = {'conv_tol': self.cc_conv_tol}
        rewf = vayesta.ewf.EWF(self.rhf, solver='FCI', bath_type=None, solver_options=solver_opts)
        rewf.sao_fragmentation()
        rewf.add_atomic_fragment([0, 1])
        rewf.kernel()
        uewf = vayesta.ewf.UEWF(self.uhf, solver='FCI', bath_type=None, solver_options=solver_opts)
        uewf.sao_fragmentation()
        uewf.add_atomic_fragment([0, 1])
        uewf.kernel()
        self.assertAlmostEqual(rewf.e_tot, uewf.e_tot, 7)
        self.assertAlmostEqual(rewf.e_corr, uewf.e_corr, 7)


class LiH_DZ_Sym(Test):

    def setUp(self):
        self.mol = pyscf.gto.Mole()
        self.mol.atom = 'Li 0 0 0 ; H 0 0 2'
        self.mol.basis = 'cc-pVDZ'
        self.mol.build()
        # RHF
        self.rhf = pyscf.scf.RHF(self.mol)
        self.rhf.conv_tol = self.mf_conv_tol
        self.rhf.kernel()
        assert self.rhf.converged
        # UHF
        self.uhf = self.rhf.to_uhf()

    def tearDown(self):
        del self.mol, self.rhf, self.uhf

    #def test_rccsd_vs_uccsd(self):
    #    rccsd = pyscf.cc.RCCSD(self.rhf)
    #    rccsd.kernel()
    #    uccsd = pyscf.cc.UCCSD(self.uhf)
    #    uccsd.kernel()
    #    self.assertAlmostEqual(rccsd.e_corr, uccsd.e_corr, 10)
    #    self.assertAlmostEqual(rccsd.e_tot, uccsd.e_tot, 10)

    def test_rhf_vs_uhf(self):
        solver_opts = {'conv_tol': self.cc_conv_tol, 'conv_tol_normt' : self.cc_conv_tol}
        # Restricted
        rewf = vayesta.ewf.EWF(self.rhf, bath_type=None, solver_options=solver_opts)
        rewf.sao_fragmentation()
        #rewf.add_all_atomic_fragments()
        rewf.add_atomic_fragment([0,1])
        rewf.kernel()

        uewf = vayesta.ewf.UEWF(self.uhf, bath_type=None, solver_options=solver_opts)
        uewf.sao_fragmentation()
        #uewf.add_all_atomic_fragments()
        uewf.add_atomic_fragment([0,1])
        uewf.kernel()

        #self.assertAlmostEqual(rewf.e_tot, -7.99502192669842, 6)
        #self.assertAlmostEqual(uewf.e_tot, -7.99502192669842, 4)
        self.assertAlmostEqual(rewf.e_tot, uewf.e_tot, 7)

#class LiH(Test):
#
#    def setUp(self):
#        self.mol = pyscf.gto.Mole()
#        self.mol.atom = 'Li 0 0 0 ; H 0 0 3'
#        self.mol.basis = 'cc-pVDZ'
#        #self.mol.verbose = 4
#        self.mol.build()
#        # UHF
#        self.uhf = pyscf.scf.UHF(self.mol)
#        self.uhf.conv_tol = self.mf_conv_tol
#        dm0 = self.uhf.get_init_guess()
#        dm0[0] += 0.1
#        dm0[1] -= 0.1
#        self.uhf.kernel(dm0=dm0)
#        assert self.uhf.converged
#
#    def tearDown(self):
#        del self.mol, self.uhf
#
#    def test_uwef_vs_uccsd(self):
#        uccsd = pyscf.cc.UCCSD(self.uhf)
#        uccsd.conv_tol = self.cc_conv_tol
#        uccsd.max_cycle = 100
#        uccsd.kernel()
#
#        uewf = vayesta.ewf.UEWF(self.uhf, bath_type=None, solver_options={'conv_tol': self.cc_conv_tol})
#        uewf.sao_fragmentation()
#        uewf.add_atomic_fragment([0, 1])
#        uewf.kernel()
#
#        # TODO: why low precision ?
#        self.assertAlmostEqual(uewf.e_corr, uccsd.e_corr, 4)
#        self.assertAlmostEqual(uewf.e_tot, uccsd.e_tot, 4)
#
#class LiH_STO6G(Test):
#
#    def setUp(self):
#        self.mol = pyscf.gto.Mole()
#        self.mol.atom = 'Li 0 0 0 ; H 0 0 3'
#        self.mol.basis = 'STO-6G'
#        #self.mol.verbose = 4
#        self.mol.build()
#        # UHF
#        self.uhf = pyscf.scf.UHF(self.mol)
#        self.uhf.conv_tol = self.mf_conv_tol
#        dm0 = self.uhf.get_init_guess()
#        dm0[0] += 0.1
#        dm0[1] -= 0.1
#        self.uhf.kernel(dm0=dm0)
#        assert self.uhf.converged
#
#    def test_fci(self):
#        uewf = vayesta.ewf.UEWF(self.uhf, solver='FCI', bath_type=None, solver_options={'conv_tol': self.cc_conv_tol})
#        uewf.sao_fragmentation()
#        uewf.add_atomic_fragment([0, 1])
#        uewf.kernel()


if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
