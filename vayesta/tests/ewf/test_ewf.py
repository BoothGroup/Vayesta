
import unittest

import pyscf
import pyscf.gto
import pyscf.scf
import pyscf.ci
import pyscf.cc

import vayesta
import vayesta.ewf

class Test(unittest.TestCase):
    mf_conv_tol = 1e-12
    cc_conv_tol = 1e-9
    places = 7


class H2_DZ(Test):

    def setUp(self):
        self.mol = pyscf.gto.Mole()
        self.mol.atom = 'H 0 0 0 ; H 0 0 1.4'
        self.mol.basis = 'cc-pVDZ'
        #self.mol.verbose = 10
        self.mol.build()
        # RHF
        self.rhf = pyscf.scf.RHF(self.mol)
        self.rhf.conv_tol = self.mf_conv_tol
        self.rhf.kernel()
        assert self.rhf.converged

    def tearDown(self):
        del self.mol, self.rhf

    def test_cisd_vs_ccsd(self):
        solver_opts = {'conv_tol': self.cc_conv_tol}
        ecisd = vayesta.ewf.EWF(self.rhf, solver='CISD', bath_type=None, solver_options=solver_opts)
        ecisd.iao_fragmentation()
        ecisd.add_all_atomic_fragments()
        ecisd.kernel()

        eccsd = vayesta.ewf.EWF(self.rhf, solver='CCSD', bath_type=None, solver_options=solver_opts)
        eccsd.iao_fragmentation()
        eccsd.add_all_atomic_fragments()
        eccsd.kernel()

        self.assertAlmostEqual(ecisd.e_corr, eccsd.e_corr)
        self.assertAlmostEqual(ecisd.e_tot, eccsd.e_tot)

        print("E_corr(CISD)= %+16.8f Ha" % ecisd.e_corr)
        print("E_corr(CCSD)= %+16.8f Ha" % eccsd.e_corr)


if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
