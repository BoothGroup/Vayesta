import pytest
import unittest

import pyscf

import vayesta
import vayesta.ewf
from vayesta.solver.ccsd import UCCSD
from vayesta.solver.cisd import UCISD

from vayesta.tests.common import TestCase
from vayesta.tests import testsystems

def fci_solver(mf, dm=False):
    h1e = mf.get_hcore()
    h2e = mf._eri
    norb = h1e[0].shape[-1]
    nelec = mf.mol.nelec
    if type(mf.mo_coeff) == tuple:
        energy, civec = pyscf.fci.direct_uhf.kernel(h1e, h2e, norb, nelec, conv_tol=1.e-14)
        if dm:
            dm1, dm2 = pyscf.fci.direct_uhf.make_rdm12s(civec, norb, nelec)
            return dict(dm1=dm1, dm2=dm2, converged=True, energy=energy)
    else:
        energy, civec = pyscf.fci.direct_spin0.kernel(h1e, h2e, norb, nelec, conv_tol=1.e-14)
        if dm:
            dm1, dm2 = pyscf.fci.direct_spin0.make_rdm12(civec, norb, nelec)
            return dict(dm1=dm1, dm2=dm2, converged=True, energy=energy)

    return dict(civec=civec, converged=True, energy=energy)

def ccsd_solver(mf, dm=False):
    if type(mf.mo_coeff) == tuple:
        cc = UCCSD(mf)
    else:
        cc = pyscf.cc.CCSD(mf)
    cc.kernel()
    t1, t2 = cc.t1, cc.t2
    l1, l2 = cc.solve_lambda()
    if dm:
        dm1, dm2, = cc.make_rdm1(), cc.make_rdm2()
        return dict(dm1=dm1, dm2=dm2, converged=True, energy=cc.e_corr)
    else:
        return dict(t1=t1, t2=t2, l1=l1, l2=l2, converged=True, energy=cc.e_corr)
    
def cisd_solver(mf, dm=False):
    if type(mf.mo_coeff) == tuple:
        ci = UCISD(mf)
    else:
        ci = pyscf.ci.CISD(mf)
    energy, civec = ci.kernel()
    c0, c1, c2 = ci.cisdvec_to_amplitudes(civec)
    if dm:
        dm1, dm2 = ci.make_rdm1(), ci.make_rdm2()
        return dict(dm1=dm1, dm2=dm2, converged=True, energy=ci.e_corr)
    else:
        return dict(c0=c0, c1=c1, c2=c2, converged=True, energy=ci.e_corr)
    
callbacks = dict(FCI=fci_solver, CCSD=ccsd_solver, CISD=cisd_solver)

class TestSolvers(TestCase):

    def _test(self, key):
        mf = getattr(getattr(testsystems, key[0]), key[1])()
        bath_opts = dict(bathtype="dmet")
        emb = vayesta.ewf.EWF(mf, solver=key[2],  energy_functional='wf', bath_options=bath_opts)
        emb.kernel()

        for dm in [True, False]:
            if key[2] == 'CISD' and dm:
                return
            callback = lambda mf: callbacks[key[2]](mf, dm=dm)
            emb_callback = vayesta.ewf.EWF(mf, solver="CALLBACK",  energy_functional='wf', bath_options=bath_opts, solver_options=dict(callback=callback))
            emb_callback.kernel()

            functional = 'dmet' if dm else 'wf'
            self.assertAlmostEqual(emb.get_e_corr(functional=functional), emb_callback.get_e_corr(functional=functional), delta=1e-6)

            for a, b in [(False, False), (True, False), (True, True)]:
                self.assertAlmostEqual(emb.get_dmet_energy(part_cumulant=a, approx_cumulant=b), emb_callback.get_dmet_energy(part_cumulant=a, approx_cumulant=b), delta=1e-6)


    def test_rccsd_water(self):
        self._test(("water_ccpvdz", "rhf", "CCSD"))
    
    def test_rfci_h6(self):
        self._test(("h6_sto6g", "rhf", "FCI"))

    # def test_rcisd_lih(self):
    #     self._test(("lih_ccpvdz", "rhf", "CISD"))

    def test_uccsd_water(self):
        self._test(("water_ccpvdz", "uhf", "CCSD"))
    
    def test_ufci_h6(self):
        self._test(("h6_sto6g", "uhf", "FCI"))
    
    # def test_ucisd_lih(self):
    #     self._test(("lih_ccpvdz", "uhf", "CISD"))



if __name__ == "__main__":
    print("Running %s" % __file__)
    unittest.main()

