import unittest

from dyson.expressions import FCI, CCSD
from dyson.solvers.chempot import AufbauPrinciple
from vayesta import egf
from vayesta.core.util import AbstractMethodError, cache
from vayesta.tests.common import TestCase
from vayesta.tests import testsystems
from vayesta.tests.egf.test_hubbard import Test_Full_Bath_FCI as Test_Full_Bath_Hubbard

class Test_Full_Bath_FCI(Test_Full_Bath_Hubbard):
    NFRAG = 2
    system = testsystems.h6_sto6g

    @classmethod
    @cache
    def emb(cls, mf):
        opts = dict(solver=cls.solver, proj=1, aux_shift=True, use_sym=True)
        bath_opts = dict(bathtype='full', dmet_threshold=1e-12)
        solver_opts = dict(conv_tol=1e-15, n_moments=(cls.NMOM_MAX_GF, cls.NMOM_MAX_GF))
        emb = egf.EGF(mf, **opts, bath_options=bath_opts, solver_options=solver_opts)
        with emb.iaopao_fragmentation() as f:
            f.add_all_atomic_fragments()
        emb.kernel()
        return emb
    
class Test_Full_Bath_FCI_Sym(Test_Full_Bath_FCI):

    @classmethod
    @cache
    def emb(cls, mf):
        opts = dict(solver=cls.solver, proj=1, aux_shift=True, use_sym=True)
        bath_opts = dict(bathtype='full', dmet_threshold=1e-12)
        solver_opts =dict(conv_tol=1e-15, n_moments=(cls.NMOM_MAX_GF, cls.NMOM_MAX_GF))
        emb = egf.EGF(mf, **opts, bath_options=bath_opts, solver_options=solver_opts)
        with emb.iaopao_fragmentation() as f:
            with f.rotational_symmetry(order=int(mf.mol.natm//cls.NFRAG), axis='z') as rot: 
                f.add_atomic_fragment(range(cls.NFRAG))
        emb.kernel()
        return emb

class Test_Full_Bath_CCSD(Test_Full_Bath_FCI):
    solver = 'CCSD'
    EXPR = CCSD


class Test_Full_Bath_CCSD_Sym(Test_Full_Bath_FCI_Sym):
    solver = 'CCSD'
    EXPR = CCSD

if __name__ == "__main__":
    # print("Running %s" % __file__)
    # unittest.main()

    t = Test_Full_Bath_FCI()
    t.setUpClass()
    t.test_gf_moments()