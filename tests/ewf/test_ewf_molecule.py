import unittest
import numpy as np
from pyscf import gto, scf, lib
from vayesta import ewf

#TODO tighten thresholds once solver interface is finished

# Use default conv_tol
EWF_CONV_TOL = None

class MoleculeEWFTest:
    ''' Abstract base class for molecular EWF tests.
    '''

    @classmethod
    def setUpClass(cls):
        cls.mol = None
        cls.mf = None
        cls.ewf = None
        cls.known_values = None

    @classmethod
    def tearDownClass(cls):
        del cls.mol, cls.mf, cls.ewf, cls.known_values

    def test_energy(self):
        self.assertAlmostEqual(self.ewf.e_tot, self.known_values['e_tot'], 6)


class MoleculeEWFTest_LiH_ccpvdz_IAO_atoms(unittest.TestCase, MoleculeEWFTest):
    @classmethod
    def setUpClass(cls):
        cls.mol = gto.Mole()
        cls.mol.atom = 'Li 0 0 0; H 0 0 1.4'
        cls.mol.basis = 'cc-pvdz'
        cls.mol.verbose = 0
        cls.mol.max_memory = 1e9
        cls.mol.build()

        cls.mf = scf.RHF(cls.mol)
        cls.mf.conv_tol = 1e-12
        cls.mf.kernel()
        assert cls.mf.converged

        cls.ewf = ewf.EWF(
                cls.mf,
                fragment_type='iao',
                bath_type='all',
                make_rdm1=True,
                make_rdm2=True,
                solver_options={
                    'conv_tol': EWF_CONV_TOL,
                },
        )
        cls.ewf.iao_fragmentation()
        cls.ewf.make_all_atom_fragments()
        cls.ewf.kernel()

        cls.known_values = {
            'e_tot':  -8.008269603007381,
            'e_dmet': -7.97664320557861,
            't1': 0.020041132110973907,
            'l1': 0.019095485495677657,
            't2': 0.15951192869005756,
            'l2': 0.15539614532743415,
            'rdm1_demo':    3.967390566642664,
            'rdm1_demo_ao': 2.8599690310182355,
            'rdm1_ccsd':    4.0,
            'rdm1_ccsd_ao': 2.9402121097868132,
            'rdm2_demo':    3.953428358419938,
            'rdm2_demo_ao': 1.9147846236993586,
            'rdm2_ccsd':    3.968948851940337,
            'rdm2_ccsd_ao': 2.0677163999542425,
        }


    # Extra tests for this system:

    def test_dmet_energy(self):
        self.assertAlmostEqual(self.ewf.get_dmet_energy(), self.known_values['e_dmet'], 6)

    def test_t1(self):
        self.assertAlmostEqual(np.linalg.norm(self.ewf.get_t1()), self.known_values['t1'], 8)
        self.assertAlmostEqual(np.linalg.norm(self.ewf.get_t1(get_lambda=True)), self.known_values['l1'], 8)

    def test_t2(self):
        self.assertAlmostEqual(np.linalg.norm(self.ewf.get_t12()[1]), self.known_values['t2'], 8)
        self.assertAlmostEqual(np.linalg.norm(self.ewf.get_t12(get_lambda=True)[1]), self.known_values['l2'], 8)

    def test_rdm1(self):
        self.assertAlmostEqual(np.trace(self.ewf.make_rdm1_demo()), self.known_values['rdm1_demo'], 8)
        self.assertAlmostEqual(np.trace(self.ewf.make_rdm1_demo(ao_basis=True)), self.known_values['rdm1_demo_ao'], 8)
        self.assertAlmostEqual(np.trace(self.ewf.make_rdm1_ccsd()), self.known_values['rdm1_ccsd'], 8)
        self.assertAlmostEqual(np.trace(self.ewf.make_rdm1_ccsd(ao_basis=True)), self.known_values['rdm1_ccsd_ao'], 8)

    def test_rdm2(self):
        trace = lambda m: np.einsum('iiii->', m)
        self.assertAlmostEqual(trace(self.ewf.make_rdm2_demo()), self.known_values['rdm2_demo'], 8)
        self.assertAlmostEqual(trace(self.ewf.make_rdm2_demo(ao_basis=True)), self.known_values['rdm2_demo_ao'], 8)
        self.assertAlmostEqual(trace(self.ewf.make_rdm2_ccsd()), self.known_values['rdm2_ccsd'], 8)
        self.assertAlmostEqual(trace(self.ewf.make_rdm2_ccsd(ao_basis=True)), self.known_values['rdm2_ccsd_ao'], 8)


class MoleculeEWFTest_LiH_ccpvdz_Lowdin_AOs(unittest.TestCase, MoleculeEWFTest):
    @classmethod
    def setUpClass(cls):
        cls.mol = gto.Mole()
        cls.mol.atom = 'Li 0 0 0; H 0 0 1.4'
        cls.mol.basis = 'cc-pvdz'
        cls.mol.verbose = 0
        cls.mol.max_memory = 1e9
        cls.mol.build()

        cls.mf = scf.RHF(cls.mol)
        cls.mf.conv_tol = 1e-12
        cls.mf.kernel()

        cls.ewf = ewf.EWF(
                cls.mf,
                fragment_type='lowdin-ao',
                bno_threshold=1e-5,
                solver_options={
                    'conv_tol': EWF_CONV_TOL,
                },
        )
        cls.ewf.sao_fragmentation()
        cls.ewf.make_ao_fragment([0, 1])
        cls.ewf.make_ao_fragment([2, 3, 4])
        cls.ewf.kernel()

        cls.known_values = {'e_tot': -7.98424889149862}


class MoleculeEWFTest_LiH_ccpvdz_Lowdin_atoms(unittest.TestCase, MoleculeEWFTest):
    @classmethod
    def setUpClass(cls):
        cls.mol = gto.Mole()
        cls.mol.atom = 'Li 0 0 0; H 0 0 1.4'
        cls.mol.basis = 'cc-pvdz'
        cls.mol.verbose = 0
        cls.mol.max_memory = 1e9
        cls.mol.build()

        cls.mf = scf.RHF(cls.mol)
        cls.mf.conv_tol = 1e-12
        cls.mf.kernel()

        cls.ewf = ewf.EWF(
                cls.mf,
                fragment_type='lowdin-ao',
                bath_type=None,
                solver_options={
                    'conv_tol': EWF_CONV_TOL,
                },
        )
        cls.ewf.sao_fragmentation()
        cls.ewf.make_all_atom_fragments()
        cls.ewf.kernel()

        cls.known_values = {'e_tot': -7.99502192669842}


class MoleculeEWFTest_N2_augccpvdz_stretched_FCI(unittest.TestCase, MoleculeEWFTest):
    @classmethod
    def setUpClass(cls):
        cls.mol = gto.Mole()
        cls.mol.atom = 'N 0 0 0; N 0 0 2.0'
        cls.mol.basis = 'aug-cc-pvdz'
        cls.mol.verbose = 0
        cls.mol.max_memory = 1e9
        cls.mol.build()

        cls.mf = scf.RHF(cls.mol)
        cls.mf.conv_tol = 1e-12
        cls.mf.kernel()

        cls.ewf = ewf.EWF(
                cls.mf,
                solver='FCI',
                bno_threshold=100,
                solver_options={
                    'conv_tol' : 1e-14,
                    'fix_spin' : 0.0,
                    }
        )
        cls.ewf.iao_fragmentation()
        cls.ewf.make_atom_fragment(0, sym_factor=2)
        cls.ewf.kernel()

        cls.known_values = {'e_tot': -108.7770291262215}


class MoleculeEWFTest_N2_ccpvdz_TCCSD(unittest.TestCase, MoleculeEWFTest):
    @classmethod
    def setUpClass(cls):
        cls.mol = gto.Mole()
        cls.mol.atom = 'N1 0 0 0; N2 0 0 1.1'
        cls.mol.basis = 'cc-pvdz'
        cls.mol.verbose = 0
        cls.mol.max_memory = 1e9
        cls.mol.build()

        cls.mf = scf.RHF(cls.mol)
        cls.mf.conv_tol = 1e-12
        cls.mf.kernel()

        cls.ewf = ewf.EWF(
                cls.mf,
                solver='TCCSD',
                bno_threshold=1e-4,
                solver_options={
                    'conv_tol': EWF_CONV_TOL,
                },
        )
        cls.ewf.iao_fragmentation()
        cls.ewf.make_atom_fragment('N1', sym_factor=2)
        cls.ewf.kernel()

        cls.known_values = {'e_tot': -109.27077981413623}


class MoleculeEWFTest_N2_ccpvdz_TCCSD_CAS(unittest.TestCase, MoleculeEWFTest):
    @classmethod
    def setUpClass(cls):
        cls.mol = gto.Mole()
        cls.mol.atom = 'N1 0 0 0; N2 0 0 1.1'
        cls.mol.basis = 'cc-pvdz'
        cls.mol.verbose = 0
        cls.mol.max_memory = 1e9
        cls.mol.build()

        cls.mf = scf.RHF(cls.mol)
        cls.mf.conv_tol = 1e-12
        cls.mf.kernel()

        cls.ewf = ewf.EWF(
                cls.mf,
                solver='TCCSD',
                bno_threshold=1e-4,
                fragment_type='iao',
                solver_options={
                    'conv_tol': EWF_CONV_TOL,
                },
        )
        cls.ewf.iao_fragmentation()
        cls.ewf.make_atom_fragment('N1', sym_factor=2)
        cls.ewf.fragments[0].set_cas(['0 N1 2p'])
        cls.ewf.kernel()

        cls.known_values = {'e_tot': -109.27252621439553}


class MoleculeEWFTest_N2_ccpvdz_sc(unittest.TestCase, MoleculeEWFTest):
    @classmethod
    def setUpClass(cls):
        cls.mol = gto.Mole()
        cls.mol.atom = 'N1 0 0 0; N2 0 0 1.1'
        cls.mol.basis = 'cc-pvdz'
        cls.mol.verbose = 0
        cls.mol.max_memory = 1e9
        cls.mol.build()

        cls.mf = scf.RHF(cls.mol)
        cls.mf.conv_tol = 1e-12
        cls.mf.kernel()

        cls.ewf = ewf.EWF(
                cls.mf,
                bno_threshold=1e-4,
                sc_mode=1,
                sc_energy_tol=1e-9,
                solver_options={
                    'conv_tol': EWF_CONV_TOL,
                },
        )
        cls.ewf.iao_fragmentation()
        cls.ewf.make_atom_fragment(0, sym_factor=2)
        cls.ewf.kernel()

        cls.known_values = {'e_tot': -109.26013012932687}


class MiscMoleculeEWFTests(unittest.TestCase):
    ''' Tests for miscellaneous features that don't fit MoleculeEWFTests.
    '''

    @classmethod
    def setUpClass(cls):
        cls.mol = gto.Mole()
        cls.mol.atom = 'N 0 0 0; N 0 0 1'
        cls.mol.basis = 'cc-pvdz'
        cls.mol.verbose = 0
        cls.mol.max_memory = 1e9
        cls.mol.build()
        cls.mf = scf.RHF(cls.mol)
        cls.mf.conv_tol = 1e-12
        cls.mf.kernel()

    @classmethod
    def tearDownClss(cls):
        del cls.mol, cls.mf

    def test_reset(self):
        emb = ewf.EWF(self.mf, solver_options={'conv_tol': 1e-10})
        emb.iao_fragmentation()
        frag = emb.make_atom_fragment(0)
        frag.kernel()
        for key in ['c_cluster_occ', 'c_cluster_vir', 'c_no_occ', 'c_no_vir', 'n_no_occ', 'n_no_vir']:
            self.assertTrue(getattr(frag, key) is not None)
        frag.reset()
        for key in ['c_cluster_occ', 'c_cluster_vir', 'c_no_occ', 'c_no_vir', 'n_no_occ', 'n_no_vir']:
            self.assertTrue(getattr(frag, key) is None)

    def test_eom(self):
        emb = ewf.EWF(
                self.mf,
                solver_options={'conv_tol': EWF_CONV_TOL},
                bno_threshold=1e-6,
                eom_ccsd=['IP', 'EA', 'EE-S', 'EE-T', 'EE-SF'],
                eom_ccsd_nroots=5,
        )
        emb.iao_fragmentation()
        frag = emb.make_atom_fragment(0)
        frag.kernel()  #FIXME using this to build the cluster orbs, repeats solver calculation
        from vayesta.solver.solver_cc import CCSDSolver  #TODO move this to solver tests?
        from pyscf import cc
        nocc = frag.c_cluster_occ.shape[1]
        nvir = frag.c_cluster_vir.shape[1]
        nocc_frozen = np.sum(self.mf.mo_occ > 0) - nocc
        nvir_frozen = np.sum(self.mf.mo_occ == 0) - nvir
        solver = CCSDSolver(frag, self.mf.mo_coeff, self.mf.mo_occ, nocc_frozen, nvir_frozen,
                eom_ccsd=['IP', 'EA', 'EE-S', 'EE-T', 'EE-SF'], eom_ccsd_nroots=5)
        res = solver.kernel()

        self.assertAlmostEqual(res.ip_energy[0], 0.5810398549938971, 6)
        self.assertAlmostEqual(res.ea_energy[0], 0.2527482232750386, 6)
        self.assertAlmostEqual(res.ee_s_energy[0], 0.4302596246755637, 6)
        self.assertAlmostEqual(res.ee_t_energy[0], 0.3755142786878773, 6)
        self.assertAlmostEqual(res.ee_sf_energy[0], 0.3755142904509986, 6)

        self.assertAlmostEqual(np.linalg.norm(res.ip_coeff[0][:nocc]), 0.9805776450121361, 6)
        self.assertAlmostEqual(np.linalg.norm(res.ea_coeff[0][:nvir]), 0.9978012299430233, 6)
        self.assertAlmostEqual(np.linalg.norm(res.ee_s_coeff[0][:nocc*nvir]), 0.6878077752215053, 6)
        self.assertAlmostEqual(np.linalg.norm(res.ee_t_coeff[0][:nocc*nvir]), 0.6932475285290554, 6)
        self.assertAlmostEqual(np.linalg.norm(res.ee_sf_coeff[0][:nocc*nvir]), 0.6932475656707386, 6)


if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
