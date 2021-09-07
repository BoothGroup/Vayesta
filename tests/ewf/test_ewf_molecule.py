import unittest
import numpy as np
from pyscf import gto, scf, lib
from vayesta import ewf


def make_test(atom, basis, kwargs, fragmentation, known_values, name=None):

    class MoleculeEWFTests(unittest.TestCase):

        shortDescription = lambda self: name

        @classmethod
        def setUpClass(cls):
            cls.mol = gto.Mole()
            cls.mol.atom = atom
            cls.mol.basis = basis
            cls.mol.verbose = 0
            cls.mol.max_memory = 1e9
            cls.mol.build()
            cls.mf = scf.RHF(cls.mol)
            cls.mf.conv_tol = 1e-12
            cls.mf.kernel()
            cls.ewf = ewf.EWF(cls.mf, solver_options={'conv_tol': 1e-10}, **kwargs)
            fragmentation(cls.ewf)
            cls.ewf.kernel()

            #TODO keeping this here for now in case we want more:
            #print(cls.ewf.e_tot)
            #print(cls.ewf.get_dmet_energy())
            #print(lib.fp(cls.ewf.get_t1()))
            #print(lib.fp(cls.ewf.get_t1(get_lambda=True)))
            #print(lib.fp(cls.ewf.get_t12()[1]))
            #print(lib.fp(cls.ewf.get_t12(get_lambda=True)[1]))
            #print(np.trace(cls.ewf.make_rdm1_demo()))
            #print(np.trace(cls.ewf.make_rdm1_demo(ao_basis=True)))
            #print(np.trace(cls.ewf.make_rdm1_ccsd()))
            #print(np.trace(cls.ewf.make_rdm1_ccsd(ao_basis=True)))
            #print(np.einsum('iiii->', cls.ewf.make_rdm2_demo()))
            #print(np.einsum('iiii->', cls.ewf.make_rdm2_demo(ao_basis=True)))
            #print(np.einsum('iiii->', cls.ewf.make_rdm2_ccsd()))
            #print(np.einsum('iiii->', cls.ewf.make_rdm2_ccsd(ao_basis=True)))

        @classmethod
        def tearDownClass(cls):
            del cls.mol, cls.mf, cls.ewf

        def test_energy(self):
            self.assertAlmostEqual(self.ewf.e_tot, known_values['e_tot'], 8)

        def test_dmet_energy(self):
            if 'e_dmet' in known_values:
                self.assertAlmostEqual(self.ewf.get_dmet_energy(), known_values['e_dmet'], 8)

        def test_t1(self):
            if 't1' in known_values:
                self.assertAlmostEqual(lib.fp(self.ewf.get_t1()), known_values['t1'], 8)
            if 'l1' in known_values:
                self.assertAlmostEqual(lib.fp(self.ewf.get_t1(get_lambda=True)), known_values['l1'], 8)

        def test_t2(self):
            if 't2' in known_values:
                self.assertAlmostEqual(lib.fp(self.ewf.get_t12()[1]), known_values['t2'], 8)
            if 'l2' in known_values:
                self.assertAlmostEqual(lib.fp(self.ewf.get_t12(get_lambda=True)[1]), known_values['l2'], 8)

        def test_rdm1(self):
            if 'rdm1_demo' in known_values:
                self.assertAlmostEqual(np.trace(self.ewf.make_rdm1_demo()), known_values['rdm1_demo'], 8)
            if 'rdm1_demo_ao' in known_values:
                self.assertAlmostEqual(np.trace(self.ewf.make_rdm1_demo(ao_basis=True)), known_values['rdm1_demo_ao'], 8)
            if 'rdm1_ccsd' in known_values:
                self.assertAlmostEqual(np.trace(self.ewf.make_rdm1_ccsd()), known_values['rdm1_ccsd'], 8)
            if 'rdm1_ccsd_ao' in known_values:
                self.assertAlmostEqual(np.trace(self.ewf.make_rdm1_ccsd(ao_basis=True)), known_values['rdm1_ccsd_ao'], 8)

        def test_rdm2(self):
            trace = lambda m: np.einsum('iiii->', m)
            if 'rdm2_demo' in known_values:
                self.assertAlmostEqual(trace(self.ewf.make_rdm2_demo()), known_values['rdm2_demo'], 8)
            if 'rdm2_demo_ao' in known_values:
                self.assertAlmostEqual(trace(self.ewf.make_rdm2_demo(ao_basis=True)), known_values['rdm2_demo_ao'], 8)
            if 'rdm2_ccsd' in known_values:
                self.assertAlmostEqual(trace(self.ewf.make_rdm2_ccsd()), known_values['rdm2_ccsd'], 8)
            if 'rdm2_ccsd_ao' in known_values:
                self.assertAlmostEqual(trace(self.ewf.make_rdm2_ccsd(ao_basis=True)), known_values['rdm2_ccsd_ao'], 8)


    return MoleculeEWFTests


LiH_ccpvdz_iao_atoms_Test = make_test(
        'Li 0 0 0; H 0 0 1.4', 'cc-pvdz',
        {'fragment_type': 'iao', 'bath_type': 'all', 'make_rdm1': True, 'make_rdm2': True},
        lambda ewf: ewf.make_all_atom_fragments(),
        {
            'e_tot':  -8.008269603007381,
            'e_dmet': -7.97664320557861,
            't1': -0.0037332701674217044,
            'l1': -0.003431346419365344,
            't2':  0.010158010207677414,
            'l2':  0.010579418095111528,
            'rdm1_demo':    3.967390566642664,
            'rdm1_demo_ao': 2.8599690310182355,
            'rdm1_ccsd':    4.0,
            'rdm1_ccsd_ao': 2.9402121097868132,
            'rdm2_demo':    3.953428358419938,
            'rdm2_demo_ao': 1.9147846236993586,
            'rdm2_ccsd':    3.968948851940337,
            'rdm2_ccsd_ao': 2.0677163999542425,
        },
        name='LiH_ccpvdz_iao_atoms_Test',
)

LiH_ccpvdz_lowdin_aos_Test = make_test(
        'Li 0 0 0; H 0 0 1.4', 'cc-pvdz',
        {'fragment_type': 'lowdin-ao', 'bno_threshold': 1e-5},
        lambda ewf: (ewf.make_ao_fragment([0, 1]), ewf.make_ao_fragment([2, 3, 4])),
        {'e_tot': -7.98424889149862},
        name='LiH_ccpvdz_lowdin_aos_Test',
)

LiH_ccpvdz_lowdin_atoms_Test = make_test(
        'Li 0 0 0; H 0 0 1.4', 'cc-pvdz',
        {'fragment_type': 'lowdin-ao', 'bath_type': 'none'},
        lambda ewf: ewf.make_all_atom_fragments(),
        {'e_tot': -7.99502192669842},
        name='LiH_ccpvdz_lowdin_atoms_Test',
)

N2_augccpvdz_stretched_FCI_Test = make_test(
        'N 0 0 0; N 0 0 2', 'aug-cc-pvdz',
        {'solver': 'FCI', 'bno_threshold': 100},
        lambda ewf: ewf.make_atom_fragment(0, sym_factor=2),
        {'e_tot': -108.7770182190321},
        name='N2_augccpvdz_stretched_FCI_Test',
)

N2_ccpvdz_tccsd_Test = make_test(
        'N1 0 0 0; N2 0 0 1.1', 'cc-pvdz',
        {'solver': 'TCCSD', 'bno_threshold': 1e-4},
        lambda ewf: ewf.make_atom_fragment('N1', sym_factor=2),
        {'e_tot': -109.27077981413623},
        name='N2_ccpvdz_tccsd_Test',
)

N2_ccpvdz_tccsd_cas_iaos_Test = make_test(
        'N1 0 0 0; N2 0 0 1.1', 'cc-pvdz',
        {'solver': 'TCCSD', 'bno_threshold': 1e-4, 'fragment_type': 'iao'},
        lambda ewf: ewf.make_atom_fragment('N1', sym_factor=2).set_cas(['0 N1 2p']),
        {'e_tot': -109.27252621439553},
        name='N2_ccpvdz_tccsd_cas_iaos_Test',
)

#FIXME: bugged
#N2_ccpvdz_tccsd_cas_lowdin_Test = make_test(
#        'N1 0 0 0; N2 0 0 1.1', 'cc-pvdz',
#        {'solver': 'TCCSD', 'bno_threshold': 1e-4, 'fragment_type': 'lowdin-ao'},
#        lambda ewf: ewf.make_atom_fragment('N1', sym_factor=2).set_cas(['0 N1 2p']),
#        {'e_tot': -109.27252621439553},
#        name='N2_ccpvdz_tccsd_cas_lowdin_Test',
#)

N2_ccpvdz_sc_Test = make_test(
        'N1 0 0 0; N2 0 0 1.1', 'cc-pvdz',
        {'bno_threshold': 1e-4, 'sc_mode': 1, 'sc_energy_tol': 1e-9},
        lambda ewf: ewf.make_atom_fragment(0, sym_factor=2),
        {'e_tot': -109.26013012932687},
        name='N2_ccpvdz_sc_Test',
)

class MiscMoleculeEWFTests(unittest.TestCase):

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
                solver_options={'conv_tol': 1e-10},
                bno_threshold=1e-6,
                eom_ccsd=['IP', 'EA', 'EE-S', 'EE-T', 'EE-SF'],
                eom_ccsd_nroots=5,
        )
        frag = emb.make_atom_fragment(0)
        frag.kernel()  #FIXME using this to build the cluster orbs, repeats solver calculation
        from vayesta.solver.solver_cc import CCSDSolver  #TODO move this to solver tests?
        from pyscf import cc
        nocc = frag.c_cluster_occ.shape[1]
        nvir = frag.c_cluster_vir.shape[1]
        nocc_frozen = np.sum(self.mf.mo_occ > 0) - nocc
        nvir_frozen = np.sum(self.mf.mo_occ == 0) - nvir
        solver = CCSDSolver(frag, self.mf.mo_coeff, self.mf.mo_occ, nocc_frozen, nvir_frozen)
        res = solver.kernel()

        self.assertAlmostEqual(res.ip_energy[0], 0.5810398549938971, 8)
        self.assertAlmostEqual(res.ea_energy[0], 0.2527482232750386, 8)
        self.assertAlmostEqual(res.ee_s_energy[0], 0.4302596246755637, 8)
        self.assertAlmostEqual(res.ee_t_energy[0], 0.3755142786878773, 8)
        self.assertAlmostEqual(res.ee_sf_energy[0], 0.3755142904509986, 8)

        self.assertAlmostEqual(np.linalg.norm(res.ip_coeff[0][:nocc]), 0.9805776450121361, 8)
        self.assertAlmostEqual(np.linalg.norm(res.ea_coeff[0][:nvir]), 0.9978012299430233, 8)
        self.assertAlmostEqual(np.linalg.norm(res.ee_s_coeff[0][:nocc*nvir]), 0.6878077752215053, 8)
        self.assertAlmostEqual(np.linalg.norm(res.ee_t_coeff[0][:nocc*nvir]), 0.6932475285290554, 8)
        self.assertAlmostEqual(np.linalg.norm(res.ee_sf_coeff[0][:nocc*nvir]), 0.6932475656707386, 8)


if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
