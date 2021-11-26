import unittest
import numpy as np

from vayesta import ewf
from vayesta.tests.cache import moles
from vayesta.solver.ccsd import CCSD_Solver


class MoleculeEWFTests(unittest.TestCase):
    PLACES_ENERGY = 7
    PLACES_T = 6
    PLACES_DM = 6
    CONV_TOL = 1e-9
    CONV_TOL_NORMT = 1e-8

    def _test_energy(self, emb, known_values):
        """Tests the EWF energy.
        """

        self.assertAlmostEqual(emb.e_tot, known_values['e_tot'], self.PLACES_ENERGY)

    def _test_dmet_energy(self, emb, known_values):
        """Tests the DMET energy.
        """

        self.assertAlmostEqual(emb.get_dmet_energy(), known_values['e_dmet'], self.PLACES_ENERGY)

    def _test_t1(self, emb, known_values):
        """Tests the T1 and L1 amplitudes.
        """

        t1 = emb.get_t1()
        l1 = emb.get_t1(get_lambda=True)

        self.assertAlmostEqual(np.linalg.norm(t1), known_values['t1'], self.PLACES_T)
        self.assertAlmostEqual(np.linalg.norm(l1), known_values['l1'], self.PLACES_T)

    def _test_t2(self, emb, known_values):
        """Tests the T2 and L2 amplitudes.
        """

        t2 = emb.get_t12()[1]
        l2 = emb.get_t12(get_lambda=True)[1]

        self.assertAlmostEqual(np.linalg.norm(t2), known_values['t2'], self.PLACES_T)
        self.assertAlmostEqual(np.linalg.norm(l2), known_values['l2'], self.PLACES_T)

    def _test_rdm1(self, emb, known_values):
        """Tests the traces of the first-order density matrices.
        """

        dm = emb.make_rdm1_demo()
        self.assertAlmostEqual(np.trace(dm), known_values['rdm1_demo'], self.PLACES_DM)

        dm = emb.make_rdm1_demo(ao_basis=True)
        self.assertAlmostEqual(np.trace(dm), known_values['rdm1_demo_ao'], self.PLACES_DM)

        dm = emb.make_rdm1_ccsd()
        self.assertAlmostEqual(np.trace(dm), known_values['rdm1_ccsd'], self.PLACES_DM)

        dm = emb.make_rdm1_ccsd(ao_basis=True)
        self.assertAlmostEqual(np.trace(dm), known_values['rdm1_ccsd_ao'], self.PLACES_DM)

    def _test_rdm2(self, emb, known_values):
        """Tests the traces of the second-order density matrices.
        """

        trace = lambda m: np.einsum('iiii->', m)

        dm = emb.make_rdm2_demo()
        self.assertAlmostEqual(trace(dm), known_values['rdm2_demo'], self.PLACES_DM)

        dm = emb.make_rdm2_demo(ao_basis=True)
        self.assertAlmostEqual(trace(dm), known_values['rdm2_demo_ao'], self.PLACES_DM)

        dm = emb.make_rdm2_ccsd()
        self.assertAlmostEqual(trace(dm), known_values['rdm2_ccsd'], self.PLACES_DM)

        dm = emb.make_rdm2_ccsd(ao_basis=True)
        self.assertAlmostEqual(trace(dm), known_values['rdm2_ccsd_ao'], self.PLACES_DM)

    def test_lih_ccpvdz_iao_atoms(self):
        """Tests EWF for LiH cc-pvdz with IAO atomic fragmentation.
        """

        emb = ewf.EWF(
                moles['lih_ccpvdz']['rhf'],
                bath_type='all',
                make_rdm1=True,
                make_rdm2=True,
                solver_options={
                    'conv_tol': self.CONV_TOL,
                    'conv_tol_normt': self.CONV_TOL_NORMT,
                },
        )
        emb.iao_fragmentation()
        emb.make_all_atom_fragments()
        emb.kernel()

        known_values = {
            'e_tot':  -8.008269603007381,
            'e_dmet': -7.976643405222605,
            't1': 0.02004058442926091,
            'l1': 0.01909464775558478,
            't2': 0.15951192869005756,
            'l2': 0.15539564391258082,
            'rdm1_demo':    3.9673905666426640,
            'rdm1_demo_ao': 2.8599690310182355,
            'rdm1_ccsd':    4.0,
            'rdm1_ccsd_ao': 2.9402121097868132,
            'rdm2_demo':    3.9534283584199380,
            'rdm2_demo_ao': 1.9147846236993586,
            'rdm2_ccsd':    3.9689488519403370,
            'rdm2_ccsd_ao': 2.0677172173309413,
        }

        self._test_energy(emb, known_values)
        self._test_dmet_energy(emb, known_values)
        self._test_t1(emb, known_values)
        self._test_t2(emb, known_values)
        self._test_rdm1(emb, known_values)
        self._test_rdm2(emb, known_values)

    def test_lih_ccpvdz_sao_orbitals(self):
        """Tests EWF for LiH cc-pvdz with SAO orbital fragmentation.
        """

        emb = ewf.EWF(
                moles['lih_ccpvdz']['rhf'],
                bno_threshold=1e-5,
                solver_options={
                    'conv_tol': self.CONV_TOL,
                    'conv_tol_normt': self.CONV_TOL_NORMT,
                },
        )
        emb.sao_fragmentation()
        emb.make_ao_fragment([0, 1])
        emb.make_ao_fragment([2, 3, 4])
        emb.kernel()

        known_values = {'e_tot': -7.98424889149862}

        self._test_energy(emb, known_values)

    def test_lih_ccpvdz_sao_atoms(self):
        """Tests EWF for LiH cc-pvdz with SAO atomic fragmentation.
        """

        emb = ewf.EWF(
                moles['lih_ccpvdz']['rhf'],
                bath_type=None,
                solver_options={
                    'conv_tol': self.CONV_TOL,
                    'conv_tol_normt': self.CONV_TOL_NORMT,
                },
        )
        emb.sao_fragmentation()
        emb.make_all_atom_fragments()
        emb.kernel()

        known_values = {'e_tot': -7.99502192669842}

        self._test_energy(emb, known_values)

    def test_h2o_ccpvdz_FCI(self):
        """Tests EWF for H2O cc-pvdz with FCI solver.
        """

        emb = ewf.EWF(
                moles['h2o_ccpvdz']['rhf'],
                bath_type=None,
                solver='FCI',
                bno_threshold=100,
                solver_options={
                    'conv_tol': self.CONV_TOL,
                    'fix_spin': 0.0,
                }
        )
        emb.iao_fragmentation()
        emb.make_atom_fragment(0)
        emb.make_atom_fragment(1, sym_factor=2)
        emb.kernel()

        known_values = {'e_tot': -76.06365118513072}

        self._test_energy(emb, known_values)

    def test_h2o_ccpvdz_TCCSD(self):
        """Tests EWF for H2O cc-pvdz with FCI solver.
        """

        emb = ewf.EWF(
                moles['h2o_ccpvdz']['rhf'],
                solver='TCCSD',
                bno_threshold=1e-4,
                solver_options={
                    'conv_tol': self.CONV_TOL,
                    'conv_tol_normt': self.CONV_TOL_NORMT,
                },
        )
        emb.iao_fragmentation()
        emb.make_atom_fragment(0)
        emb.make_atom_fragment(1, sym_factor=2)
        emb.kernel()

        known_values = {'e_tot': -76.23613568648057}

        self._test_energy(emb, known_values)

    def test_h2o_ccpvdz_TCCSD_CAS(self):
        """Tests EWF for H2O cc-pvdz with TCCSD solver and CAS picker.
        """

        emb = ewf.EWF(
                moles['h2o_ccpvdz']['rhf'],
                solver='TCCSD',
                bno_threshold=1e-4,
                solver_options={
                    'conv_tol': self.CONV_TOL,
                    'conv_tol_normt': self.CONV_TOL_NORMT,
                },
        )
        emb.iao_fragmentation()
        emb.make_atom_fragment(0)
        emb.make_atom_fragment(1, sym_factor=2)
        emb.fragments[0].set_cas(['0 O 2p'])
        emb.kernel()

        known_values = {'e_tot': -76.23559827815198}

        self._test_energy(emb, known_values)

    def test_h2o_ccpvdz_sc(self):
        """Tests EWF for H2O cc-pvdz with self-consistency.
        """

        emb = ewf.EWF(
                moles['h2o_ccpvdz']['rhf'],
                bno_threshold=1e-4,
                sc_mode=1,
                sc_energy_tol=1e-9,
                solver_options={
                    'conv_tol': self.CONV_TOL,
                    'conv_tol_normt': self.CONV_TOL_NORMT,
                },
        )
        emb.iao_fragmentation()
        emb.make_atom_fragment(0)
        emb.make_atom_fragment(1, sym_factor=2)
        emb.kernel()

        known_values = {'e_tot': -76.23147227604929}

        self._test_energy(emb, known_values)

    def test_h2_ccpvdz_cisd(self):
        """Compares CCSD and CISD solvers for a two-electron system.
        """

        ecisd = ewf.EWF(
                moles['h2_ccpvdz']['rhf'],
                bath_type=None,
                solver='CISD',
                solver_options={
                    'conv_tol': self.CONV_TOL,
                    'conv_tol_normt': self.CONV_TOL_NORMT,
                },
        )
        ecisd.iao_fragmentation()
        ecisd.add_all_atomic_fragments()
        ecisd.kernel()

        eccsd = ewf.EWF(
                moles['h2_ccpvdz']['rhf'],
                bath_type=None,
                solver='CCSD',
                solver_options={
                    'conv_tol': self.CONV_TOL,
                    'conv_tol_normt': self.CONV_TOL_NORMT,
                },
        )
        eccsd.iao_fragmentation()
        eccsd.add_all_atomic_fragments()
        eccsd.kernel()

        self.assertAlmostEqual(ecisd.e_corr, eccsd.e_corr)
        self.assertAlmostEqual(ecisd.e_tot, eccsd.e_tot)

    def test_eom(self):
        """Tests EWF EOM-CCSD support.
        """

        emb = ewf.EWF(
                moles['n2_631g']['rhf'],
                bno_threshold=1e-6,
                eom_ccsd=['IP', 'EA', 'EE-S', 'EE-T', 'EE-SF'],
                eom_ccsd_nroots=5,
                solver_options={
                    'conv_tol': self.CONV_TOL,
                    'conv_tol_normt': self.CONV_TOL_NORMT,
                },
        )
        emb.iao_fragmentation()
        frag = emb.make_atom_fragment(0)
        frag.kernel()

        nocc = frag.c_cluster_occ.shape[1]
        nvir = frag.c_cluster_vir.shape[1]
        nocc_frozen = np.sum(moles['n2_631g']['rhf'].mo_occ > 0) - nocc
        nvir_frozen = np.sum(moles['n2_631g']['rhf'].mo_occ == 0) - nvir
        solver = CCSD_Solver(
                frag,
                moles['n2_631g']['rhf'].mo_coeff,
                moles['n2_631g']['rhf'].mo_occ,
                nocc_frozen,
                nvir_frozen,
                eom_ccsd=['IP', 'EA', 'EE-S', 'EE-T', 'EE-SF'],
                eom_ccsd_nroots=5,
        )
        res = solver.kernel()

        self.assertAlmostEqual(res.ip_energy[0],    0.57188303464319880, self.PLACES_ENERGY)
        self.assertAlmostEqual(res.ea_energy[0],    0.18735362996670174, self.PLACES_ENERGY)
        self.assertAlmostEqual(res.ee_s_energy[0],  0.34472451128191955, self.PLACES_ENERGY)
        self.assertAlmostEqual(res.ee_t_energy[0],  0.29463644214328730, self.PLACES_ENERGY)
        self.assertAlmostEqual(res.ee_sf_energy[0], 0.29463644349001655, self.PLACES_ENERGY)

        self.assertAlmostEqual(np.linalg.norm(res.ip_coeff[0][:nocc]),         0.9782629400729406, self.PLACES_ENERGY)
        self.assertAlmostEqual(np.linalg.norm(res.ea_coeff[0][:nvir]),         0.9979110706595700, self.PLACES_ENERGY)
        self.assertAlmostEqual(np.linalg.norm(res.ee_s_coeff[0][:nocc*nvir]),  0.6846761442905756, self.PLACES_ENERGY)
        self.assertAlmostEqual(np.linalg.norm(res.ee_t_coeff[0][:nocc*nvir]),  0.9990012466882306, self.PLACES_ENERGY)
        self.assertAlmostEqual(np.linalg.norm(res.ee_sf_coeff[0][:nocc*nvir]), 0.9990012297986601, self.PLACES_ENERGY)


class UMoleculeEWFTests(unittest.TestCase):
    PLACES_ENERGY = 7
    CONV_TOL = 1e-9
    CONV_TOL_NORMT = 1e-7

    def test_lih_ccpvdz_rhf_vs_uhf(self):
        """Compares RHF to UHF LiH cc-pvdz.
        """

        rewf = ewf.EWF(
                moles['lih_ccpvdz']['rhf'],
                bath_type=None,
                solver_options={
                    'conv_tol': self.CONV_TOL,
                    'conv_tol_normt': self.CONV_TOL_NORMT,
                },
        )
        rewf.sao_fragmentation()
        rewf.add_atomic_fragment([0, 1])
        rewf.kernel()

        uewf = ewf.UEWF(
                moles['lih_ccpvdz']['uhf'],
                bath_type=None,
                solver_options={
                    'conv_tol': self.CONV_TOL,
                    'conv_tol_normt': self.CONV_TOL_NORMT,
                },
        )
        uewf.sao_fragmentation()
        uewf.add_atomic_fragment([0, 1])
        uewf.kernel()

        self.assertAlmostEqual(rewf.e_corr, uewf.e_corr, self.PLACES_ENERGY)
        self.assertAlmostEqual(rewf.e_tot, uewf.e_tot, self.PLACES_ENERGY)

    def test_lih_ccpvdz_rhf_vs_uhf_cisd(self):
        """Compares RHF to UHF LiH cc-pvdz.
        """

        rewf = ewf.EWF(
                moles['lih_ccpvdz']['rhf'],
                solver='CISD',
                bath_type=None,
                solver_options={
                    'conv_tol': self.CONV_TOL,
                    'conv_tol_normt': self.CONV_TOL_NORMT,
                },
        )
        rewf.sao_fragmentation()
        rewf.add_atomic_fragment([0, 1])
        rewf.kernel()

        uewf = ewf.UEWF(
                moles['lih_ccpvdz']['uhf'],
                solver='CISD',
                bath_type=None,
                solver_options={
                    'conv_tol': self.CONV_TOL,
                    'conv_tol_normt': self.CONV_TOL_NORMT,
                },
        )
        uewf.sao_fragmentation()
        uewf.add_atomic_fragment([0, 1])
        uewf.kernel()

        self.assertAlmostEqual(rewf.e_corr, uewf.e_corr, self.PLACES_ENERGY)
        self.assertAlmostEqual(rewf.e_tot, uewf.e_tot, self.PLACES_ENERGY)

    def test_h2_ccpvdz_rhf_vs_uhf_fci(self):
        """Compares RHF to UHF with an FCI solver for H2 cc-pvdz.
        """

        rewf = ewf.EWF(
                moles['h2_ccpvdz']['rhf'],
                solver='FCI',
                bath_type=None,
                solver_options={
                    'conv_tol': self.CONV_TOL,
                },
        )
        rewf.sao_fragmentation()
        rewf.add_atomic_fragment([0, 1])
        rewf.kernel()

        uewf = ewf.UEWF(
                moles['h2_ccpvdz']['uhf'],
                solver='FCI',
                bath_type=None,
                solver_options={
                    'conv_tol': self.CONV_TOL,
                },
        )
        uewf.sao_fragmentation()
        uewf.add_atomic_fragment([0, 1])
        uewf.kernel()

        self.assertAlmostEqual(rewf.e_corr, uewf.e_corr, self.PLACES_ENERGY)
        self.assertAlmostEqual(rewf.e_tot, uewf.e_tot, self.PLACES_ENERGY)


if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
