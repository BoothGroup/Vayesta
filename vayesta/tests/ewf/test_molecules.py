import unittest
import numpy as np

from vayesta import ewf
from vayesta.tests.common import TestCase
from vayesta.tests import testsystems


BATH_OPTS = dict(project_dmet_order = 0)


class MoleculeTests(TestCase):
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

        self.assertAlmostEqual(emb.get_dmet_energy(part_cumulant=False), known_values['e_dmet'], self.PLACES_ENERGY)

    def _test_t1(self, emb, known_values):
        """Tests the T1 and L1 amplitudes.
        """
        t1 = emb.get_global_t1()
        l1 = emb.get_global_l1()

        self.assertAlmostEqual(np.linalg.norm(t1), known_values['t1'], self.PLACES_T)
        self.assertAlmostEqual(np.linalg.norm(l1), known_values['l1'], self.PLACES_T)

    def _test_t2(self, emb, known_values):
        """Tests the T2 and L2 amplitudes.
        """
        t2 = emb.get_global_t2()
        l2 = emb.get_global_l2()

        self.assertAlmostEqual(np.linalg.norm(t2), known_values['t2'], self.PLACES_T)
        self.assertAlmostEqual(np.linalg.norm(l2), known_values['l2'], self.PLACES_T)

    def _test_rdm1(self, emb, known_values):
        """Tests the traces of the first-order density matrices.
        """

        dm = emb.make_rdm1_demo()
        self.assertAlmostEqual(np.trace(dm), known_values['rdm1_demo'], self.PLACES_DM)

        dm = emb.make_rdm1_demo(ao_basis=True)
        self.assertAlmostEqual(np.trace(dm), known_values['rdm1_demo_ao'], self.PLACES_DM)

        dm = emb._make_rdm1_ccsd()
        self.assertAlmostEqual(np.trace(dm), known_values['rdm1_ccsd'], self.PLACES_DM)

        dm = emb._make_rdm1_ccsd(ao_basis=True)
        self.assertAlmostEqual(np.trace(dm), known_values['rdm1_ccsd_ao'], self.PLACES_DM)

    def _test_rdm2(self, emb, known_values):
        """Tests the traces of the second-order density matrices.
        """

        trace = lambda m: np.einsum('iiii->', m)

        dm = emb.make_rdm2_demo()
        self.assertAlmostEqual(trace(dm), known_values['rdm2_demo'], self.PLACES_DM)

        dm = emb.make_rdm2_demo(ao_basis=True, part_cumulant=False)
        self.assertAlmostEqual(trace(dm), known_values['rdm2_demo_ao'], self.PLACES_DM)

        dm = emb._make_rdm2_ccsd_global_wf()
        self.assertAlmostEqual(trace(dm), known_values['rdm2_ccsd'], self.PLACES_DM)

        dm = emb._make_rdm2_ccsd_global_wf(ao_basis=True)
        self.assertAlmostEqual(trace(dm), known_values['rdm2_ccsd_ao'], self.PLACES_DM)

    def test_lih_iao_atoms(self):
        """Tests EWF for LiH cc-pvdz with IAO atomic fragmentation.
        """

        emb = ewf.EWF(
                testsystems.lih_ccpvdz.rhf(),
                bath_options={'bathtype': 'full', **BATH_OPTS},
                solver_options={
                    'solve_lambda': True,
                    'conv_tol': self.CONV_TOL,
                    'conv_tol_normt': self.CONV_TOL_NORMT,
                },
        )
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

    def test_lih_sao_orbitals(self):
        """Tests EWF for LiH cc-pvdz with SAO orbital fragmentation.
        """

        emb = ewf.EWF(
                testsystems.lih_ccpvdz.rhf(),
                bath_options={'threshold': 1e-5/2, **BATH_OPTS},
                solver_options={
                    'conv_tol': self.CONV_TOL,
                    'conv_tol_normt': self.CONV_TOL_NORMT,
                },
        )
        with emb.sao_fragmentation() as f:
            f.add_orbital_fragment([0, 1])
            f.add_orbital_fragment([2, 3, 4])
        emb.kernel()

        known_values = {'e_tot': -7.98424889149862}

        self._test_energy(emb, known_values)

    def test_lih_sao_atoms(self):
        """Tests EWF for LiH cc-pvdz with SAO atomic fragmentation.
        """

        emb = ewf.EWF(
                testsystems.lih_ccpvdz.rhf(),
                bath_options={'bathtype': 'dmet'},
                solver_options={
                    'conv_tol': self.CONV_TOL,
                    'conv_tol_normt': self.CONV_TOL_NORMT,
                },
        )
        with emb.sao_fragmentation() as f:
            f.add_all_atomic_fragments()
        emb.kernel()

        known_values = {'e_tot': -7.99502192669842}

        self._test_energy(emb, known_values)

    def test_h2o_FCI(self):
        """Tests EWF for H2O cc-pvdz with FCI solver.
        """

        emb = ewf.EWF(
                testsystems.water_ccpvdz.rhf(),
                bath_options={'bathtype': 'dmet'},
                solver='FCI',
                solver_options={
                    'conv_tol': 1e-12,
                    'fix_spin': 0.0,
                }
        )
        with emb.iao_fragmentation() as f:
            f.add_atomic_fragment(0)
            f.add_atomic_fragment(1, sym_factor=2)
        emb.kernel()

        known_values = {'e_tot': -76.06365118513072}

        self._test_energy(emb, known_values)

    def test_h2o_FCI_noguess(self):
        """Tests EWF for H2O cc-pvdz with FCI solver.
        """

        emb = ewf.EWF(
                testsystems.water_ccpvdz.rhf(),
                bath_options={'bathtype': 'dmet'},
                solver='FCI',
                solver_options={
                    'init_guess': 'meanfield',
                    'conv_tol': 100,
                }
        )
        emb.kernel()

        known_values = {'e_tot': -76.02677312899381}

        self._test_energy(emb, known_values)

    def test_h2o_TCCSD(self):
        """Tests EWF for H2O cc-pvdz with FCI solver.
        """

        emb = ewf.EWF(
                testsystems.water_ccpvdz.rhf(),
                solver='TCCSD',
                bath_options={'threshold': 1e-4/2, **BATH_OPTS},
                solver_options={
                    'conv_tol': self.CONV_TOL,
                    'conv_tol_normt': self.CONV_TOL_NORMT,
                },
        )
        with emb.iao_fragmentation() as f:
            f.add_atomic_fragment(0)
            f.add_atomic_fragment(1, sym_factor=2)
        emb.kernel()

        known_values = {'e_tot': -76.23613576956096}
        self.assertAlmostEqual(emb.e_tot, known_values['e_tot'], 6)

    def test_tccsd_solver(self):
        """Tests EWF for H2O cc-pvdz with TCCSD solver and CAS picker."""
        emb = ewf.EWF(
                testsystems.water_ccpvdz.rhf(),
                solver='TCCSD',
                bath_options={'threshold': 1e-4/2, **BATH_OPTS},
                solver_options={
                    'conv_tol': self.CONV_TOL,
                    'conv_tol_normt': self.CONV_TOL_NORMT,
                },
        )
        with emb.iao_fragmentation() as f:
            frag1 = f.add_atomic_fragment(0)
            f.add_atomic_fragment(1, sym_factor=2)
        frag1.set_cas(['0 O 2p'])
        emb.kernel()
        self.assertAllclose(emb.e_tot, -76.23559827815198, rtol=0)

    def test_tccsd_via_fragment(self):
        """Tests EWF for H2O cc-pvdz with tailoring FCI fragment.
        (Should reproduce result of test_tccsd_solver)"""
        emb = ewf.EWF(
                testsystems.water_ccpvdz.rhf(),
                bath_options={'threshold': 1e-4/2, **BATH_OPTS},
                solver_options={
                    'conv_tol': self.CONV_TOL,
                    'conv_tol_normt': self.CONV_TOL_NORMT,
                },
        )
        with emb.iao_fragmentation() as f:
            cas_o = f.add_atomic_fragment(0, orbital_filter='2p', solver='FCI', bath_options=dict(bathtype='dmet'))
            cas_h = f.add_atomic_fragment(1, solver='FCI', bath_options=dict(bathtype='dmet'))
            frag_o = f.add_atomic_fragment(0, solver='extCCSD', active=False)
            frag_h = f.add_atomic_fragment(1, solver='extCCSD', sym_factor=2, active=False)
        emb.kernel()
        frag_o.tailor_with_fragments([cas_o], projectors=0)
        frag_h.tailor_with_fragments([cas_h], projectors=0)
        cas_o.active = cas_h.active = False
        frag_o.active = frag_h.active = True
        emb.kernel()
        self.assertAllclose(emb.e_tot, -76.23559827815198, atol=1e-6, rtol=0)

    def test_h2o_sc(self):
        """Tests EWF for H2O cc-pvdz with self-consistency.
        """

        emb = ewf.EWF(
                testsystems.water_ccpvdz.rhf(),
                bath_options={'threshold': 1e-4/2, **BATH_OPTS},
                sc_mode=1,
                solver_options={
                    'conv_tol': self.CONV_TOL,
                    'conv_tol_normt': self.CONV_TOL_NORMT,
                },
        )
        with emb.iao_fragmentation() as f:
            f.add_atomic_fragment(0)
            f.add_atomic_fragment(1, sym_factor=2)
        emb.kernel()

        known_values = {'e_tot': -76.23147227604929}

        self._test_energy(emb, known_values)

    def test_h2_cisd(self):
        """Compares CCSD and CISD solvers for a two-electron system.
        """

        ecisd = ewf.EWF(
                testsystems.h2_ccpvdz.rhf(),
                bath_type='dmet',
                solver='CISD',
                solver_options={
                    'conv_tol': self.CONV_TOL,
                    'conv_tol_normt': self.CONV_TOL_NORMT,
                },
        )
        ecisd.kernel()

        eccsd = ewf.EWF(
                testsystems.h2_ccpvdz.rhf(),
                bath_type='dmet',
                solver='CCSD',
                solver_options={
                    'conv_tol': self.CONV_TOL,
                    'conv_tol_normt': self.CONV_TOL_NORMT,
                },
        )
        eccsd.kernel()

        self.assertAlmostEqual(ecisd.e_corr, eccsd.e_corr)
        self.assertAlmostEqual(ecisd.e_tot, eccsd.e_tot)

    def test_h2o_mp2_compression(self):
        """Compares MP2 solver with ERIs, uncompressed CDERIs, and compressed CDERIs.
        """
        rewf = ewf.EWF(
                testsystems.water_ccpvdz.rhf(),
                bath_type='dmet',
                solver="MP2"
        )

        with rewf.iao_fragmentation() as f:
            f.add_all_atomic_fragments()
        rewf.kernel()

        df_rewf = ewf.EWF(
                testsystems.water_ccpvdz_df.rhf(),
                bath_type='dmet',
                solver="MP2",
                solver_options={
                    "compress_cderi":False
                }
        )
        with df_rewf.iao_fragmentation() as f:
            f.add_all_atomic_fragments()
        df_rewf.kernel()
        # Check that density fitting error is smaller than in mean-field.
        self.assertAlmostEqual(rewf.e_corr, df_rewf.e_corr, int(np.log10(abs(rewf.e_mf - df_rewf.e_mf))))

        cdf_rewf = ewf.EWF(
                testsystems.water_ccpvdz_df.rhf(),
                bath_type='dmet',
                solver="MP2",
                solver_options={
                    "compress_cderi":True
                }
        )
        with cdf_rewf.iao_fragmentation() as f:
            f.add_all_atomic_fragments()
        cdf_rewf.kernel()
        # Compression shouldn't change the answer.
        self.assertAlmostEqual(df_rewf.e_tot, cdf_rewf.e_tot, self.PLACES_ENERGY)

class MoleculeTestsUnrestricted(unittest.TestCase):
    PLACES_ENERGY = 7
    CONV_TOL = 1e-9
    CONV_TOL_NORMT = 1e-7

    def test_lih_rhf_vs_uhf(self):
        """Compares RHF to UHF LiH cc-pvdz.
        """

        rewf = ewf.EWF(
                testsystems.lih_ccpvdz.rhf(),
                bath_type='dmet',
                solver_options={
                    'conv_tol': self.CONV_TOL,
                    'conv_tol_normt': self.CONV_TOL_NORMT,
                },
        )
        with rewf.sao_fragmentation() as f:
            f.add_atomic_fragment([0, 1])
        rewf.kernel()

        uewf = ewf.UEWF(
                testsystems.lih_ccpvdz.uhf(),
                bath_type='dmet',
                solver_options={
                    'conv_tol': self.CONV_TOL,
                    'conv_tol_normt': self.CONV_TOL_NORMT,
                },
        )
        with uewf.sao_fragmentation() as f:
            f.add_atomic_fragment([0, 1])
        uewf.kernel()

        self.assertAlmostEqual(rewf.e_corr, uewf.e_corr, self.PLACES_ENERGY)
        self.assertAlmostEqual(rewf.e_tot, uewf.e_tot, self.PLACES_ENERGY)

    def test_lih_rhf_vs_uhf_cisd(self):
        """Compares RHF to UHF LiH cc-pvdz.
        """

        rewf = ewf.EWF(
                testsystems.lih_ccpvdz.rhf(),
                solver='CISD',
                bath_type='dmet',
                solver_options={
                    'conv_tol': self.CONV_TOL,
                    'conv_tol_normt': self.CONV_TOL_NORMT,
                },
        )
        with rewf.sao_fragmentation() as f:
            f.add_atomic_fragment([0, 1])
        rewf.kernel()

        uewf = ewf.UEWF(
                testsystems.lih_ccpvdz.uhf(),
                solver='CISD',
                bath_type='dmet',
                solver_options={
                    'conv_tol': self.CONV_TOL,
                    'conv_tol_normt': self.CONV_TOL_NORMT,
                },
        )
        with uewf.sao_fragmentation() as f:
            f.add_atomic_fragment([0, 1])
        uewf.kernel()

        self.assertAlmostEqual(rewf.e_corr, uewf.e_corr, self.PLACES_ENERGY)
        self.assertAlmostEqual(rewf.e_tot, uewf.e_tot, self.PLACES_ENERGY)

    def test_h2_rhf_vs_uhf_fci(self):
        """Compares RHF to UHF with an FCI solver for H2 cc-pvdz.
        """

        rewf = ewf.EWF(
                testsystems.h2_ccpvdz.rhf(),
                solver='FCI',
                bath_type='dmet',
                solver_options={
                    'conv_tol': 1e-12,
                },
        )
        with rewf.sao_fragmentation() as f:
            f.add_atomic_fragment([0, 1])
        rewf.kernel()

        uewf = ewf.UEWF(
                testsystems.h2_ccpvdz.uhf(),
                solver='FCI',
                bath_type='dmet',
                solver_options={
                    'conv_tol': 1e-12,
                },
        )
        with uewf.sao_fragmentation() as f:
            f.add_atomic_fragment([0, 1])
        uewf.kernel()

        self.assertAlmostEqual(rewf.e_corr, uewf.e_corr, self.PLACES_ENERGY)
        self.assertAlmostEqual(rewf.e_tot, uewf.e_tot, self.PLACES_ENERGY)

    def test_lih_rhf_vs_uhf_CAS(self):
        """Compares RHF to UHF LiH cc-pvdz using a 2,4 CAS as a fragment.
        """

        rewf = ewf.EWF(
                testsystems.lih_ccpvdz.rhf(),
                bath_type='dmet',
                solver_options={
                    'conv_tol': self.CONV_TOL,
                    'conv_tol_normt': self.CONV_TOL_NORMT,
                },
        )
        with rewf.cas_fragmentation() as f:
            f.add_cas_fragment(4, 2)
        rewf.kernel()

        uewf = ewf.UEWF(
                testsystems.lih_ccpvdz.uhf(),
                bath_type='dmet',
                solver_options={
                    'conv_tol': self.CONV_TOL,
                    'conv_tol_normt': self.CONV_TOL_NORMT,
                },
        )
        with uewf.cas_fragmentation() as f:
            f.add_cas_fragment(4, 2)
        uewf.kernel()

        self.assertAlmostEqual(rewf.e_corr, uewf.e_corr, self.PLACES_ENERGY)
        self.assertAlmostEqual(rewf.e_tot, uewf.e_tot, self.PLACES_ENERGY)

    def test_h2o_nelectron_target(self):
        """Tests `nelectron_target` fragment option results in a local dm1 with correct number of electrons in fragment space.
        Also compares to ensure UHF obtains the same result."""

        remb = ewf.EWF(
                testsystems.water_ccpvdz.rhf(),
                bath_options={
                    "bathtype":"dmet"
                },
                solver_options={
                    'conv_tol': self.CONV_TOL,
                    'conv_tol_normt': self.CONV_TOL_NORMT,
                },
        )
        with remb.iao_fragmentation() as f:
            f.add_atomic_fragment([0], nelectron_target=8.0)

        remb.kernel()

        fr = remb.fragments[0]
        p = fr.get_fragment_projector(fr.cluster.c_active)
        nel = np.einsum("pq,qr,rp->", p, fr.results.wf.make_rdm1(), p)
        self.assertAlmostEqual(nel, 8.0, places=4)

        uemb = ewf.EWF(
                testsystems.water_ccpvdz.uhf(),
                bath_options={
                    "bathtype":"dmet"
                },
                solver_options={
                    'conv_tol': self.CONV_TOL,
                    'conv_tol_normt': self.CONV_TOL_NORMT,
                },
        )
        with uemb.iao_fragmentation() as f:
            f.add_atomic_fragment([0], nelectron_target=8.0)

        uemb.kernel()

        self.assertAlmostEqual(remb.e_corr, uemb.e_corr, self.PLACES_ENERGY)
        self.assertAlmostEqual(remb.e_corr, -0.018721331700187655, self.PLACES_ENERGY)


if __name__ == '__main__':
    print('Running %s' % __file__)
    unittest.main()
