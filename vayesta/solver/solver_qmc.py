import dataclasses
from timeit import default_timer as timer

import numpy as np

import pyscf
import pyscf.ao2mo
import pyscf.ci
import pyscf.mcscf
import pyscf.fci
from pyscf import tools, ao2mo, fci


from .M7_config_yaml_helper import M7_config_to_dict
from subprocess import Popen, PIPE

#import pyscf.fci.direct_spin0
#import pyscf.fci.direct_spin1

from vayesta.core.util import *
from .solver import ClusterSolver

from .cisd_coeff import Hamiltonian, RestoredCisdCoeffs
from .rdm_utils import load_spinfree_1rdm_from_m7, load_spinfree_1_2rdm_from_m7

class FCIQMCSolver(ClusterSolver):


    class Options(ClusterSolver.Options):
        threads: int = 1
        lindep: float = None
        conv_tol: float = None


    @dataclasses.dataclass
    class Results(ClusterSolver.Results):
        # CI coefficients
        c0: float = None
        c1: np.array = None
        c2: np.array = None


    def setup_M7(self, path_to_M7, eris, ham_pkl_name, h5_name, stats_name, random_seed):

        """Setup for M7 calculation.
        """

        c_act = self.mo_coeff[:, self.get_active_slice()]

        if eris is None:
            # Temporary implementation
            import pyscf.ao2mo
            t0 = timer()
            eris = pyscf.ao2mo.full(self.mf._eri, c_act, compact=False).reshape(4 * [self.nactive])
            self.log.timing("Time for AO->MO of (ij|kl):  %s", time_string(timer() - t0))

        nocc = self.nocc - self.nocc_frozen
        occ = np.s_[:nocc]
        vir = np.s_[nocc:]

        f_act = np.linalg.multi_dot((c_act.T, self.base.get_fock(), c_act))
        v_act = 2 * einsum('iipq->pq', eris[occ, occ]) - einsum('iqpi->pq', eris[occ, :, :, occ])
        h_eff = f_act - v_act
        # This should be equivalent to:
        # core = np.s_[:self.nocc_frozen]
        # dm_core = 2*np.dot(self.mo_coeff[:,core], self.mo_coeff[:,core].T)
        # v_core = self.mf.get_veff(dm=dm_core)
        # h_eff = np.linalg.multi_dot((c_act.T, self.base.get_hcore()+v_core, c_act))

        fcisolver = pyscf.fci.direct_spin1.FCISolver(self.mol)
        if self.opts.threads is not None: fcisolver.threads = self.opts.threads
        if self.opts.conv_tol is not None: fcisolver.conv_tol = self.opts.conv_tol
        if self.opts.lindep is not None: fcisolver.lindep = self.opts.lindep

        nelec = sum(self.mo_occ[self.get_active_slice()])

        h0 = 0.0  # No 0-electron energy for lattice models
        FCIDUMP_name = 'FCIDUMP_cluster' + str(int(self.fragment.id))

        # Writing Hamiltonian for M7 (FCIDUMP) and Python analysis (pkl)
        qmc_H = Hamiltonian()
        qmc_H.from_arrays(h0, h_eff, eris, nelec)
        qmc_H.to_pickle(ham_pkl_name)
        qmc_H.write_fcidump(FCIDUMP_name)


        M7_config_obj = M7_config_to_dict(path_to_M7)
        # Make the changes on M7 config you want here, for example:
        M7_config_obj.M7_config_dict['prng']['seed'] = random_seed
        M7_config_obj.M7_config_dict['wavefunction']['nw_init'] = 3
        M7_config_obj.M7_config_dict['wavefunction']['load_balancing']['period'] = 100
        M7_config_obj.M7_config_dict['wavefunction']['load_balancing']['nnull_updates_deactivate'] = 8
        M7_config_obj.M7_config_dict['propagator']['nw_target'] = 30000

        M7_config_obj.M7_config_dict['av_ests']['delay'] = 1000
        M7_config_obj.M7_config_dict['av_ests']['ncycle'] = 5000
        M7_config_obj.M7_config_dict['av_ests']['ref_excits']['max_exlvl'] = 2
        M7_config_obj.M7_config_dict['av_ests']['ref_excits']['archive']['save'] = 'yes'
        rdm_ranks = []
        if self.opts.make_rdm1: rdm_ranks.append('1')
        if self.opts.make_rdm2: rdm_ranks.append('2')
        M7_config_obj.M7_config_dict['av_ests']['rdm']['ranks'] = rdm_ranks
        if (len(rdm_ranks)):
            M7_config_obj.M7_config_dict['av_ests']['rdm']['archive']['save'] = 'yes'

        M7_config_obj.M7_config_dict['hamiltonian']['nboson_max'] = self.opts.nboson_max


        M7_config_obj.M7_config_dict['archive']['save_path'] = h5_name
        M7_config_obj.M7_config_dict['hamiltonian']['fcidump']['path'] = FCIDUMP_name
        M7_config_obj.M7_config_dict['stats']['path'] = stats_name
        return M7_config_obj

    def gen_M7_results(self, h5_name, ham_pkl_name, M7_config_obj, coeff_pkl_name, eris):
        """Generate M7 results object."""

        # Calling analysis for M7 FCIQMC
        ham = Hamiltonian()

        # Loading Hamiltonian from the pickle
        ham.from_pickle(ham_pkl_name)
        cisd_coeffs = RestoredCisdCoeffs(ham)
        cisd_coeffs.from_m7(M7_config_obj.M7_config_dict['archive']['save_path'])
        cisd_coeffs.normalise()

        c0_qmc = cisd_coeffs.c0

        c1a_qmc = cisd_coeffs.c1a
        c1b_qmc = cisd_coeffs.c1b
        c1_qmc = (c1a_qmc + c1b_qmc) / 2

        c2aa_qmc = cisd_coeffs.c2aa.transpose(0, 2, 1, 3)
        c2bb_qmc = cisd_coeffs.c2bb.transpose(0, 2, 1, 3)
        c2ab_qmc = cisd_coeffs.c2ab.transpose(0, 2, 1, 3)
        c2_qmc = (c2aa_qmc + 2 * c2ab_qmc.transpose(0, 1, 3, 2) + c2bb_qmc) / 2

        e_qmc = cisd_coeffs.energy()
        e_corr_qmc = e_qmc - cisd_coeffs.ref_energy()
        print('e_qmc', e_qmc)
        # saving coefficients into pickle
        cisd_coeffs.to_pickle(coeff_pkl_name)

        results = self.Results(
            converged=1, e_corr=e_corr_qmc, c_occ=self.c_active_occ, c_vir=self.c_active_vir, eris=eris,
            c0=c0_qmc, c1=c1_qmc, c2=c2_qmc)

        if self.opts.make_rdm2:
            nelec = sum(self.mo_occ[self.get_active_slice()])
            results.dm1, results.dm2 = load_spinfree_1_2rdm_from_m7(h5_name, nelec)
            print('Running RDM checks')
            tr1 = np.sum(np.diag(results.dm1))
            tr2 = np.sum(np.diagonal(results.dm2))
            print("nelec", nelec)
            print("Trace of 1RDM %1.16f Expected %3f", tr1, nelec)
            print("Trace of 2RDM %1.16f Expected %3f", tr1, nelec * (nelec - 1) / 2)

            print('Partial trace of 2RDM/1RDM')
            print(np.diag(np.einsum('ijkk->ij', results.dm2)) / (nelec - 1))
            print(np.diag(results.dm1))

            assert 0

        elif self.opts.make_rdm1:
            results.dm1 = load_spinfree_1rdm_from_m7(h5_name)


        return results

    def kernel(self, init_guess=None, eris=None):
        """Run FCI kernel."""

        # All parameters that need to be set here...
        random_seed = 1
        path_to_M7 = '/Users/marcell/Desktop/RESEARCH/M7'
        ham_pkl_name = 'Hubbard_Hamiltonian_cluster' + str(int(self.fragment.id)) + '.pkl'
        h5_name = 'M7.cluster' + str(int(self.fragment.id)) + '.' + str(random_seed) + '.h5'
        stats_name = 'M7.cluster' + str(int(self.fragment.id)) + '.stats'
        coeff_pkl_name = 'cluster' + str(int(self.fragment.id)) + '_coeff.pkl'
        # This function covers all M7 configuration, so if it is overloaded
        M7_config_obj = self.setup_M7(path_to_M7, eris, ham_pkl_name, h5_name, stats_name, random_seed)
        # Writing settings for M7 in a yaml
        yaml_name = 'M7_settings.yaml'

        M7_config_obj.write_yaml(yaml_name)

        # Run M7
        print("Running M7 FCIQMC...")
        process = Popen('/usr/local/bin/mpirun -np 1 ' + path_to_M7 + '/build/src/release ' + yaml_name, stdout=PIPE, stderr=PIPE, shell=True)
        stdout, stderr = process.communicate()
        stdout = stdout.decode("utf-8")
        stderr = stderr.decode("utf-8")
        
        if stderr == "":
            print("Successful M7 run!")
        else:
            print("Error from M7... printing the log:")
            print(stdout)
            print(stderr)
            assert 0

        return self.gen_M7_results(ham_pkl_name, M7_config_obj, coeff_pkl_name, h5_name, eris)
