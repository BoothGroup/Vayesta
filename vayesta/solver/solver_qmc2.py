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

# import pyscf.fci.direct_spin0
# import pyscf.fci.direct_spin1

from vayesta.core.util import *
from .fci2 import FCI_Solver, UFCI_Solver

from .cisd_coeff import Hamiltonian, RestoredCisdCoeffs
from .rdm_utils import load_spinfree_1rdm_from_m7, load_spinfree_1_2rdm_from_m7


class FCIQMCSolver(FCI_Solver):
    @dataclasses.dataclass
    class Options(FCI_Solver.Options):
        threads: int = 1
        lindep: float = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def read_mean_variational_energy(self, mae_stats_fname):
        return np.mean(np.loadtxt(mae_stats_fname)[:, 2])

    def setup_M7(self, path_to_M7, eris, ham_pkl_name, h5_name, stats_name, random_seed):
        """Setup for M7 calculation.
        """

        c_act = self.mo_coeff[:, self.get_active_slice()]

        if eris is None:
            eris = self.get_eris()

        h_eff = self.get_heff(eris, with_vext=True)

        fcisolver = pyscf.fci.direct_spin1.FCISolver(self.mol)
        if self.opts.threads is not None: fcisolver.threads = self.opts.threads
        if self.opts.conv_tol is not None: fcisolver.conv_tol = self.opts.conv_tol
        if self.opts.lindep is not None: fcisolver.lindep = self.opts.lindep

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
        M7_config_obj.M7_config_dict['wavefunction']['load_balancing']['acceptable_imbalance'] = 0.05
        M7_config_obj.M7_config_dict['propagator']['nw_target'] = 1000
        M7_config_obj.M7_config_dict['propagator']['stochastic'] = False
        M7_config_obj.M7_config_dict['propagator']['nadd'] = 0.0

        M7_config_obj.M7_config_dict['av_ests']['delay'] = 1000
        M7_config_obj.M7_config_dict['av_ests']['ncycle'] = 5000
        # M7_config_obj.M7_config_dict['av_ests']['ref_excits']['max_exlvl'] = 2
        # M7_config_obj.M7_config_dict['av_ests']['ref_excits']['archive']['save'] = 'yes'
        # M7_config_obj.M7_config_dict['av_ests']['rdm']['ranks'] = ['1', '2']
        M7_config_obj.M7_config_dict['av_ests']['rdm']['archive']['save'] = 'yes'

        M7_config_obj.M7_config_dict['archive']['save_path'] = h5_name
        M7_config_obj.M7_config_dict['hamiltonian']['fcidump']['path'] = FCIDUMP_name
        M7_config_obj.M7_config_dict['stats']['path'] = stats_name
        return M7_config_obj

    # def gen_M7_results(self, h5_name, ham_pkl_name, M7_config_obj, coeff_pkl_name, eris):
    def gen_M7_results(self, h5_name, eris):
        """Generate M7 results object."""
        e_qmc = self.read_mean_variational_energy('M7.maes.stats')
        print(f'mean variational energy from M7 MAE stats: {e_qmc}')
        ref_energy = self.fragment.base.e_mf

        # e_qmc = cisd_coeffs.energy()
        e_corr_qmc = e_qmc - ref_energy
        # print('e_qmc', e_qmc)
        # saving coefficients into pickle
        # cisd_coeffs.to_pickle(coeff_pkl_name)

        results = self.Results(
            converged=1, e_corr=e_corr_qmc, c_occ=self.c_active_occ, c_vir=self.c_active_vir, eris=eris,
            c0=None, c1=None, c2=None)
        # c0=c0_qmc, c1=c1_qmc, c2=c2_qmc)

        if self.opts.make_rdm2:
            results.dm1, results.dm2 = load_spinfree_1_2rdm_from_m7(h5_name, self.nelec)
            print('Running RDM checks')
            tr1 = np.sum(np.diag(results.dm1))
            tr2 = np.sum(np.diagonal(results.dm2))
            print("nelec", self.nelec)
            print("Trace of 1RDM %1.16f Expected %3f", tr1, self.nelec)
            print("Trace of 2RDM %1.16f Expected %3f", tr1, self.nelec * (self.nelec - 1) / 2)

            print('Partial trace of 2RDM/1RDM')
            print(np.diag(np.einsum('ijkk->ij', results.dm2)) / (self.nelec - 1))
            print(np.diag(results.dm1))

        elif self.opts.make_rdm1:
            results.dm1 = load_spinfree_1rdm_from_m7(h5_name)

        return results

    def test(self):
        M7_config_obj = M7_config_to_dict('/work/robert/ebfciqmc/M7')
        return M7_config_obj

    def kernel(self, init_guess=None, eris=None):
        """Run FCI kernel."""

        # All parameters that need to be set here...
        random_seed = 1
        h5_name = 'M7.cluster' + str(int(self.fragment.id)) + '.' + str(random_seed) + '.h5'
        stats_name = 'M7.cluster' + str(int(self.fragment.id)) + '.stats'
        # This function covers all M7 configuration, so if it is overloaded
        print(self.setup_M7(path_to_M7, eris, h5_name, stats_name, random_seed))
        M7_config_obj = self.setup_M7(path_to_M7, eris, h5_name, stats_name, random_seed)
        # Writing settings for M7 in a yaml
        yaml_name = 'M7_settings.yaml'

        M7_config_obj.write_yaml(yaml_name)

        '''
        # Run M7
        print("Running M7 FCIQMC...")
        process = Popen(f'{mpirun_exe} -np {nrank_mpi} {path_to_M7} {yaml_name}', stdout=PIPE, stderr=PIPE, shell=True)
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
        '''

        return self.gen_M7_results(ham_pkl_name, eris)  # M7_config_obj, coeff_pkl_name, h5_name, eris)
        # return self.gen_M7_results(ham_pkl_name, M7_config_obj, coeff_pkl_name, h5_name, eris)
