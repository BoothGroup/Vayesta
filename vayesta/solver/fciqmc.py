import dataclasses
from timeit import default_timer as timer

import numpy as np

from pyscf import tools, ao2mo, fci
from vayesta.core import ufcidump

from .M7_config_yaml_helper import M7_config_to_dict
from .m7_settings import path_to_M7, nrank_mpi, mpirun_exe

# import pyscf.fci.direct_spin0
# import pyscf.fci.direct_spin1

from vayesta.core.util import *
from .fci2 import FCI_Solver, UFCI_Solver

from .rdm_utils import load_spinfree_1rdm_from_m7, load_spinfree_1_2rdm_from_m7


header = ''' &FCI NORB= {}
UHF=.TRUE.
 &END
'''

class FCIQMCSolver(FCI_Solver):
    @dataclasses.dataclass
    class Options(FCI_Solver.Options):
        threads: int = 1

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def read_mean_variational_energy(self, mae_stats_fname):
        return np.mean(np.loadtxt(mae_stats_fname)[:, 2])

    def write_fcidumps(self, eris):
        if eris is None:
            eris = self.get_eris()

        h_eff = self.get_heff(eris, with_vext=True)

        h0 = self.fragment.base.mf.energy_nuc()  # No 0-electron energy for lattice models
        self.FCIDUMP_name = 'FCIDUMP_cluster' + str(int(self.fragment.id))

        tools.fcidump.from_integrals(self.FCIDUMP_name, h_eff, eris, self.cluster.norb_active, self.nelec, h0, 0,
                                     orbsym=None)#[1,]*self.norb_active)

    def setup_M7(self, path_to_M7, eris, h5_name, stats_name, random_seed):
        """Setup for M7 calculation.
        """
        self.write_fcidumps(eris)

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
        M7_config_obj.M7_config_dict['hamiltonian']['fcidump']['path'] = self.FCIDUMP_name
        M7_config_obj.M7_config_dict['stats']['path'] = stats_name
        return M7_config_obj

    def get_e_corr(self):
        """Generate M7 results object."""
        e_qmc = self.read_mean_variational_energy('M7.maes.stats')
        print(f'mean variational energy from M7 MAE stats: {e_qmc}')
        ref_energy = self.fragment.base.e_mf
        return e_qmc - ref_energy

    def test(self):
        M7_config_obj = M7_config_to_dict('/work/robert/ebfciqmc/M7')
        return M7_config_obj

    def kernel(self, init_guess=None, eris=None):
        """Run FCI kernel."""

        # All parameters that need to be set here...
        random_seed = 1
        self.h5_name = 'M7.cluster' + str(int(self.fragment.id)) + '.' + str(random_seed) + '.h5'
        self.stats_name = 'M7.cluster' + str(int(self.fragment.id)) + '.stats'
        # This function covers all M7 configuration, so if it is overloaded
        print(self.setup_M7(path_to_M7, eris, self.h5_name, self.stats_name, random_seed))
        M7_config_obj = self.setup_M7(path_to_M7, eris, self.h5_name, self.stats_name, random_seed)
        # Writing settings for M7 in a yaml
        self.yaml_name = 'M7_settings.yaml'

        M7_config_obj.write_yaml(self.yaml_name)
        self.log.info("Wrote M7 input files to ", self.yaml_name, self.FCIDUMP_name)
        self.log.info("Expect output files to be written to ", self.h5_name, self.stats_name)
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
        try:
            self.e_corr = self.get_e_corr()
        except Exception as e:
            self.log.critical("Failed to get correlation energy estimate from M7; have output files been correctly created?")
            raise e

    def make_rdm1(self):
        return load_spinfree_1rdm_from_m7(self.h5_name)

    def make_rdm12(self):
        dm1, dm2 = load_spinfree_1_2rdm_from_m7(self.h5_name, self.nelec)
        print('Running RDM checks')
        tr1 = np.sum(np.diag(dm1))
        tr2 = np.sum(np.diagonal(dm2))
        print("nelec", self.nelec)
        print("Trace of 1RDM %1.16f Expected %3f", tr1, self.nelec)
        print("Trace of 2RDM %1.16f Expected %3f", tr1, self.nelec * (self.nelec - 1) / 2)

        print('Partial trace of 2RDM/1RDM')
        print(np.diag(np.einsum('ijkk->ij', dm2)) / (self.nelec - 1))
        print(np.diag(dm1))
        return dm1, dm2

    def make_rdm2(self):
        return self.make_rdm12()[2]


class UFCIQMCSolver(FCIQMCSolver, UFCI_Solver):
    def write_fcidumps(self, eris):
        if eris is None:
            eris = self.get_eris()

        h_eff = self.get_heff(eris, with_vext=True)

        h0 = self.fragment.base.mf.energy_nuc()  # No 0-electron energy for lattice models
        self.FCIDUMP_name = 'FCIDUMP_cluster' + str(int(self.fragment.id))

        ufcidump.from_integrals(self.FCIDUMP_name, h_eff, eris, self.cluster.norb_active, self.nelec, h0, 0,
                                     orbsym=None)  # [1,]*self.norb_active)
