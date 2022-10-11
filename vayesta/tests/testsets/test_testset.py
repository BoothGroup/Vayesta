from collections import namedtuple, defaultdict
import unittest
import numpy as np
import pyscf
import pyscf.scf
import pyscf.mp
import pyscf.cc
import vayesta
import vayesta.ewf
import vayesta.misc
from vayesta.core.util import *
from vayesta.misc import gmtkn55
from vayesta.tests.common import TestCase


Result = namedtuple('Result', ['ncluster_mean', 'ncluster_max', 'energy_dm_error', 'energy_wf_error', 'time'])


def _run_hf(mol):
    if mol.spin > 0:
        hf = pyscf.scf.UHF(mol)
    else:
        hf = pyscf.scf.RHF(mol)
    hf.kernel()
    return hf

def _run_mp2(hf):
    if hf.mol.spin > 0:
        mp2 = pyscf.mp.UMP2(hf)
    else:
        mp2 = pyscf.mp.MP2(hf)
    mp2.kernel()
    return mp2

def _run_ccsd(hf):
    if hf.mol.spin > 0:
        cc = pyscf.cc.UCCSD(hf)
    else:
        cc = pyscf.cc.CCSD(hf)
    cc.kernel()
    return cc

class Test_TestSet(TestCase):

    @classmethod
    def setUpClass(cls):
        cls.results = defaultdict(dict)

    @classmethod
    def tearDownClass(cls):
        cls.analyze('%s.out' % cls.__name__)

    @classmethod
    def analyze(self, filename):
        n_mean = np.zeros(len(self.results))
        n_max = np.zeros(len(self.results))
        e_dm_mae = np.zeros(len(self.results))
        e_wf_mae = np.zeros(len(self.results))
        e_dm_rmse = np.zeros(len(self.results))
        e_wf_rmse = np.zeros(len(self.results))
        t_mean = np.zeros(len(self.results))
        for idx, (peta, samples) in enumerate(self.results.items()):
            for sample, result in samples.items():
                n_mean[idx] += result.ncluster_mean
                n_max[idx] = max(n_max[idx], result.ncluster_max)
                e_dm_mae[idx] += abs(result.energy_dm_error)
                e_wf_mae[idx] += abs(result.energy_wf_error)
                e_dm_rmse[idx] += result.energy_dm_error**2
                e_wf_rmse[idx] += result.energy_wf_error**2
                t_mean[idx] += result.time
            n_mean[idx] = n_mean[idx] / len(samples)
            e_dm_mae[idx] = e_dm_mae[idx] / len(samples)
            e_wf_mae[idx] = e_wf_mae[idx] / len(samples)
            e_dm_rmse[idx] = np.sqrt(e_dm_rmse[idx] / len(samples))
            e_wf_rmse[idx] = np.sqrt(e_wf_rmse[idx] / len(samples))
            t_mean[idx] = t_mean[idx] / len(samples)
        data = np.vstack((n_mean, n_max, e_wf_mae, e_dm_mae, e_wf_rmse, e_dm_rmse, t_mean)).T
        np.savetxt(filename, data, fmt='%.8e')

    @classmethod
    def get_embedding(cls, hf, *args, **kwargs):
        return vayesta.ewf.EWF(hf, *args, **kwargs)

    @classmethod
    def add_tests(cls, testset, solver, basis, petas, **kwargs):
        kwargs['min_atoms'] = kwargs.get('min_atoms', 2)
        for key, mol in testset.loop(basis=basis, **kwargs):
            for peta in petas:
                cls.add_test(mol, key, solver, peta)

    @classmethod
    def add_test(cls, mol, key, solver, peta):

        name = 'test_%s_%d' % (key, 10*peta)
        eta = 10.0**(-peta)

        def test(self):
            print("Testing system %s" % key)
            # --- Mean-field
            hf = _run_hf(mol)
            self.assertTrue(hf.converged)
            # --- Benchmark
            if solver == 'MP2':
                cc = _run_mp2(hf)
            elif solver == 'CCSD':
                cc = _run_ccsd(hf)
                self.assertTrue(cc.converged)
            else:
                raise ValueError
            # --- Embedding
            t0 = timer()
            emb = self.get_embedding(hf, solver=solver, bath_options=dict(threshold=eta))
            emb.kernel()
            time = timer() - t0
            ncluster_mean = emb.get_mean_cluster_size()
            ncluster_max = emb.get_max_cluster_size()
            energy_dm_error = emb.get_dm_energy() - cc.e_tot
            energy_wf_error = emb.get_proj_energy() - cc.e_tot
            self.results[peta][key] = Result(ncluster_mean, ncluster_max, energy_dm_error, energy_wf_error, time)

        print("Adding test %s" % name)
        setattr(cls, name, test)

class Test_W411_DZ(Test_TestSet):
    pass

class Test_W411_DZ_project_dmet(Test_TestSet):

    @classmethod
    def get_embedding(cls, hf, *args, **kwargs):
        kwargs['bath_options']['project_dmet'] = 'full'
        return vayesta.ewf.EWF(hf, *args, **kwargs)


if __name__ == '__main__':
    print('Running %s' % __file__)

    petas = np.arange(3.0, 9.1, 0.5)
    Test_W411_DZ.add_tests(gmtkn55.W411, 'MP2', 'cc-pvdz', petas)
    Test_W411_DZ_project_dmet.add_tests(gmtkn55.W411, 'MP2', 'cc-pvdz', petas)

    unittest.main()
