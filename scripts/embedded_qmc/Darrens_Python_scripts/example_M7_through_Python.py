#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 17:06:00 2021

@author: darrenlean
"""

import numpy
from cisd_coeff import Hamiltonian
from hubbard_rotate import transform, make_integrals
from M7_config_yaml_helper import M7_config_to_dict
from subprocess import Popen, PIPE

'''
2D Hubbard model example
'''

# Working directory
directory = "test_run_M7"
path_to_M7 = '/Users/darrenlean/Documents/UROP/Code/M7'

# First, set up the Hamiltonian
nx, ny = 2, 4
nsite = nx * ny
nelec = nsite # half-filling
u = 6.0
h0 = 0
h1e, eri = make_integrals(nx, ny, u)

# We want to work in the canonical basis
e, v = numpy.linalg.eigh(h1e)
h1e, eri = transform(h1e, eri, v)

# Hamiltonian() is a class that can help to save (load) Hamiltonian into (from) files
ham = Hamiltonian()
ham.from_arrays(h0, h1e, eri, nelec)

# pickle the hamiltonian for analysis 
ham.to_pickle(directory + "/hamiltonian.pkl")

# write fcidump for M7 to load for FCIQMC
ham.write_fcidump(directory + "/FCIDUMP_test")

yaml_file = 'M7_settings.yaml'

M7_config_obj = M7_config_to_dict(path_to_M7)
#Make the changes on M7 config you want here, for example:
M7_config_obj.M7_config_dict['wavefunction']['nw_init'] = 3
M7_config_obj.M7_config_dict['wavefunction']['load_balancing']['period'] = 100
M7_config_obj.M7_config_dict['wavefunction']['load_balancing']['nnull_updates_deactivate'] = 8
M7_config_obj.M7_config_dict['propagator']['nw_target'] = 1000
M7_config_obj.M7_config_dict['av_ests']['delay'] = 100
M7_config_obj.M7_config_dict['av_ests']['ncycle'] = 100
M7_config_obj.M7_config_dict['av_ests']['ref_excits']['max_exlvl'] = 2
M7_config_obj.M7_config_dict['av_ests']['ref_excits']['archive']['save'] = 'yes'
M7_config_obj.M7_config_dict['archive']['save_path'] = 'M7.test.h5'
M7_config_obj.M7_config_dict['hamiltonian']['fcidump']['path'] = 'FCIDUMP_test'
M7_config_obj.M7_config_dict['stats']['path'] = 'M7.test.stats'

M7_config_obj.write_yaml(directory + '/' + yaml_file)

process = Popen('/usr/local/bin/mpirun -np 1 ' + path_to_M7 + '/build/src/release ' + yaml_file, stdout=PIPE, stderr=PIPE, shell=True, cwd=directory)
stdout, stderr = process.communicate()
stdout = stdout.decode("utf-8")
stderr = stderr.decode("utf-8")

if stderr == "":
    print("Successful M7 run!")
else:
    print("Error from M7... printing the log:")
    print(stdout)
    print(stderr)