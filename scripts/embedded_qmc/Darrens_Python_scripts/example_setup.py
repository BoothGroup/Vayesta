#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 19 17:00:29 2021

@author: darrenlean
"""

import numpy
from cisd_coeff import Hamiltonian
from hubbard_rotate import transform, make_integrals

'''
2D Hubbard model example
'''

# Working directory
file_name = "hubbard 3x4 benchmark U=6"

# First, set up the Hamiltonian
nx, ny = 3, 4
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
ham.to_pickle(file_name + "/hamiltonian.pkl")

# write fcidump for M7 to load for FCIQMC
ham.write_fcidump(file_name + "/FCIDUMP")

ham = Hamiltonian()
ham.from_pickle(file_name + "/hamiltonian.pkl")