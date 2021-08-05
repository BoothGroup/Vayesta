#!/usr/bin/env python
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

'''
A simple example to run CISD calculation.
'''

import numpy as np
import pyscf
import pyscf.ci

mol = pyscf.M(
    atom = 'H 0 0 0; F 0 0 1.1',
    basis = 'ccpvdz')

mf = mol.HF().run()
ci = mf.CISD().run()

norm = np.linalg.norm(ci.ci)
print(norm)

np.dot(ci.ci.flatten(), ci.ci.flatten())

c0, c1, c2 = ci.cisdvec_to_amplitudes(ci.ci)

s = pyscf.ci.cisd.dot(ci.ci, ci.ci, nmo=ci.nmo, nocc=ci.nocc)
#print(s)
#1/0

#print(c0)
print(c0.shape)
print(c1.shape)
print(c2.shape)
print(c0.size)
print(c1.size)
print(c2.size)

vec = np.hstack((c0, c1.flatten(), c2.flatten()))
norm = np.linalg.norm(vec)
print(norm)
