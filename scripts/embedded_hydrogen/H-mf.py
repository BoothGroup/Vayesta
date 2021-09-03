import numpy as np
from numpy import einsum
import numpy as np
from numpy import einsum
import vayesta
import vayesta.ewf
import vayesta.lattmod
import pyscf.tools
import pyscf.tools.ring
from functools import reduce
import numpy
from pyscf import gto, scf, ao2mo, fci

import vayesta
import vayesta.ewf
import vayesta.lattmod

from pyscf import lib

from scmf import SelfConsistentMF

# Hydrogen chain calculation

nsite = 12 # 10 atom H chain
nimp = 4# Size of atomic fragments
doping =0
boundary_cond = ('PBC')

R_min = 0.60 # Minimum symmetric stretching [A]
R_max = 2.2# Maximum symmetric stretching [A]
R_step = 0.05 # Resolution


R_range = np.arange(R_min, R_max+R_step, R_step)


Ridx = 0
for R in R_range:
    

    ring = pyscf.tools.ring.make(nsite, R)
    #print(ring)
    atom_config = [('H %f %f %f') % xyz for xyz in ring]

    # Create atomic chain configuration along x axis
    #for aidx, atom in enumerate(range(nsite)):
    #    atom_config.append(['H', (R*(aidx+1), 0, 0)])
    ##print(['H', (R*(aidx+1), 0, 0)])
    #print(atom_config)

    # Use minimal basis set (single 1s orbital per atom)
    mol = gto.Mole(atom=atom_config, basis='STO-6G', verbose=3, output='pyscf.out').build()
    # Run Hartree-Fock mean-field calculation

    mf = scf.RHF(mol)
    mf.conv_tol = 1.e-13

    mf.run(verbose=1)
    # Run Brueckner self-consistency
    print('Symmetric Hydrogen Ring stretching')
    print('----------------------------------')
    print('Radius [A]', R)
    print('HF energy [Ha]', mf.e_tot)
    print('HF 1-body DM main diag.')
    print(np.diag(mf.make_rdm1()))
    print('HF 1-body DM 1st diag.')
    print(np.diag(mf.make_rdm1(), k=1))
    
    
    e_mf = mf.e_tot/mf.mol.nelectron

    with open('H-energies-dm-hf.txt', 'a') as f:
        if Ridx == 0:
            f.write("%12s %16s %16s %16s %16s \n" % ("#", "nsites", "nimps", "doping", "boundary"))
            f.write("%12s  %16d  %16d %16d %16s \n" % ("# ", (nsite), (nimp), (doping), str(boundary_cond)))
            f.write("%12s  %16s \n" % ("# Radius [A]", "Hartree-Fock"))
        f.write('%12.3f  %16.10f  \n' % (R, e_mf))



    Ridx += 1
