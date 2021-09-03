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

def get_energy(mf, frags):
    # DMET energy
    e = np.asarray([get_e_dmet(mf, f) for f in frags])
    with open('Fragment_energies.txt', 'a') as f:
        f.write(str(mf.mol.atom)+'\n'+str(e)+'\n')
    print(e)
    assert abs(e.min()-e.max()) < 1e-6
    e = np.sum(e)
    return e

# Hydrogen chain calculation

nsite = 12 # 10 atom H chain
nimp = 4# Size of atomic fragments
doping =0
boundary_cond = ('PBC')

R_min = 0.6 # Minimum symmetric stretching [A]
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
    

    ewf = vayesta.ewf.EWF(mf, solver='FCI', fragment_type='Lowdin-AO', bath_type=None, make_rdm1=True, make_rdm2=True)
    f = ewf.make_atom_fragment(list(range(nsite)))

    # --- One-shot calculation:
    ewf.kernel()
    e_fci_ewf = ewf.get_total_energy() / mol.nelectron
    e_fci = ewf.get_dmet_energy() / mol.nelectron
    
    docc_fci = np.einsum("ijkl,i,j,k,l->", f.results.dm2, *(4*[f.c_active[0]]))/2
        
    with open('H-energies-dm-fci.txt', 'a') as f:
        if Ridx == 0:
            f.write("%12s %16s %16s %16s %16s \n" % ("#", "nsites", "nimps", "doping", "boundary"))
            f.write("%12s  %16d  %16d %16d %16s \n" % ("# ", (nsite), (nimp), (doping), str(boundary_cond)))
            f.write("%12s  %16s \n" % ("# Radius [A]", "fci"))
        f.write('%12.3f  %16.10f  \n' % (R, e_fci))
    
    s = nsite //2
    with open('H-docc-fci.txt', 'a') as f:
        if Ridx == 0:
            f.write("%12s %16s %16s %16s %16s \n" % ("#", "nsites", "nimps", "doping", "boundary"))
            f.write("%12s  %16d  %16d %16d %16s \n" % ("# ", (nsite), (nimp), (doping), str(boundary_cond)))
            f.write("%12s  %16s \n" % ("# Radius [A]", "fci"))
        f.write('%12.3f  %16.10f  \n' % (R, docc_fci))



    Ridx += 1
