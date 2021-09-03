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

def get_e_dmet(mf,f):
    """DMET (projected DM) energy of fragment f."""
    c = f.c_active
    p = f.get_fragment_projector(c)
    #print(c.shape)
    #print(p.shape)
    #print(f.results.dm1.shape)
    #print(f.results.dm2.shape)
    #print(mf._eri.shape)
    
    n = mf.mol.nelectron
    eri = ao2mo.addons.restore('s1', mf._eri, n).reshape(n, n, n, n)
    print(eri.shape)
    pdm1 = np.einsum('ix,xj,ai,bj->ab', p, f.results.dm1, c, c, optimize=True)
    pdm2 = np.einsum('ix,xjkl,ai,bj,ck,dl->abcd', p, f.results.dm2, c, c, c, c, optimize=True)
    e1 = np.einsum('ij,ij->', mf.get_hcore(), pdm1, optimize=True)
    e2 = np.einsum('ijkl,ijkl->', eri, pdm2,optimize=True)/2
    return e1 + e2

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
    

    fci_dmet = vayesta.dmet.DMET(mf, fragment_type='Lowdin-AO', solver='FCI', make_rdm1=True, make_rdm2=True, charge_consistent=False)
    for s in range(0, nsite, nimp):
        f =fci_dmet.make_atom_fragment(list(range(s, s+nimp)))
    #f_dmet = fci_dmet.make_atom_fragment(list(range(nimp)))
    #frags_dmet = f_dmet.make_tsymmetric_fragments(tvecs=tvecs)
    
    fci_dmet.kernel()
    assert (len(fci_dmet.fragments) == nsite//nimp)
    for f in fci_dmet.fragments:
        assert f.results.converged
        
    dmet_e_tot = fci_dmet.e_tot / mol.nelectron
    dmet_docc = np.zeros((nsite,))
    for f, s in enumerate(range(0, nsite, nimp)):
        imp = fci_dmet.fragments[f].c_active[s:s+nimp]
        d = einsum("ijkl,xi,xj,xk,xl->x", fci_dmet.fragments[f].results.dm2, *(4*[imp]))/2
        dmet_docc[s:s+nimp] = d
    dmet_converged = fci_dmet.converged
    


    with open('H-energies-dm-vdmet.txt', 'a') as f:
        if Ridx == 0:
            f.write("%12s %16s %16s %16s %16s \n" % ("#", "nsites", "nimps", "doping", "boundary"))
            f.write("%12s  %16d  %16d %16d %16s \n" % ("# ", (nsite), (nimp), (doping), str(boundary_cond)))
            f.write("%12s  %16s \n" % ("# Radius [A]", "vdmet"))
        f.write('%12.3f  %16.10f  \n' % (R, dmet_e_tot))
    with open('H-energies-amp-vdmet.txt', 'a') as f:
        if Ridx == 0:
            f.write("%12s %16s %16s %16s %16s \n" % ("#", "nsites", "nimps", "doping", "boundary"))
            f.write("%12s  %16d  %16d %16d %16s \n" % ("# ", (nsite), (nimp), (doping), str(boundary_cond)))
            f.write("%12s  %16s \n" % ("# Radius [A]", "vdmet"))
        f.write('%12.3f  %16.10f  \n' % (R, dmet_e_tot))

    s = nimp // 2
    with open('H-docc-vdmet.txt', 'a') as f:
        if Ridx == 0:
            f.write("%12s %16s %16s %16s %16s \n" % ("#", "nsites", "nimps", "doping", "boundary"))
            f.write("%12s  %16d  %16d %16d %16s \n" % ("# ", (nsite), (nimp), (doping), str(boundary_cond)))
            f.write("%12s  %16s \n" % ("# Radius [A]", "vdmet"))
        f.write('%12.3f  %16.10f \n' % (R, dmet_docc[s]))
        
    with open('H-convergence-vdmet.txt', 'a') as f:
        if Ridx == 0:
            f.write("%12s %16s %16s %16s %16s \n" % ("#", "nsites", "nimps", "doping", "boundary"))
            f.write("%12s  %16d  %16d %16d %16s \n" % ("# ", (nsite), (nimp), (doping), str(boundary_cond)))
            f.write("%12s  %16s \n" % ("# Radius [A]", "vdmet"))
        f.write('%12.3f %16.10s %d \n' % (R,  str(dmet_converged), 1))

    Ridx += 1
